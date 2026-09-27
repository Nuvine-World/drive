"""Small trainable BEV center detector, not a PointPillars reproduction."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.data.loader import BoundingBox3D
from src.data.coinfra import in_bounds
from src.metrics.evaluation import nms


class BEVDetector(nn.Module):
    def __init__(self, classes, hidden, objects_per_cell=1):
        super().__init__()
        self.classes = classes
        self.objects_per_cell = objects_per_cell
        self.encoder = nn.Sequential(nn.Conv2d(3,hidden,3,padding=1), nn.ReLU(),
                                     nn.Conv2d(hidden,hidden,3,padding=1), nn.ReLU())
        slots = len(classes) * objects_per_cell
        self.heatmap = nn.Conv2d(hidden, slots, 1)
        self.regression = nn.Conv2d(hidden, 8 * slots, 1)
        nn.init.constant_(self.heatmap.bias, -2.19)

    def heads(self, features):
        return self.heatmap(features), self.regression(features)

    def forward(self, bev, mode="early"):
        if mode == "intermediate" and bev.ndim == 5:
            batch, nodes, channels, height, width = bev.shape
            features = self.encoder(bev.reshape(batch * nodes, channels, height, width))
            features = features.reshape(batch, nodes, *features.shape[1:]).amax(dim=1)
        else:
            features = self.encoder(bev)
            if mode == "intermediate":
                features = features.amax(dim=0, keepdim=True)
        return self.heads(features)


def targets(boxes, cfg, shape, sample_id=None):
    h, w = shape
    classes, b, r = cfg["classes"], cfg["bounds"], cfg["resolution"]
    slots = cfg.get("objects_per_cell", 1)
    heat = np.zeros((len(classes), slots, h, w), np.float32)
    reg = np.zeros((len(classes), slots, 8, h, w), np.float32)
    mask = np.zeros((len(classes), slots, h, w), bool)
    occupied = {}
    for box in boxes:
        x, y = (box.x-b[0])/r, (box.y-b[1])/r
        ix, iy, c = int(x), int(y), classes.index(box.label)
        if not (0 <= ix < w and 0 <= iy < h):
            continue
        key = c, iy, ix
        colocated = occupied.setdefault(key, [])
        if len(colocated) >= slots:
            previous = colocated[0]
            raise ValueError(
                f"More than {slots} same-class centers occupy one grid cell: "
                f"sample={sample_id!r}, class={box.label!r}, cell=(x={ix}, y={iy}), "
                f"resolution={r}, first_center=({previous.x:.6f}, {previous.y:.6f}), "
                f"second_center=({box.x:.6f}, {box.y:.6f}), "
                f"delta=({abs(previous.x-box.x):.6f}, {abs(previous.y-box.y):.6f})"
            )
        slot = len(colocated)
        colocated.append(box)
        heat[c, slot, iy, ix] = 1
        mask[c, slot, iy, ix] = True
        reg[c, slot, :, iy, ix] = [
            x - ix,
            y - iy,
            box.z,
            math.log(box.length),
            math.log(box.width),
            math.log(box.height),
            math.sin(box.yaw),
            math.cos(box.yaw),
        ]
    return heat, reg, mask


def loss(outputs, target):
    logits, regression = outputs
    heat, reg, mask = [torch.as_tensor(v,device=logits.device) for v in target]
    if heat.ndim == 4:
        heat, reg, mask = heat[None], reg[None], mask[None]
    batch, classes, slots, height, width = heat.shape
    p = logits.reshape(batch, classes, slots, height, width)
    p = p.sigmoid().float().clamp(1e-5, 1 - 1e-5)
    positive = heat == 1
    axes = tuple(range(1, heat.ndim))
    focal = -torch.where(positive, (1 - p) ** 2 * p.log(), p**2 * (1 - p).log()).sum(
        dim=axes
    )
    focal = focal / positive.sum(dim=axes).clamp(min=1)
    prediction = regression.float().reshape(batch, classes, slots, 8, height, width)
    prediction = prediction.permute(0, 1, 2, 4, 5, 3)
    expected = reg.permute(0, 1, 2, 4, 5, 3)
    regression_losses = []
    for index in range(len(reg)):
        selected = mask[index]
        regression_losses.append(
            F.smooth_l1_loss(prediction[index][selected], expected[index][selected])
            if selected.any()
            else prediction[index].sum() * 0
        )
    return (focal + torch.stack(regression_losses)).mean()


def decode(outputs, cfg):
    logits, regression = outputs
    settings, d = cfg["evaluation"], cfg["dataset"]
    classes = len(d["classes"])
    slots = d.get("objects_per_cell", 1)
    # Keep thresholding, local-max selection, and box decoding stable under AMP.
    _, _, h, w = logits.shape
    p = logits[0].float().reshape(classes, slots, h, w).sigmoid()
    flat_heatmaps = p.reshape(classes * slots, h, w)
    maxima = F.max_pool2d(flat_heatmaps[None], 3, stride=1, padding=1)[0]
    scores = torch.where(
        flat_heatmaps == maxima, flat_heatmaps, torch.zeros_like(flat_heatmaps)
    ).flatten()
    values, indices = scores.topk(min(settings["max_detections"], scores.numel()))
    reg = regression[0].float().reshape(classes, slots, 8, h, w).detach().cpu().numpy()
    boxes = []
    for score, index in zip(values.detach().cpu().tolist(),indices.detach().cpu().tolist()):
        if score < settings['score_threshold']:
            continue
        channel, cell = divmod(index, h * w)
        c, slot = divmod(channel, slots)
        iy, ix = divmod(cell, w)
        v = reg[c, slot, :, iy, ix]
        if not np.isfinite(v).all():
            raise ValueError('Nonfinite model output')
        x = d['bounds'][0]+(ix+float(v[0]))*d['resolution']
        y = d['bounds'][1]+(iy+float(v[1]))*d['resolution']
        if not in_bounds(x,y,float(v[2]),d['bounds']):
            continue
        size = np.exp(np.clip(v[3:6],-4,5))
        boxes.append(BoundingBox3D(x,y,float(v[2]),*size,math.atan2(v[6],v[7]),d['classes'][c],score))
    return nms(boxes,settings['nms_iou'],settings['max_detections'])
