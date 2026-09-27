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
    def __init__(self, classes, hidden):
        super().__init__()
        self.classes = classes
        self.encoder = nn.Sequential(nn.Conv2d(3,hidden,3,padding=1), nn.ReLU(),
                                     nn.Conv2d(hidden,hidden,3,padding=1), nn.ReLU())
        self.heatmap = nn.Conv2d(hidden,len(classes),1)
        self.regression = nn.Conv2d(hidden,8*len(classes),1)
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
    classes, b, r = cfg['classes'], cfg['bounds'], cfg['resolution']
    heat = np.zeros((len(classes),h,w),np.float32)
    reg = np.zeros((len(classes),8,h,w),np.float32)
    mask = np.zeros((len(classes),h,w),bool)
    occupied = {}
    for box in boxes:
        x, y = (box.x-b[0])/r, (box.y-b[1])/r
        ix, iy, c = int(x), int(y), classes.index(box.label)
        if not (0 <= ix < w and 0 <= iy < h):
            continue
        key = c, iy, ix
        if key in occupied:
            previous = occupied[key]
            raise ValueError(
                "Two same-class centers occupy one grid cell: "
                f"sample={sample_id!r}, class={box.label!r}, cell=(x={ix}, y={iy}), "
                f"resolution={r}, first_center=({previous.x:.6f}, {previous.y:.6f}), "
                f"second_center=({box.x:.6f}, {box.y:.6f}), "
                f"delta=({abs(previous.x-box.x):.6f}, {abs(previous.y-box.y):.6f})"
            )
        occupied[key] = box
        heat[c,iy,ix] = 1
        mask[c,iy,ix] = True
        reg[c,:,iy,ix] = [x-ix,y-iy,box.z,math.log(box.length),math.log(box.width),
                          math.log(box.height),math.sin(box.yaw),math.cos(box.yaw)]
    return heat, reg, mask


def loss(outputs, target):
    logits, regression = outputs
    heat, reg, mask = [torch.as_tensor(v,device=logits.device) for v in target]
    if heat.ndim == 3:
        heat, reg, mask = heat[None], reg[None], mask[None]
    p = logits.sigmoid().float().clamp(1e-5, 1 - 1e-5)
    positive = heat == 1
    axes = tuple(range(1, heat.ndim))
    focal = -torch.where(positive, (1 - p) ** 2 * p.log(), p**2 * (1 - p).log()).sum(
        dim=axes
    )
    focal = focal / positive.sum(dim=axes).clamp(min=1)
    prediction = regression.float().reshape(
        reg.shape[0], reg.shape[1], 8, *reg.shape[-2:]
    )
    prediction = prediction.permute(0, 1, 3, 4, 2)
    expected = reg.permute(0, 1, 3, 4, 2)
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
    settings, d = cfg['evaluation'], cfg['dataset']
    p = logits[0].sigmoid()
    maxima = F.max_pool2d(p[None],3,stride=1,padding=1)[0]
    scores = torch.where(p == maxima,p,torch.zeros_like(p)).flatten()
    values, indices = scores.topk(min(settings['max_detections'], scores.numel()))
    _, h, w = p.shape
    reg = regression[0].reshape(len(d['classes']),8,h,w).detach().cpu().numpy()
    boxes = []
    for score, index in zip(values.detach().cpu().tolist(),indices.detach().cpu().tolist()):
        if score < settings['score_threshold']:
            continue
        c, cell = divmod(index,h*w)
        iy, ix = divmod(cell,w)
        v = reg[c,:,iy,ix]
        if not np.isfinite(v).all():
            raise ValueError('Nonfinite model output')
        x = d['bounds'][0]+(ix+float(v[0]))*d['resolution']
        y = d['bounds'][1]+(iy+float(v[1]))*d['resolution']
        if not in_bounds(x,y,float(v[2]),d['bounds']):
            continue
        size = np.exp(np.clip(v[3:6],-4,5))
        boxes.append(BoundingBox3D(x,y,float(v[2]),*size,math.atan2(v[6],v[7]),d['classes'][c],score))
    return nms(boxes,settings['nms_iou'],settings['max_detections'])
