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

    def forward(self, bev, mode='early'):
        features = self.encoder(bev)
        if mode == 'intermediate':
            features = features.amax(dim=0, keepdim=True)
        return self.heads(features)


def targets(boxes, cfg, shape):
    h, w = shape
    classes, b, r = cfg['classes'], cfg['bounds'], cfg['resolution']
    heat = np.zeros((len(classes),h,w),np.float32)
    reg = np.zeros((len(classes),8,h,w),np.float32)
    mask = np.zeros((len(classes),h,w),bool)
    for box in boxes:
        x, y = (box.x-b[0])/r, (box.y-b[1])/r
        ix, iy, c = int(x), int(y), classes.index(box.label)
        if not (0 <= ix < w and 0 <= iy < h):
            continue
        if mask[c,iy,ix]:
            raise ValueError('Two same-class centers occupy one grid cell; reduce dataset.resolution')
        heat[c,iy,ix] = 1
        mask[c,iy,ix] = True
        reg[c,:,iy,ix] = [x-ix,y-iy,box.z,math.log(box.length),math.log(box.width),
                          math.log(box.height),math.sin(box.yaw),math.cos(box.yaw)]
    return heat, reg, mask


def loss(outputs, target):
    logits, regression = outputs
    heat, reg, mask = [torch.as_tensor(v,device=logits.device) for v in target]
    p = logits[0].sigmoid().clamp(1e-5,1-1e-5)
    positive = heat == 1
    focal = -(torch.where(positive,(1-p)**2*p.log(),p**2*(1-p).log())).sum()/positive.sum().clamp(min=1)
    prediction = regression[0].reshape_as(reg).permute(0,2,3,1)
    expected = reg.permute(0,2,3,1)
    regression_loss = F.smooth_l1_loss(prediction[mask],expected[mask]) if mask.any() else regression.sum()*0
    return focal + regression_loss


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
