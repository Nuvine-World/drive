"""Compact two-scale BEV center detector used by all fusion strategies."""

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.data.loader import BoundingBox3D
from src.data.coinfra import in_bounds
from src.metrics.evaluation import nms, paired_iou3d

# Per slot: x/y offset, z, log length/width/height, sin/cos yaw, IoU logit.
BOX_CHANNELS = 8
REGRESSION_CHANNELS = BOX_CHANNELS + 1


class TargetCollisionError(ValueError):
    """More same-class centers share one output cell than there are slots."""


class SeparableResidual(nn.Module):
    """Large-context residual block with inexpensive depthwise convolution."""

    def __init__(self, channels, dilation):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=dilation,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(1, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(1, channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, value):
        return self.activation(value + self.block(value))


class ResidualBlock(nn.Module):
    """Dense residual block for the coarse, large-context stage."""

    def __init__(self, channels, dilation):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels, channels, 3, padding=dilation, dilation=dilation, bias=False
            ),
            nn.GroupNorm(1, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                channels, channels, 3, padding=dilation, dilation=dilation, bias=False
            ),
            nn.GroupNorm(1, channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, value):
        return self.activation(value + self.block(value))


class BEVEncoder(nn.Module):
    """Two-scale encoder whose output is the transmitted stride-2 feature map.

    A cheap stride-2 path keeps small-object detail; a dense stride-4 path with
    twice the channels adds capacity and a receptive field longer than a bus.
    The ReLU output makes zero mean "no evidence", which is the natural
    identity for max fusion and keeps empty space compressible.
    """

    def __init__(self, hidden, in_channels=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.SiLU(inplace=True),
        )
        self.fine = nn.Sequential(
            SeparableResidual(hidden, dilation=1),
            SeparableResidual(hidden, dilation=2),
        )
        self.down = nn.Sequential(
            nn.Conv2d(hidden, 2 * hidden, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, 2 * hidden),
            nn.SiLU(inplace=True),
        )
        self.coarse = nn.Sequential(
            ResidualBlock(2 * hidden, dilation=1),
            ResidualBlock(2 * hidden, dilation=2),
            ResidualBlock(2 * hidden, dilation=4),
        )
        self.lateral = nn.Conv2d(2 * hidden, hidden, 1, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.ReLU(inplace=True),
        )

    def forward(self, bev):
        fine = self.fine(self.stem(bev))
        coarse = self.lateral(self.coarse(self.down(fine)))
        coarse = F.interpolate(
            coarse, size=fine.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.fuse(fine + coarse)


class BEVDetector(nn.Module):
    def __init__(
        self, classes, hidden, objects_per_cell=1, output_stride=2, in_channels=4
    ):
        super().__init__()
        self.classes = classes
        self.objects_per_cell = objects_per_cell
        if output_stride != 2:
            raise ValueError("BEVDetector currently supports output_stride=2")
        self.output_stride = output_stride
        self.encoder = BEVEncoder(hidden, in_channels)
        # Through the stride-4 path: 135 input cells (54 m at 0.4 m/cell).
        self.receptive_field_cells = 135
        slots = len(classes) * objects_per_cell
        self.heatmap = nn.Conv2d(hidden, slots, 1)
        # The last channel of each slot predicts the 3D IoU of its own box and
        # rescales the detection score, so well-localized boxes rank first.
        self.regression = nn.Conv2d(hidden, REGRESSION_CHANNELS * slots, 1)
        # A learned, content-dependent utility map used by budget-conditioned
        # sparse fusion. One score is produced for every node and BEV cell.
        self.utility = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.constant_(self.heatmap.bias, -2.19)

    def heads(self, features):
        return self.heatmap(features), self.regression(features)

    def adaptive_features(self, features, budget, selector="learned", generators=None):
        """Fuse a dense reference with top-utility collaborator cells.

        The first node is the reference observation. Each additional node may
        transmit only ``budget`` of its spatial cells. Feature values remain
        differentiable; the discrete top-k decision is intentionally explicit
        so evaluation has a deterministic byte budget.
        """
        if not 0 < budget <= 1:
            raise ValueError("adaptive budget must be in (0, 1]")
        if features.ndim != 5:
            raise ValueError("adaptive fusion expects [batch, nodes, channels, H, W]")
        batch, nodes, channels, height, width = features.shape
        if nodes == 1:
            empty = torch.zeros(
                batch, 0, height, width, dtype=torch.bool, device=features.device
            )
            return features[:, 0], empty
        collaborators = features[:, 1:]
        if selector == "learned":
            flat = collaborators.reshape(batch * (nodes - 1), channels, height, width)
            scores = self.utility(flat).reshape(batch, nodes - 1, height * width)
        elif selector == "energy":
            scores = (
                collaborators.float()
                .abs()
                .mean(dim=2)
                .reshape(batch, nodes - 1, height * width)
            )
        elif selector == "random":
            if generators is None:
                scores = torch.rand(
                    batch, nodes - 1, height * width, device=features.device
                )
            else:
                if len(generators) != batch:
                    raise ValueError("one random generator is required per batch item")
                scores = torch.stack(
                    [
                        torch.rand(
                            nodes - 1,
                            height * width,
                            device=features.device,
                            generator=generator,
                        )
                        for generator in generators
                    ]
                )
        else:
            raise ValueError(f"unknown adaptive selector: {selector}")
        count = max(1, min(height * width, math.ceil(budget * height * width)))
        selected = scores.topk(count, dim=-1).indices
        hard_mask = torch.zeros_like(scores)
        hard_mask.scatter_(-1, selected, 1.0)
        # Straight-through relaxation: forward communication is exactly top-k,
        # while every utility score receives a useful training gradient.
        # Only the proposed learned selector needs the surrogate gradient.
        if selector == "learned":
            soft_mask = scores.sigmoid()
            gate = hard_mask + soft_mask - soft_mask.detach()
        else:
            gate = hard_mask
        mask = hard_mask.bool().reshape(batch, nodes - 1, height, width)
        # Selection methods differ only in which cells they choose. All three
        # transmit and fuse the same unscaled encoder features.
        sparse = collaborators * gate.reshape(batch, nodes - 1, 1, height, width)
        # Unsent cells must not take part in max fusion. Filling them with the
        # lowest finite value (not zero) leaves the receiver's features exactly
        # unchanged there, whatever the encoder's output range.
        sparse = sparse.masked_fill(~mask[:, :, None], torch.finfo(sparse.dtype).min)
        fused = torch.maximum(features[:, 0], sparse.amax(dim=1))
        return fused, mask

    def forward(self, bev, mode="early"):
        base_mode, _, budget_text = mode.partition("@")
        if base_mode in {"intermediate", "adaptive"} and bev.ndim == 5:
            batch, nodes, channels, height, width = bev.shape
            features = self.encoder(bev.reshape(batch * nodes, channels, height, width))
            features = features.reshape(batch, nodes, *features.shape[1:])
            if base_mode == "adaptive":
                budget = float(budget_text) if budget_text else 0.1
                features, _ = self.adaptive_features(features, budget)
            else:
                features = features.amax(dim=1)
        else:
            features = self.encoder(bev)
            if base_mode == "intermediate":
                features = features.amax(dim=0, keepdim=True)
            elif base_mode == "adaptive":
                if features.ndim != 4:
                    raise ValueError("unexpected adaptive feature shape")
                budget = float(budget_text) if budget_text else 0.1
                features, _ = self.adaptive_features(features[None], budget)
        return self.heads(features)


def _grid_resolution(cfg, shape):
    """Return x/y metres per output cell for an arbitrary prediction grid."""
    h, w = shape
    bounds = cfg["bounds"]
    return (bounds[3] - bounds[0]) / w, (bounds[4] - bounds[1]) / h


def output_shape(cfg, output_stride):
    """Prediction-grid shape for the configured bounds, resolution, and stride."""
    b, r = cfg["bounds"], cfg["resolution"]
    height = math.ceil((b[4] - b[1]) / r)
    width = math.ceil((b[3] - b[0]) / r)
    return (
        (height + output_stride - 1) // output_stride,
        (width + output_stride - 1) // output_stride,
    )


def grid_cell_size(cfg, output_stride):
    """Metres per prediction-grid cell along x and y."""
    return _grid_resolution(cfg, output_shape(cfg, output_stride))


def target_cell(box, cfg, shape):
    """Return ``(class_index, ix, iy, x, y)`` on the prediction grid, or None."""
    h, w = shape
    b = cfg["bounds"]
    rx, ry = _grid_resolution(cfg, shape)
    x, y = (box.x - b[0]) / rx, (box.y - b[1]) / ry
    ix, iy = int(x), int(y)
    if not (0 <= ix < w and 0 <= iy < h):
        return None
    return cfg["classes"].index(box.label), ix, iy, x, y


def _gaussian_radius(height, width, minimum_overlap=0.1):
    """CenterNet radius for a box measured in output-grid cells."""
    a1, b1 = 1.0, height + width
    c1 = width * height * (1 - minimum_overlap) / (1 + minimum_overlap)
    radius1 = (b1 + math.sqrt(max(0.0, b1 * b1 - 4 * a1 * c1))) / 2
    a2, b2 = 4.0, 2 * (height + width)
    c2 = (1 - minimum_overlap) * width * height
    radius2 = (b2 + math.sqrt(max(0.0, b2 * b2 - 4 * a2 * c2))) / 2
    a3, b3 = 4 * minimum_overlap, -2 * minimum_overlap * (height + width)
    c3 = (minimum_overlap - 1) * width * height
    radius3 = (b3 + math.sqrt(max(0.0, b3 * b3 - 4 * a3 * c3))) / (2 * a3)
    return max(0, int(min(radius1, radius2, radius3)))


def _draw_gaussian(heatmap, center, radius):
    diameter = 2 * radius + 1
    sigma = max(diameter / 6, 1e-6)
    coordinates = np.arange(diameter, dtype=np.float32) - radius
    gaussian = np.exp(
        -(coordinates[:, None] ** 2 + coordinates[None, :] ** 2) / (2 * sigma**2)
    )
    x, y = center
    height, width = heatmap.shape
    left, right = min(x, radius), min(width - x - 1, radius)
    top, bottom = min(y, radius), min(height - y - 1, radius)
    patch = heatmap[y - top : y + bottom + 1, x - left : x + right + 1]
    kernel = gaussian[
        radius - top : radius + bottom + 1,
        radius - left : radius + right + 1,
    ]
    np.maximum(patch, kernel, out=patch)


def targets(boxes, cfg, shape, sample_id=None):
    h, w = shape
    classes = cfg["classes"]
    rx, ry = _grid_resolution(cfg, shape)
    slots = cfg.get("objects_per_cell", 1)
    heat = np.zeros((len(classes), slots, h, w), np.float32)
    reg = np.zeros((len(classes), slots, 8, h, w), np.float32)
    mask = np.zeros((len(classes), slots, h, w), bool)
    occupied = {}
    for box in boxes:
        cell = target_cell(box, cfg, shape)
        if cell is None:
            continue
        c, ix, iy, x, y = cell
        key = c, iy, ix
        colocated = occupied.setdefault(key, [])
        if len(colocated) >= slots:
            previous = colocated[0]
            resolution_text = f"{rx:g}" if math.isclose(rx, ry) else f"({rx:g}, {ry:g})"
            raise TargetCollisionError(
                f"More than {slots} same-class centers occupy one grid cell: "
                f"sample={sample_id!r}, class={box.label!r}, cell=(x={ix}, y={iy}), "
                f"resolution={resolution_text}, first_center=({previous.x:.6f}, {previous.y:.6f}), "
                f"second_center=({box.x:.6f}, {box.y:.6f}), "
                f"delta=({abs(previous.x-box.x):.6f}, {abs(previous.y-box.y):.6f})"
            )
        slot = len(colocated)
        colocated.append(box)
        radius = max(1, _gaussian_radius(box.width / ry, box.length / rx))
        _draw_gaussian(heat[c, slot], (ix, iy), radius)
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


def regression_boxes(values, cell_size):
    """``[N, 8]`` regression rows as ``[x, y, z, l, w, h, yaw]`` within a cell.

    Rows share their cell, so only offsets within it matter for IoU.
    """
    values = values.detach().double().cpu().numpy()
    sizes = np.exp(np.clip(values[:, 3:6], -4, 5))
    return np.column_stack(
        (
            values[:, 0] * cell_size[0],
            values[:, 1] * cell_size[1],
            values[:, 2],
            sizes,
            np.arctan2(values[:, 6], values[:, 7]),
        )
    )


def loss(outputs, target, cell_size=(1.0, 1.0)):
    """Class-balanced focal heatmap loss, L1 box regression, and IoU quality.

    ``cell_size`` is the prediction-grid cell in metres along x and y; it
    converts offset errors to metres for the IoU-quality target.
    """
    logits, regression = outputs
    heat, reg, mask = [torch.as_tensor(v, device=logits.device) for v in target]
    if heat.ndim == 4:
        heat, reg, mask = heat[None], reg[None], mask[None]
    batch, classes, slots, height, width = heat.shape
    # Work in float32 log space. Under AMP an fp16 sigmoid rounds to 1.0 above
    # a logit of about 8, and a probability clamp then zeroes the gradient of
    # exactly the confident false positives the loss should correct.
    x = logits.float().reshape(batch, classes, slots, height, width)
    p = x.sigmoid()
    positive = heat.eq(1)
    negative = heat.lt(1)
    negative_weight = (1 - heat).pow(4)
    # Normalize across the batch per class so abundant cars cannot drown out
    # the rarer truck, bus, bicycle, and pedestrian positives.
    axes = (0, 2, 3, 4)
    positive_loss = ((1 - p).pow(2) * F.logsigmoid(x) * positive).sum(dim=axes)
    negative_loss = (p.pow(2) * F.logsigmoid(-x) * negative_weight * negative).sum(
        dim=axes
    )
    positive_count = positive.sum(dim=axes)
    focal_by_class = -(positive_loss + negative_loss) / positive_count.clamp(min=1)
    focal_weights = torch.where(
        positive_count > 0,
        torch.ones_like(focal_by_class),
        torch.full_like(focal_by_class, 0.05),
    )
    focal = (focal_by_class * focal_weights).sum() / focal_weights.sum()
    prediction = regression.float().reshape(
        batch, classes, slots, REGRESSION_CHANNELS, height, width
    )
    prediction = prediction.permute(0, 1, 2, 4, 5, 3)
    boxes, quality = prediction[..., :BOX_CHANNELS], prediction[..., BOX_CHANNELS]
    expected = reg.permute(0, 1, 2, 4, 5, 3)
    regression_losses = []
    for class_index in range(classes):
        selected = mask[:, class_index]
        if selected.any():
            # L1, as in CenterNet/CenterPoint: smooth L1 with beta=1 is
            # quadratic for the sub-cell offsets and weakens the final refinement.
            regression_losses.append(
                F.l1_loss(
                    boxes[:, class_index][selected],
                    expected[:, class_index][selected],
                )
            )
    regression_loss = (
        torch.stack(regression_losses).mean()
        if regression_losses
        else prediction.sum() * 0
    )
    if mask.any():
        selected_boxes = boxes[mask]
        selected_expected = expected[mask]
        if (
            torch.isfinite(selected_boxes).all()
            and torch.isfinite(selected_expected).all()
        ):
            with torch.no_grad():
                iou = paired_iou3d(
                    regression_boxes(selected_boxes, cell_size),
                    regression_boxes(selected_expected, cell_size),
                )
            quality_loss = F.binary_cross_entropy_with_logits(
                quality[mask],
                torch.as_tensor(iou, dtype=quality.dtype, device=quality.device),
            )
        else:
            # Preserve a nonfinite result so the training loop can retry an
            # AMP-overflowed batch in fp32 without passing NaNs into GEOS.
            quality_loss = selected_boxes.sum() * 0 + selected_expected.sum() * 0
    else:
        quality_loss = quality.sum() * 0
    return focal + regression_loss + quality_loss


def decode(outputs, cfg, score_threshold=None):
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
    reg = regression[0].float().reshape(classes * slots, REGRESSION_CHANNELS, h * w)
    selected_channels = torch.div(indices, h * w, rounding_mode="floor")
    selected_cells = indices.remainder(h * w)
    selected_regression = (
        reg[selected_channels, :, selected_cells].detach().cpu().numpy()
    )
    # IoU-rectified confidence: heat^(1-a) * predicted_iou^a.
    weight = settings.get("iou_score_weight", 0.0)
    if weight:
        predicted_iou = torch.sigmoid(
            torch.from_numpy(selected_regression[:, BOX_CHANNELS])
        ).numpy()
        rectified = (
            values.detach().cpu().numpy() ** (1 - weight) * predicted_iou**weight
        )
    else:
        rectified = values.detach().cpu().numpy()
    threshold = (
        settings.get("ap_score_threshold", settings["score_threshold"])
        if score_threshold is None
        else score_threshold
    )
    boxes = []
    for score, index, v in zip(
        rectified.tolist(),
        indices.detach().cpu().tolist(),
        selected_regression,
    ):
        # Written as a negated >= so a non-finite score is also dropped.
        if not score >= threshold:
            continue
        channel, cell = divmod(index, h * w)
        c, slot = divmod(channel, slots)
        iy, ix = divmod(cell, w)
        v = v[:BOX_CHANNELS]
        if not np.isfinite(v).all():
            continue
        rx, ry = _grid_resolution(d, (h, w))
        x = d["bounds"][0] + (ix + float(v[0])) * rx
        y = d["bounds"][1] + (iy + float(v[1])) * ry
        if not in_bounds(x, y, float(v[2]), d["bounds"]):
            continue
        size = np.exp(np.clip(v[3:6], -4, 5))
        boxes.append(
            BoundingBox3D(
                x, y, float(v[2]), *size, math.atan2(v[6], v[7]), d["classes"][c], score
            )
        )
    return nms(boxes, settings["nms_iou"], settings["max_detections"])
