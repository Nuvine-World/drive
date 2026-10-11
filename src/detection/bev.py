"""DRIVE cooperative BEV detector: one network for every fusion strategy.

A shared encoder turns each node's BEV into the feature map that node would
transmit. The receiver fuses whatever node features it holds (dense, sparse,
or stale) cell by cell, then a shared multi-scale backbone and CenterPoint
heads detect every class. Early fusion and single-node detection are the
one-node case of the same network, and late fusion runs it once per node.
"""

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
FUSIONS = ("attention", "mean", "max")
SELECTORS = ("learned", "random", "energy", "confidence")


class TargetCollisionError(ValueError):
    """More same-class centers share one output cell than there are slots."""


class LayerNorm2d(nn.Module):
    """LayerNorm over the channels of each BEV cell.

    It uses no spatial statistics, so a cell's features do not depend on how
    much of the map is empty, on the batch size after an OOM split, or on how
    many nodes were fused. GroupNorm over a mostly empty BEV map amplifies the
    few occupied cells by an amount that changes with sparsity.
    """

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, value):
        normalized = F.layer_norm(
            value.permute(0, 2, 3, 1),
            (value.shape[1],),
            self.weight,
            self.bias,
            self.eps,
        )
        # Autocast computes layer_norm in fp32; keep activations in the AMP dtype.
        return normalized.permute(0, 3, 1, 2).to(value.dtype)


class ConvNeXtBlock(nn.Module):
    """Large-kernel depthwise residual block with per-cell normalization."""

    def __init__(self, channels, expansion=4, kernel=7):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, kernel, padding=kernel // 2, groups=channels
        )
        self.norm = LayerNorm2d(channels)
        self.expand = nn.Conv2d(channels, expansion * channels, 1)
        self.project = nn.Conv2d(expansion * channels, channels, 1)
        # A small residual scale starts each block near identity (layer scale).
        self.scale = nn.Parameter(torch.full((channels, 1, 1), 0.1))

    def forward(self, value):
        branch = self.project(F.gelu(self.expand(self.norm(self.depthwise(value)))))
        return value + self.scale.to(branch.dtype) * branch


def downsample(in_channels, out_channels):
    return nn.Sequential(
        LayerNorm2d(in_channels), nn.Conv2d(in_channels, out_channels, 2, stride=2)
    )


def detection_head(channels, outputs, bias=None):
    layers = nn.Sequential(
        nn.Conv2d(channels, channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(channels, outputs, 1),
    )
    if bias is not None:
        nn.init.constant_(layers[-1].bias, bias)
    return layers


class NodeEncoder(nn.Module):
    """Shared per-node encoder; its stride-2 output is what a node transmits."""

    def __init__(self, in_channels, channels, blocks):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, stride=2, padding=1),
            LayerNorm2d(channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *(ConvNeXtBlock(channels, expansion=2) for _ in range(blocks))
        )

    def forward(self, bev):
        return self.blocks(self.stem(bev))


class NodeFusion(nn.Module):
    """Per-cell fusion of the node features available to the receiver.

    ``attention`` weighs every available node by a learned score of its
    features, its own point density, and the age of its message, then merges
    that weighted average with the per-cell maximum. ``mean`` (the CoInfra
    baseline) and ``max`` (F-Cooper) are fixed-rule ablations.
    """

    def __init__(self, channels, mode="attention"):
        super().__init__()
        if mode not in FUSIONS:
            raise ValueError(f"fusion must be one of {FUSIONS}")
        self.mode = mode
        if mode == "attention":
            self.score = nn.Sequential(
                nn.Conv2d(channels + 2, channels // 2, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(channels // 2, 1, 1),
            )
            self.merge = nn.Conv2d(2 * channels, channels, 1)

    def forward(self, features, density, available, ages):
        """Fuse ``features`` [B, N, C, h, w] into [B, C, h, w].

        ``available`` [B, N, h, w] marks the cells the receiver holds for each
        node; node 0 is the receiver and is always available. ``density`` is
        each node's log point density [B, N, 1, h, w] and ``ages`` [B, N] the
        age of each node's message in seconds.
        """
        batch, nodes, _, height, width = features.shape
        mask = available[:, :, None]
        # The receiver never holds unsent cells; zero them so no convolution
        # below can read across the mask.
        features = features.masked_fill(~mask, 0)
        lowest = torch.finfo(features.dtype).min
        maximum = features.masked_fill(~mask, lowest).amax(dim=1)
        if self.mode == "max":
            return maximum
        if self.mode == "mean":
            count = mask.sum(dim=1).clamp(min=1).to(features.dtype)
            return (features * mask).sum(dim=1) / count
        age = ages.to(features.dtype)[:, :, None, None, None]
        cues = torch.cat(
            (
                features,
                density.to(features.dtype).masked_fill(~mask, 0),
                age.expand(batch, nodes, 1, height, width),
            ),
            dim=2,
        )
        logits = self.score(cues.flatten(0, 1)).reshape(batch, nodes, 1, height, width)
        logits = logits.float().masked_fill(~mask, float("-inf"))
        weights = torch.softmax(logits, dim=1).to(features.dtype)
        average = (weights * features).sum(dim=1)
        return self.merge(torch.cat((average, maximum), dim=1))


class Backbone(nn.Module):
    """Three-scale backbone over the fused map with top-down (FPN) merging."""

    def __init__(self, in_channels, channels, blocks=(2, 2, 2)):
        super().__init__()
        widths = (channels, 2 * channels, 2 * channels)
        self.entry = nn.Conv2d(in_channels, widths[0], 1)
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for index, (width, count) in enumerate(zip(widths, blocks)):
            if index:
                self.downsamples.append(downsample(widths[index - 1], width))
            self.stages.append(
                nn.Sequential(*(ConvNeXtBlock(width) for _ in range(count)))
            )
        self.laterals = nn.ModuleList(nn.Conv2d(width, channels, 1) for width in widths)
        self.output = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            LayerNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, fused):
        scales = []
        value = self.entry(fused)
        for index, stage in enumerate(self.stages):
            if index:
                value = self.downsamples[index - 1](value)
            value = stage(value)
            scales.append(value)
        merged = self.laterals[-1](scales[-1])
        for scale, lateral in zip(scales[-2::-1], list(self.laterals)[-2::-1]):
            merged = lateral(scale) + F.interpolate(
                merged, size=scale.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.output(merged)


class DRIVEDetector(nn.Module):
    def __init__(
        self,
        classes,
        in_channels,
        objects_per_cell=1,
        node_channels=64,
        backbone_channels=128,
        fusion="attention",
        node_blocks=1,
        backbone_blocks=(2, 2, 2),
        output_stride=2,
    ):
        super().__init__()
        if output_stride != 2:
            raise ValueError("DRIVEDetector currently supports output_stride=2")
        self.classes = classes
        self.objects_per_cell = objects_per_cell
        self.output_stride = output_stride
        slots = len(classes) * objects_per_cell
        self.encoder = NodeEncoder(in_channels, node_channels, node_blocks)
        # Learned, content-dependent value of each collaborator cell for
        # budget-limited sparse transmission.
        self.utility = nn.Sequential(
            nn.Conv2d(node_channels, node_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(node_channels, 1, 1),
        )
        self.fusion = NodeFusion(node_channels, fusion)
        self.backbone = Backbone(node_channels, backbone_channels, backbone_blocks)
        self.heatmap = detection_head(backbone_channels, slots, bias=-2.19)
        # The last channel of each slot predicts the 3D IoU of its own box and
        # rescales the detection score, so well-localized boxes rank first.
        self.regression = detection_head(backbone_channels, REGRESSION_CHANNELS * slots)

    def encode(self, bev):
        """Node features [B, N, C, h, w] and log point density [B, N, 1, h, w]
        for a [B, N, C_in, H, W] stack of per-node BEVs."""
        batch, nodes = bev.shape[:2]
        flat = bev.flatten(0, 1)
        features = self.encoder(flat)
        # BEV channel 0 is log(1 + point count); pool it onto the feature grid.
        density = F.avg_pool2d(flat[:, :1], 2, ceil_mode=True)
        return (
            features.reshape(batch, nodes, *features.shape[1:]),
            density.reshape(batch, nodes, *density.shape[1:]),
        )

    def fuse(self, features, density, available=None, ages=None):
        batch, nodes, _, height, width = features.shape
        if available is None:
            available = torch.ones(
                batch, nodes, height, width, dtype=torch.bool, device=features.device
            )
        if ages is None:
            ages = torch.zeros(batch, nodes, device=features.device)
        return self.fusion(features, density, available, ages)

    def detect(self, fused):
        value = self.backbone(fused)
        return self.heatmap(value), self.regression(value)

    def select(self, features, density, budget, selector="learned", generators=None):
        """Cells each collaborator transmits under ``budget``.

        Node 0 is the receiver and keeps its dense map; every other node sends
        exactly ``ceil(budget * cells)`` cells. Returns the availability mask
        [B, N, h, w] and, for the learned selector, a straight-through gate
        [B, N-1, h, w] whose forward value is 1 on sent cells while every
        utility score still receives a gradient.
        """
        if not 0 < budget <= 1:
            raise ValueError("adaptive budget must be in (0, 1]")
        if selector not in SELECTORS:
            raise ValueError(f"unknown adaptive selector: {selector}")
        batch, nodes, channels, height, width = features.shape
        receiver = torch.ones(
            batch, 1, height, width, dtype=torch.bool, device=features.device
        )
        if nodes == 1:
            return receiver, None
        collaborators = features[:, 1:]
        if selector == "learned":
            flat = collaborators.flatten(0, 1)
            scores = self.utility(flat).reshape(batch, nodes - 1, height * width)
        elif selector == "energy":
            scores = (
                collaborators.float()
                .abs()
                .mean(dim=2)
                .reshape(batch, nodes - 1, height * width)
            )
        elif selector == "confidence":
            # Where2comm-style: each collaborator ranks its cells by the
            # confidence of its own single-node detections.
            own = self.detect(
                self.fuse(
                    collaborators.flatten(0, 1)[:, None],
                    density[:, 1:].flatten(0, 1)[:, None],
                )
            )[0]
            scores = own.float().amax(dim=1).reshape(batch, nodes - 1, height * width)
        elif generators is None:
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
        count = max(1, min(height * width, math.ceil(budget * height * width)))
        selected = scores.topk(count, dim=-1).indices
        hard = torch.zeros_like(scores)
        hard.scatter_(-1, selected, 1.0)
        mask = hard.bool().reshape(batch, nodes - 1, height, width)
        gate = None
        if selector == "learned":
            soft = scores.sigmoid()
            gate = (hard + soft - soft.detach()).reshape(
                batch, nodes - 1, height, width
            )
        return torch.cat((receiver, mask), dim=1), gate

    def forward(self, bev, mode="early", ages=None):
        """Detect from a [B, N, C_in, H, W] (or one-node [B, C_in, H, W]) BEV.

        ``mode`` is ``early``, ``intermediate``, or ``adaptive@budget``.
        Early and single-node inputs are one node, so all modes share every
        weight; only what reaches the fusion differs.
        """
        if bev.ndim == 4:
            bev = bev[:, None]
        base, _, budget_text = mode.partition("@")
        features, density = self.encode(bev)
        available = None
        if base == "adaptive" and features.shape[1] > 1:
            available, gate = self.select(
                features, density, float(budget_text) if budget_text else 0.1
            )
            features = torch.cat(
                (features[:, :1], features[:, 1:] * gate[:, :, None]), dim=1
            )
        return self.detect(self.fuse(features, density, available, ages))


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
    """Focal heatmap loss, L1 box regression, and IoU quality.

    The focal loss is normalized by the number of positives in the batch, as
    in CenterNet and CenterPoint; class imbalance is handled by repeat-factor
    sampling of training frames instead. ``cell_size`` is the prediction-grid
    cell in metres along x and y; it converts offset errors to metres for the
    IoU-quality target.
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
    # A per-class normalization divides a class absent from the batch by one,
    # so its negatives over the whole grid would dominate the loss.
    positive_loss = ((1 - p).pow(2) * F.logsigmoid(x) * positive).sum()
    negative_loss = (p.pow(2) * F.logsigmoid(-x) * negative_weight * negative).sum()
    focal = -(positive_loss + negative_loss) / positive.sum().clamp(min=1)
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
