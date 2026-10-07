"""Rotated 3D IoU, class-wise NMS and dataset-level all-points AP."""
import math
from collections import defaultdict
import numpy as np
import shapely

CORNERS = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float) / 2


def _box_array(boxes):
    return np.array(
        [[b.x, b.y, b.z, b.length, b.width, b.height, b.yaw] for b in boxes],
        dtype=float,
    ).reshape(-1, 7)


def _footprints(values):
    """Vectorized rotated BEV rectangles for ``[x, y, z, l, w, h, yaw]`` rows."""
    local = CORNERS[None] * values[:, None, 3:5]
    c, s = np.cos(values[:, 6])[:, None], np.sin(values[:, 6])[:, None]
    x = local[..., 0] * c - local[..., 1] * s + values[:, 0, None]
    y = local[..., 0] * s + local[..., 1] * c + values[:, 1, None]
    return shapely.polygons(np.stack((x, y), axis=-1))


def _pair_iou(a, b, footprints_a, footprints_b, rows, columns):
    """Rotated 3D IoU for the index pairs ``(rows[k], columns[k])``."""
    result = np.zeros(len(rows))
    if not len(rows):
        return result
    # If the horizontal bounding circles do not intersect, neither can the boxes.
    radius_a = np.hypot(a[rows, 3], a[rows, 4]) / 2
    radius_b = np.hypot(b[columns, 3], b[columns, 4]) / 2
    distance = np.hypot(a[rows, 0] - b[columns, 0], a[rows, 1] - b[columns, 1])
    z = np.maximum(
        0,
        np.minimum(a[rows, 2] + a[rows, 5] / 2, b[columns, 2] + b[columns, 5] / 2)
        - np.maximum(a[rows, 2] - a[rows, 5] / 2, b[columns, 2] - b[columns, 5] / 2),
    )
    touching = (distance <= radius_a + radius_b) & (z > 0)
    if not touching.any():
        return result
    rows, columns = rows[touching], columns[touching]
    area = shapely.area(shapely.intersection(footprints_a[rows], footprints_b[columns]))
    intersection = area * z[touching]
    union = (
        a[rows, 3] * a[rows, 4] * a[rows, 5]
        + b[columns, 3] * b[columns, 4] * b[columns, 5]
        - intersection
    )
    result[touching] = np.divide(
        intersection, union, out=np.zeros_like(union), where=union > 0
    )
    return result


def iou3d_matrix(boxes_a, boxes_b):
    """Pairwise rotated 3D IoU; only pairs whose footprints can touch use shapely."""
    a, b = _box_array(boxes_a), _box_array(boxes_b)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    rows, columns = np.indices((len(a), len(b))).reshape(2, -1)
    values = _pair_iou(a, b, _footprints(a), _footprints(b), rows, columns)
    return values.reshape(len(a), len(b))


def iou3d(a, b):
    return float(iou3d_matrix([a], [b])[0, 0])


def nms(boxes, threshold, limit):
    """Greedy class-wise NMS: a box survives unless a kept, higher-scoring box
    of the same class overlaps it by more than ``threshold``."""
    ordered = sorted(boxes, key=lambda box: box.score, reverse=True)
    if not ordered:
        return []
    values = _box_array(ordered)
    footprints = _footprints(values)
    labels = np.array([box.label for box in ordered])
    suppressed = np.zeros(len(ordered), dtype=bool)
    kept = []
    for index, box in enumerate(ordered):
        if suppressed[index]:
            continue
        kept.append(box)
        if len(kept) >= limit:
            break
        later = np.nonzero(~suppressed & (labels == box.label))[0]
        later = later[later > index]
        overlaps = _pair_iou(
            values,
            values,
            footprints,
            footprints,
            np.full(len(later), index),
            later,
        )
        suppressed[later[overlaps > threshold]] = True
    return kept


def angle_error(a, b):
    """Smallest yaw error for a rectangular box (180-degree symmetry), in degrees."""
    return math.degrees(abs((a - b + math.pi / 2) % math.pi - math.pi / 2))


def evaluate(
    frames,
    classes,
    thresholds,
    operating_score_threshold=None,
    class_thresholds=None,
):
    """Dataset-level detection metrics; never average per-frame AP.

    AP ranks every retained low-threshold prediction. Precision, recall, F1, and
    localization diagnostics use ``operating_score_threshold`` when provided,
    so threshold calibration cannot truncate the AP curve.

    ``thresholds`` are the nominal IoU levels. ``class_thresholds`` may replace
    them for a class with a list of the same length (for example, looser
    levels for pedestrians). Each row records the applied ``iou`` and its
    ``iou_level`` index.
    """
    class_thresholds = class_thresholds or {}
    results = []
    for label in classes:
        levels = class_thresholds.get(label, thresholds)
        truths = [[b for b in gt if b.label == label] for _, gt in frames]
        count = sum(map(len, truths))
        predictions = sorted([(p.score, i, p) for i, (pred, _) in enumerate(frames)
                              for p in pred if p.label == label], key=lambda v: v[0], reverse=True)
        ranks_by_frame = defaultdict(list)
        for rank, (_, frame_index, _) in enumerate(predictions):
            ranks_by_frame[frame_index].append(rank)
        overlaps = {}
        for frame_index, ranks in ranks_by_frame.items():
            matrix = iou3d_matrix(
                [predictions[rank][2] for rank in ranks], truths[frame_index]
            )
            overlaps.update(zip(ranks, matrix))
        for level, threshold in enumerate(levels):
            matched = [set() for _ in frames]
            tp = []
            matched_ious, center_errors, yaw_errors, size_ious = [], [], [], []
            for rank, (score, i, pred) in enumerate(predictions):
                # Standard ranked detection matching: select the best GT first;
                # a duplicate whose best GT is already claimed is a false positive.
                row = overlaps[rank]
                if len(row):
                    # On exact ties prefer the last GT, as max() over (iou, j) did.
                    j = len(row) - 1 - int(np.argmax(row[::-1]))
                    overlap = float(row[j])
                else:
                    overlap, j = 0.0, -1
                success = j >= 0 and j not in matched[i] and overlap >= threshold
                tp.append(int(success))
                at_operating_point = (
                    operating_score_threshold is None
                    or score >= operating_score_threshold
                )
                if success:
                    matched[i].add(j)
                    if at_operating_point:
                        gt = truths[i][j]
                        matched_ious.append(overlap)
                        center_errors.append(
                            math.dist((pred.x, pred.y, pred.z), (gt.x, gt.y, gt.z))
                        )
                        yaw_errors.append(angle_error(pred.yaw, gt.yaw))
                        intersection = (
                            min(pred.length, gt.length)
                            * min(pred.width, gt.width)
                            * min(pred.height, gt.height)
                        )
                        union = (
                            pred.length * pred.width * pred.height
                            + gt.length * gt.width * gt.height
                            - intersection
                        )
                        size_ious.append(intersection / union if union > 0 else 0.0)
            ap, recall = None, None
            operating = [
                value
                for value, (score, _, _) in zip(tp, predictions)
                if operating_score_threshold is None
                or score >= operating_score_threshold
            ]
            true_positives = sum(operating)
            prediction_count = len(operating)
            false_positives = prediction_count - true_positives
            false_negatives = count - true_positives
            precision = (
                true_positives / prediction_count
                if prediction_count
                else (0.0 if count else None)
            )
            f1 = (
                2
                * precision
                * true_positives
                / count
                / (precision + true_positives / count)
                if count
                and precision is not None
                and precision + true_positives / count > 0
                else (0.0 if count else None)
            )
            if count:
                cumulative = np.cumsum(tp)
                recall_curve = cumulative/count
                precision_curve = cumulative/np.arange(1, len(tp)+1)
                r = np.r_[0, recall_curve, 1]
                p = np.r_[0, precision_curve, 0]
                p = np.maximum.accumulate(p[::-1])[::-1]
                changes = np.where(r[1:] != r[:-1])[0]
                ap = float(np.sum((r[changes + 1] - r[changes]) * p[changes + 1]))
                recall = float(true_positives / count)
            results.append(
                {
                    "class": label,
                    "iou_level": level,
                    "iou": threshold,
                    "ap_3d": ap,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "true_positives": true_positives,
                    "false_positives": false_positives,
                    "false_negatives": false_negatives,
                    "mean_matched_iou_3d": (
                        float(np.mean(matched_ious)) if matched_ious else None
                    ),
                    "mean_center_error_m": (
                        float(np.mean(center_errors)) if center_errors else None
                    ),
                    "mean_yaw_error_deg": (
                        float(np.mean(yaw_errors)) if yaw_errors else None
                    ),
                    "mean_size_iou": float(np.mean(size_ious)) if size_ious else None,
                    "ground_truth": count,
                    "predictions": prediction_count,
                    "ranked_predictions": len(predictions),
                    "frames": len(frames),
                }
            )
    return results
