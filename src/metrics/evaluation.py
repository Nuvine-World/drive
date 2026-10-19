"""Rotated 3D/BEV IoU, class-wise NMS and dataset-level all-points AP."""

import dataclasses
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


def _pair_iou(a, b, footprints_a, footprints_b, rows, columns, bev=False):
    """Rotated 3D IoU (or BEV IoU of the footprints, ignoring height) for the
    index pairs ``(rows[k], columns[k])``."""
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
    touching = (distance <= radius_a + radius_b) & ((z > 0) | bev)
    if not touching.any():
        return result
    rows, columns = rows[touching], columns[touching]
    area = shapely.area(shapely.intersection(footprints_a[rows], footprints_b[columns]))
    if bev:
        intersection = area
        union = a[rows, 3] * a[rows, 4] + b[columns, 3] * b[columns, 4] - area
    else:
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


def iou3d_matrix(boxes_a, boxes_b, bev=False):
    """Pairwise rotated 3D (or BEV) IoU; only pairs whose footprints can touch
    use shapely."""
    a, b = _box_array(boxes_a), _box_array(boxes_b)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    rows, columns = np.indices((len(a), len(b))).reshape(2, -1)
    values = _pair_iou(a, b, _footprints(a), _footprints(b), rows, columns, bev)
    return values.reshape(len(a), len(b))


def iou3d(a, b):
    return float(iou3d_matrix([a], [b])[0, 0])


def paired_iou3d(a, b):
    """Rotated 3D IoU of row ``k`` of ``a`` with row ``k`` of ``b``.

    Both are ``[N, 7]`` arrays of ``[x, y, z, l, w, h, yaw]``.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if not len(a):
        return np.zeros(0)
    index = np.arange(len(a))
    return _pair_iou(a, b, _footprints(a), _footprints(b), index, index)


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


def _operating_threshold(operating_score_threshold, label):
    if isinstance(operating_score_threshold, dict):
        return operating_score_threshold.get(label)
    return operating_score_threshold


def _class_candidates(frames, label, bev=False):
    """Ranked predictions of one class and their overlaps with its truths.

    A frame is ``(predictions, truths)`` or ``(predictions, truths, ignored)``;
    ignored truths are real objects that are not scored.
    """
    truths = [[b for b in frame[1] if b.label == label] for frame in frames]
    ignored = [
        [b for b in (frame[2] if len(frame) > 2 else ()) if b.label == label]
        for frame in frames
    ]
    predictions = sorted(
        [
            (p.score, i, p)
            for i, frame in enumerate(frames)
            for p in frame[0]
            if p.label == label
        ],
        key=lambda v: v[0],
        reverse=True,
    )
    ranks_by_frame = defaultdict(list)
    for rank, (_, frame_index, _) in enumerate(predictions):
        ranks_by_frame[frame_index].append(rank)
    overlaps, ignored_overlaps = {}, {}
    for frame_index, ranks in ranks_by_frame.items():
        boxes = [predictions[rank][2] for rank in ranks]
        overlaps.update(zip(ranks, iou3d_matrix(boxes, truths[frame_index], bev)))
        if ignored[frame_index]:
            best = iou3d_matrix(boxes, ignored[frame_index], bev).max(axis=1)
            ignored_overlaps.update(zip(ranks, best.tolist()))
    return truths, ignored, predictions, overlaps, ignored_overlaps


def _match(frame_count, predictions, overlaps, ignored_overlaps, threshold):
    """Standard ranked matching: each prediction tries its best truth first.

    Returns one ``(status, overlap, truth_index)`` per prediction, where
    status is ``tp``, ``fp``, or ``ignored``.
    """
    matched = [set() for _ in range(frame_count)]
    outcomes = []
    for rank, (_, i, _) in enumerate(predictions):
        row = overlaps[rank]
        if len(row):
            # On exact ties prefer the last GT, as max() over (iou, j) did.
            j = len(row) - 1 - int(np.argmax(row[::-1]))
            overlap = float(row[j])
        else:
            overlap, j = 0.0, -1
        if j >= 0 and j not in matched[i] and overlap >= threshold:
            matched[i].add(j)
            outcomes.append(("tp", overlap, j))
        elif ignored_overlaps.get(rank, 0.0) >= threshold:
            # Detecting an unscored object is neither a true nor a false
            # positive, as with KITTI DontCare regions and COCO crowd boxes.
            outcomes.append(("ignored", float(ignored_overlaps[rank]), -1))
        else:
            # A duplicate whose best GT is already claimed is a false positive.
            outcomes.append(("fp", overlap, j))
    return outcomes


def evaluate(
    frames,
    classes,
    thresholds,
    operating_score_threshold=None,
    class_thresholds=None,
    records=None,
    iou_type="3d",
):
    """Dataset-level detection metrics; never average per-frame AP.

    AP ranks every retained low-threshold prediction. Precision, recall, F1, and
    localization diagnostics use ``operating_score_threshold`` when provided,
    so threshold calibration cannot truncate the AP curve. The operating
    threshold may be one number or a ``{class: threshold}`` mapping.

    ``thresholds`` are the nominal IoU levels. ``class_thresholds`` may replace
    them for a class with a list of the same length (for example, looser
    levels for pedestrians). Each row records the applied ``iou`` and its
    ``iou_level`` index.

    Frames may carry a third element of ignored truths. Predictions matching
    only an ignored truth are dropped from AP and from the operating counts.
    When ``records`` is a list, it receives the outcome of every
    operating-point prediction and every truth at the first IoU level.
    ``iou_type`` is ``3d`` (rotated boxes) or ``bev`` (rotated footprints).
    """
    if iou_type not in ("3d", "bev"):
        raise ValueError("iou_type must be 3d or bev")
    class_thresholds = class_thresholds or {}
    results = []
    for label in classes:
        levels = class_thresholds.get(label, thresholds)
        operating_threshold = _operating_threshold(operating_score_threshold, label)
        truths, ignored, predictions, overlaps, ignored_overlaps = _class_candidates(
            frames, label, iou_type == "bev"
        )
        count = sum(map(len, truths))
        for level, threshold in enumerate(levels):
            outcomes = _match(
                len(frames), predictions, overlaps, ignored_overlaps, threshold
            )
            tp = []
            true_positives = prediction_count = ignored_predictions = 0
            found = set()
            matched_ious, center_errors, yaw_errors, size_ious = [], [], [], []
            for (score, i, pred), (status, overlap, j) in zip(predictions, outcomes):
                at_operating_point = (
                    operating_threshold is None or score >= operating_threshold
                )
                if records is not None and level == 0 and at_operating_point:
                    records.append(
                        {
                            "frame": i,
                            "record": "prediction",
                            "class": label,
                            "status": status,
                            "score": score,
                            "iou": overlap,
                            "box": pred,
                        }
                    )
                if status == "ignored":
                    ignored_predictions += int(at_operating_point)
                    continue
                tp.append(int(status == "tp"))
                if not at_operating_point:
                    continue
                prediction_count += 1
                if status != "tp":
                    continue
                true_positives += 1
                found.add((i, j))
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
            if records is not None and level == 0:
                for i, frame_truths in enumerate(truths):
                    for j, gt in enumerate(frame_truths):
                        records.append(
                            {
                                "frame": i,
                                "record": "ground_truth",
                                "class": label,
                                "status": "matched" if (i, j) in found else "missed",
                                "score": None,
                                "iou": None,
                                "box": gt,
                            }
                        )
                    for gt in ignored[i]:
                        records.append(
                            {
                                "frame": i,
                                "record": "ground_truth",
                                "class": label,
                                "status": "ignored",
                                "score": None,
                                "iou": None,
                                "box": gt,
                            }
                        )
            ap, recall = None, None
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
                    "operating_score_threshold": operating_threshold,
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
                    "ranked_predictions": len(tp),
                    "ignored_ground_truth": sum(map(len, ignored)),
                    "ignored_predictions": ignored_predictions,
                    "frames": len(frames),
                }
            )
    return results


def group_frames(frames, groups, nms_iou, limit):
    """Relabel frames to merged categories, such as LV = car + truck.

    A detector that separates car and truck can report one vehicle under
    both, so each group's predictions pass a class-agnostic NMS first.
    Boxes of classes outside every group are dropped.
    """
    mapping = {label: name for name, labels in groups.items() for label in labels}

    def relabel(boxes):
        return [
            dataclasses.replace(box, label=mapping[box.label])
            for box in boxes
            if box.label in mapping
        ]

    return [
        (nms(relabel(frame[0]), nms_iou, limit), *(relabel(part) for part in frame[1:]))
        for frame in frames
    ]


def f1_optimal_thresholds(
    frames, classes, thresholds, class_thresholds=None, fallback=None, minimum=0.0
):
    """Per-class score threshold that maximizes F1 at the first IoU level.

    The matching of each ranked prediction does not depend on the cut-off, so
    F1 at every distinct score follows from one ranked pass. Ties prefer the
    higher threshold. Classes without ground truth or true positives keep
    ``fallback``.
    """
    class_thresholds = class_thresholds or {}
    chosen = {}
    for label in classes:
        threshold = class_thresholds.get(label, thresholds)[0]
        truths, _, predictions, overlaps, ignored_overlaps = _class_candidates(
            frames, label
        )
        count = sum(map(len, truths))
        outcomes = _match(
            len(frames), predictions, overlaps, ignored_overlaps, threshold
        )
        kept = [
            (score, status == "tp")
            for (score, _, _), (status, _, _) in zip(predictions, outcomes)
            if status != "ignored"
        ]
        if not count or not kept:
            chosen[label] = fallback
            continue
        scores = np.array([score for score, _ in kept])
        hits = np.cumsum([hit for _, hit in kept])
        # A cut-off falls only after the last of equal scores.
        last = np.r_[scores[1:] != scores[:-1], True]
        f1 = 2 * hits / (np.arange(1, len(kept) + 1) + count)
        f1 = np.where(last & (scores >= minimum), f1, -1.0)
        if f1.max() <= 0:
            chosen[label] = fallback
            continue
        chosen[label] = float(scores[int(np.argmax(f1))])
    return chosen
