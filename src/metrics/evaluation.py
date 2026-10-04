"""Rotated 3D IoU, class-wise NMS and dataset-level all-points AP."""
import math
import numpy as np
from shapely.geometry import Polygon


def polygon(box):
    corners = np.array([[-1,-1], [1,-1], [1,1], [-1,1]]) * [box.length/2, box.width/2]
    c, s = math.cos(box.yaw), math.sin(box.yaw)
    return Polygon(corners @ np.array([[c,s],[-s,c]]) + [box.x,box.y])


def iou3d(a, b, polygons=None):
    # If the horizontal bounding circles do not intersect, neither can the boxes.
    radius_a = math.hypot(a.length, a.width) / 2
    radius_b = math.hypot(b.length, b.width) / 2
    if math.hypot(a.x - b.x, a.y - b.y) > radius_a + radius_b:
        return 0.0
    if polygons is None:
        polygon_a, polygon_b = polygon(a), polygon(b)
    else:
        if id(a) not in polygons:
            polygons[id(a)] = polygon(a)
        if id(b) not in polygons:
            polygons[id(b)] = polygon(b)
        polygon_a, polygon_b = polygons[id(a)], polygons[id(b)]
    intersection = polygon_a.intersection(polygon_b).area
    z = max(0, min(a.z+a.height/2, b.z+b.height/2)-max(a.z-a.height/2, b.z-b.height/2))
    intersection *= z
    union = a.length*a.width*a.height + b.length*b.width*b.height - intersection
    return intersection/union if union > 0 else 0.0


def nms(boxes, threshold, limit):
    kept = []
    for box in sorted(boxes, key=lambda b: b.score, reverse=True):
        if all(box.label != b.label or iou3d(box, b) <= threshold for b in kept):
            kept.append(box)
            if len(kept) >= limit:
                break
    return kept


def angle_error(a, b):
    """Smallest yaw error for a rectangular box (180-degree symmetry), in degrees."""
    return math.degrees(abs((a - b + math.pi / 2) % math.pi - math.pi / 2))


def evaluate(frames, classes, thresholds, operating_score_threshold=None):
    """Dataset-level detection metrics; never average per-frame AP.

    AP ranks every retained low-threshold prediction. Precision, recall, F1, and
    localization diagnostics use ``operating_score_threshold`` when provided,
    so threshold calibration cannot truncate the AP curve.
    """
    results = []
    for label in classes:
        truths = [[b for b in gt if b.label == label] for _, gt in frames]
        count = sum(map(len, truths))
        predictions = sorted([(p.score, i, p) for i, (pred, _) in enumerate(frames)
                              for p in pred if p.label == label], key=lambda v: v[0], reverse=True)
        polygons = {}
        overlaps = {
            (id(pred), j): iou3d(pred, gt, polygons)
            for _, frame_index, pred in predictions
            for j, gt in enumerate(truths[frame_index])
        }
        for threshold in thresholds:
            matched = [set() for _ in frames]
            tp = []
            matched_ious, center_errors, yaw_errors, size_ious = [], [], [], []
            for score, i, pred in predictions:
                # Standard ranked detection matching: select the best GT first;
                # a duplicate whose best GT is already claimed is a false positive.
                candidates = [
                    (overlaps[(id(pred), j)], j) for j in range(len(truths[i]))
                ]
                overlap, j = max(candidates, default=(0.0, -1))
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
