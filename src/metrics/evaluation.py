"""Rotated 3D IoU, class-wise NMS and dataset-level all-points AP."""
import math
import numpy as np
from shapely.geometry import Polygon


def polygon(box):
    corners = np.array([[-1,-1], [1,-1], [1,1], [-1,1]]) * [box.length/2, box.width/2]
    c, s = math.cos(box.yaw), math.sin(box.yaw)
    return Polygon(corners @ np.array([[c,s],[-s,c]]) + [box.x,box.y])


def iou3d(a, b):
    intersection = polygon(a).intersection(polygon(b)).area
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


def evaluate(frames, classes, thresholds):
    """Dataset-level detection metrics; never average per-frame AP.

    Predictions have already passed the configured score threshold and NMS, so
    precision/F1 describe that deployed operating point. AP remains ranked over
    every retained prediction. Localization errors are calculated on true-positive
    matches only and are left undefined when there are no matches.
    """
    results = []
    for label in classes:
        truths = [[b for b in gt if b.label == label] for _, gt in frames]
        count = sum(map(len, truths))
        predictions = sorted([(p.score, i, p) for i, (pred, _) in enumerate(frames)
                              for p in pred if p.label == label], key=lambda v: v[0], reverse=True)
        for threshold in thresholds:
            matched = [set() for _ in frames]
            tp = []
            matched_ious, center_errors, yaw_errors, size_ious = [], [], [], []
            for _, i, pred in predictions:
                candidates = [(iou3d(pred, gt), j) for j, gt in enumerate(truths[i]) if j not in matched[i]]
                overlap, j = max(candidates, default=(0.0, -1))
                success = j >= 0 and overlap >= threshold
                tp.append(int(success))
                if success:
                    matched[i].add(j)
                    gt = truths[i][j]
                    matched_ious.append(overlap)
                    center_errors.append(math.dist((pred.x, pred.y, pred.z), (gt.x, gt.y, gt.z)))
                    yaw_errors.append(angle_error(pred.yaw, gt.yaw))
                    intersection = min(pred.length, gt.length) * min(pred.width, gt.width) * min(pred.height, gt.height)
                    union = pred.length * pred.width * pred.height + gt.length * gt.width * gt.height - intersection
                    size_ious.append(intersection / union if union > 0 else 0.0)
            ap, recall = None, None
            true_positives = sum(tp)
            false_positives = len(tp) - true_positives
            false_negatives = count - true_positives
            precision = true_positives / len(tp) if tp else (0.0 if count else None)
            f1 = (2 * precision * true_positives / count / (precision + true_positives / count)
                  if count and precision is not None and precision + true_positives / count > 0 else
                  (0.0 if count else None))
            if count:
                cumulative = np.cumsum(tp)
                recall_curve = cumulative/count
                precision_curve = cumulative/np.arange(1, len(tp)+1)
                r = np.r_[0, recall_curve, 1]
                p = np.r_[0, precision_curve, 0]
                p = np.maximum.accumulate(p[::-1])[::-1]
                changes = np.where(r[1:] != r[:-1])[0]
                ap = float(np.sum((r[changes+1]-r[changes])*p[changes+1]))
                recall = float(true_positives/count)
            results.append({'class':label, 'iou':threshold, 'ap_3d':ap,
                            'precision':precision, 'recall':recall, 'f1':f1,
                            'true_positives':true_positives, 'false_positives':false_positives,
                            'false_negatives':false_negatives,
                            'mean_matched_iou_3d':float(np.mean(matched_ious)) if matched_ious else None,
                            'mean_center_error_m':float(np.mean(center_errors)) if center_errors else None,
                            'mean_yaw_error_deg':float(np.mean(yaw_errors)) if yaw_errors else None,
                            'mean_size_iou':float(np.mean(size_ious)) if size_ious else None,
                            'ground_truth':count, 'predictions':len(predictions), 'frames':len(frames)})
    return results
