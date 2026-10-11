"""ECCO preprocessing, training, inference, evaluation, and reporting pipeline."""

import copy
import csv
import gc
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import time
import zlib
import gzip
import platform
import warnings
import shutil
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np
import torch
from src.data.api import annotations, bev_channels, discover, load_cloud, rasterize
from src.data.augmentation import augment_scene, sample_parameters
from src.data.visibility import box_point_counts, node_origin
from src.detection.bev import (
    DRIVEDetector,
    TargetCollisionError,
    decode,
    grid_cell_size,
    loss,
    output_shape,
    target_cell,
    targets,
)
from src.metrics.evaluation import evaluate, f1_optimal_thresholds, group_frames, nms
from src.network.control import PastOnlyLinkEstimator, past_frame
from src.network.deadline import (
    Link,
    Option,
    OutcomeLog,
    dense_bytes,
    feature_controllers,
    make_trace,
    sparse_bytes,
)

EVALUATION_VERSION = 6
MODEL_VERSION = 7
GT_SCOPES = ("full", "visible")
# CoInfra benchmark, LiDAR only, all weather, AP@0.5 (arXiv 2507.02245v3,
# Table 5). Frames there are split 80/10/10 at random, not by session.
COINFRA_PUBLISHED_AP = {
    "GlobalLidar": {"Bus": 0.997, "LV": 0.977, "VRU": 0.615},
    "LocalLidar": {"Bus": 0.803, "LV": 0.807, "VRU": 0.128},
}
RANGE_BINS_M = (30.0, 60.0)
# Throughput prior before the first link-capacity report.
DEADLINE_INITIAL_MBPS = 10.0


def write_json(path, value):
    atomic_write_text(path, json.dumps(value, indent=2, allow_nan=False))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as stream:
        if rows:
            # Resumed runs can append metrics introduced by a newer version of
            # the pipeline to history loaded from an older checkpoint.  Keep
            # first-seen column order while including keys from every row.
            fieldnames = list(dict.fromkeys(key for row in rows for key in row))
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_write_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def plot_training(path, rows):
    if not rows:
        return
    try:
        import matplotlib
    except ImportError:
        print('Plot skipped: install matplotlib from requirements.txt', flush=True)
        return
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    epochs = [r['epoch'] for r in rows]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(epochs, [r['train_loss'] for r in rows], label='train loss')
    axes[0].plot(epochs, [r['val_loss'] for r in rows], label='validation loss')
    axes[0].set(xlabel='epoch', ylabel='loss', title='Loss')
    axes[0].legend()
    axes[1].plot(epochs, [r['val_map'] for r in rows], label='validation mAP')
    axes[1].plot(epochs, [r['val_precision'] for r in rows], label='validation precision')
    axes[1].plot(epochs, [r['val_recall'] for r in rows], label='validation recall')
    axes[1].plot(epochs, [r['val_f1'] for r in rows], label='validation F1')
    axes[1].plot(epochs, [r['selection_score'] for r in rows], label='selection score')
    axes[1].set(xlabel='epoch', ylabel='score', ylim=(0, 1), title='Validation metrics')
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def validate_batch(model, examples, cfg, device):
    """Score one prepared batch under every validation view.

    Every frame is scored twice: with all of its nodes (each configured fusion
    strategy) and with one deterministic single node, so checkpoint selection
    sees both sides of the cooperation comparison.
    """
    early = device_tensor(np.stack([item[0] for item in examples]), device)
    per_node = device_tensor(np.stack([item[1] for item in examples]), device)
    early_target, single_target = (
        tuple(
            device_tensor(np.stack([item[position][i] for item in examples]), device)
            for i in range(3)
        )
        for position in (3, 6)
    )
    strategies = cfg["evaluation"]["strategies"]
    with torch.no_grad(), autocast(cfg, device):
        early_outputs = model(early, "early")
        batch_size, nodes = per_node.shape[:2]
        features, density = model.encode(per_node)
        rows = torch.arange(batch_size, device=features.device)
        single_nodes = torch.as_tensor(
            [item[4] for item in examples], device=features.device
        )
        single = model.detect(
            model.fuse(
                features[rows, single_nodes][:, None],
                density[rows, single_nodes][:, None],
            )
        )
        intermediate = (
            model.detect(model.fuse(features, density))
            if "intermediate" in strategies
            else None
        )
        adaptive = None
        if "adaptive" in strategies:
            validation_budget = cfg["adaptive_fusion"]["evaluation_budgets"][0]
            available, _ = model.select(features, density, validation_budget)
            adaptive = model.detect(model.fuse(features, density, available))
        late = (
            model.detect(
                model.fuse(
                    features.flatten(0, 1)[:, None], density.flatten(0, 1)[:, None]
                )
            )
            if "late" in strategies
            else None
        )
    # Per-sample losses keep validation loss independent of batch size,
    # batch composition, and OOM splitting.
    # Each view is scored against the boxes its own nodes can support.
    losses = []
    cells = cell_size(cfg)
    for index in range(batch_size):

        def sample_target(target):
            return tuple(value[index : index + 1] for value in target)

        losses.append(
            0.5
            * (
                float(
                    loss(
                        batch_outputs(early_outputs, index),
                        sample_target(early_target),
                        cells,
                    )
                )
                + float(
                    loss(
                        batch_outputs(single, index),
                        sample_target(single_target),
                        cells,
                    )
                )
            )
        )
    frames = {strategy: [] for strategy in validation_views(cfg)}
    for index, item in enumerate(examples):
        frames["single"].append((decode(batch_outputs(single, index), cfg), *item[5]))
        if "early" in frames:
            frames["early"].append(
                (decode(batch_outputs(early_outputs, index), cfg), *item[2])
            )
        if intermediate is not None:
            frames["intermediate"].append(
                (decode(batch_outputs(intermediate, index), cfg), *item[2])
            )
        if adaptive is not None:
            frames["adaptive"].append(
                (decode(batch_outputs(adaptive, index), cfg), *item[2])
            )
        if late is not None:
            offset = index * nodes
            local = [
                decode(batch_outputs(late, offset + node), cfg) for node in range(nodes)
            ]
            frames["late"].append(
                (
                    nms(
                        [box for group in local for box in group],
                        cfg["evaluation"]["nms_iou"],
                        cfg["evaluation"]["late_max_detections"],
                    ),
                    *item[2],
                )
            )
    return losses, frames


def validate_examples(model, examples, cfg, device, retries):
    """Validate one prepared batch, splitting only that batch after CUDA OOM."""
    try:
        return validate_batch(model, examples, cfg, device)
    except (torch.OutOfMemoryError, RuntimeError) as error:
        if not cuda_oom(error, device):
            raise
        oom_message = str(error)
    # Retry only after the exception and its traceback have left scope.
    gc.collect()
    release_cuda_memory(device)
    if len(examples) > 1:
        midpoint = len(examples) // 2
        print(
            f"[{utc_now()}] CUDA OOM during validation; reducing current batch "
            f"{len(examples)} -> {midpoint}+{len(examples)-midpoint}",
            flush=True,
        )
        left = validate_examples(model, examples[:midpoint], cfg, device, retries)
        right = validate_examples(model, examples[midpoint:], cfg, device, retries)
        return left[0] + right[0], {
            view: left[1][view] + right[1][view] for view in validation_views(cfg)
        }
    if retries > 0:
        wait_for_resources(cfg, device, "single-example validation CUDA OOM")
        return validate_examples(model, examples, cfg, device, retries - 1)
    raise ValueError(
        "CUDA cannot fit one validation example after resource retries; reduce "
        "resolution, hidden_channels, or node_counts, or use training.device=cpu. "
        f"Original error: {oom_message}"
    )


def validation_views(cfg):
    return [*cfg["evaluation"]["strategies"], "single"]


def _mean(values):
    return float(np.mean(values)) if values else 0.0


def validation_frames(model, samples, cfg, device, batch_size):
    """Per-sample validation losses and ``{view: [(predictions, visible,
    ignored)]}`` for every validation view."""
    losses = []
    frames = {view: [] for view in validation_views(cfg)}
    grouped = defaultdict(list)
    for sample in samples:
        grouped[len(sample["nodes"])].append(sample)
    size = batch_size
    batches = [
        group[start : start + size]
        for group in grouped.values()
        for start in range(0, len(group), size)
    ]
    if not batches:
        raise ValueError("Validation split contains no samples")
    workers = cfg["training"]["preprocessing_workers"]
    retries = cfg["training"]["recovery"]["cuda_oom_retries"]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for batch, examples in prefetched_batches(
            executor, batches, submit_validation_batch, cfg
        ):
            batch_losses, batch_frames = validate_examples(
                model, examples, cfg, device, retries
            )
            losses.extend(batch_losses)
            for view in frames:
                frames[view].extend(batch_frames[view])
    return losses, frames


def _validation_metrics(model, samples, cfg, device, batch_size):
    losses, frames = validation_frames(model, samples, cfg, device, batch_size)
    # Each view is scored on the boxes its nodes support (visible scope), the
    # quantity the visibility-filtered training targets optimize.
    # Selection metrics use the primary IoU level only. Averaging in the
    # strict level (AP@0.7 is near zero early in training) would halve the
    # number without adding signal. Every level is still logged.
    details = {}
    levels = cfg["evaluation"]["iou_thresholds"]
    for view, view_frames in frames.items():
        rows = evaluate(
            view_frames,
            cfg["dataset"]["classes"],
            levels,
            cfg["evaluation"]["score_threshold"],
            cfg["evaluation"]["class_iou_thresholds"],
        )
        for level, nominal in enumerate(levels):
            level_rows = [r for r in rows if r["iou_level"] == level]
            suffix = "" if level == 0 else f"_iou{nominal:g}"
            for metric, key in (
                ("map", "ap_3d"),
                ("precision", "precision"),
                ("recall", "recall"),
                ("f1", "f1"),
            ):
                # A class without ground truth has no AP or recall, but its
                # false positives give precision 0; leave it out of every mean.
                details[f"val_{metric}_{view}{suffix}"] = _mean(
                    [
                        r[key]
                        for r in level_rows
                        if r[key] is not None and r["ground_truth"]
                    ]
                )
        for result in rows:
            if result["iou_level"] != 0:
                continue
            label = result["class"]
            details[f"val_ap_{view}_{label}"] = result["ap_3d"]
            details[f"val_precision_{view}_{label}"] = result["precision"]
            details[f"val_recall_{view}_{label}"] = result["recall"]
            details[f"val_f1_{view}_{label}"] = result["f1"]
    # Weight the single-node view equally with the cooperative strategies so a
    # checkpoint is not chosen for multi-node quality alone, which would bias
    # the single-versus-cooperative comparison made on the test split.
    strategies = cfg["evaluation"]["strategies"]
    for metric in ("map", "precision", "recall", "f1"):
        cooperative = _mean([details[f"val_{metric}_{name}"] for name in strategies])
        details[f"val_{metric}_cooperative"] = cooperative
        details[f"val_{metric}"] = 0.5 * (details[f"val_{metric}_single"] + cooperative)
    val_loss = float(np.mean(losses))
    return val_loss, details["val_map"], details["val_recall"], details


def validation_metrics(model, samples, cfg, device, runtime_batches=None, retries=None):
    """Validate in batches; OOM recovery is local to each failing batch."""
    if runtime_batches is None:
        runtime_batches = {"evaluation": cfg["evaluation"]["batch_size"]}
    return _validation_metrics(
        model, samples, cfg, device, runtime_batches["evaluation"]
    )


def patience_progress(history, min_delta):
    """Early-stopping reference and stale-epoch count implied by ``history``."""
    reference, stale = -float("inf"), 0
    for row in history:
        score = float(row["selection_score"])
        if score > reference + min_delta:
            reference, stale = score, 0
        else:
            stale += 1
    return reference, stale


def selection_score(cfg, val_loss, val_map, val_recall):
    weights = cfg['training']['checkpointing']['weights']
    values = {'val_loss_quality': 1.0 / (1.0 + val_loss),
              'val_map': val_map, 'val_recall': val_recall}
    total = sum(weights.values())
    return sum(weights[key] * values[key] for key in values) / total


def prune_checkpoints(directory, keep):
    candidates = []
    for metadata in directory.glob('epoch_*.json'):
        try:
            value = json.loads(metadata.read_text())
        except (OSError, ValueError) as error:
            print(
                f"Ignoring unreadable checkpoint metadata {metadata}: {error}",
                flush=True,
            )
            continue
        checkpoint = metadata.with_suffix('.pt')
        if checkpoint.exists():
            candidates.append((float(value['selection_score']), int(value['epoch']), checkpoint, metadata))
    candidates.sort(reverse=True)
    for _, _, checkpoint, metadata in candidates[keep:]:
        checkpoint.unlink()
        metadata.unlink()
    return candidates[:keep]


def wait_for_resources(cfg, device, reason='resource check'):
    settings = cfg['training']['recovery']
    if not settings['wait_for_resources']:
        return
    started = time.monotonic()
    disk_required = settings['minimum_free_disk_gb'] * 1_000_000_000
    gpu_required = settings['minimum_free_gpu_gb'] * 1_000_000_000
    output = Path(cfg['output']['root'])
    output.mkdir(parents=True, exist_ok=True)
    while True:
        disk_free = shutil.disk_usage(output).free
        gpu_free = gpu_held = None
        if device.type == 'cuda':
            physical_free, _ = torch.cuda.mem_get_info(device)
            gpu_held = torch.cuda.memory_allocated(device)
            reclaimable = max(0, torch.cuda.memory_reserved(device) - gpu_held)
            gpu_free = physical_free + reclaimable
        disk_ok = disk_free >= disk_required
        gpu_ok = gpu_free is None or gpu_free >= gpu_required
        if disk_ok and gpu_ok:
            return
        elapsed = time.monotonic() - started
        timeout = settings['timeout_minutes'] * 60
        details = f'disk_free={disk_free/1e9:.2f}GB'
        if gpu_free is not None:
            # Waiting frees only memory other processes hold; a large value
            # here means this process itself is holding tensors.
            details += (
                f" gpu_free={gpu_free/1e9:.2f}GB"
                f" gpu_held_by_this_process={gpu_held/1e9:.2f}GB"
            )
        if timeout and elapsed >= timeout:
            raise TimeoutError(f'Resource wait timed out ({reason}): {details}')
        print(f'[{utc_now()}] WAITING ({reason}): {details}; retrying in '
              f'{settings["poll_seconds"]}s', flush=True)
        time.sleep(settings['poll_seconds'])


def atomic_torch_save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def fingerprint(cfg):
    return hashlib.sha256(json.dumps(cfg['dataset'],sort_keys=True).encode()).hexdigest()


def signature(cfg):
    """Fields that determine trained weights; operational settings are excluded."""
    d, t = cfg["dataset"], cfg["training"]
    dataset_keys = (
        "type",
        "weather",
        "nodes",
        "frame_stride",
        "max_frames_per_slice",
        "annotation_file",
        "yaw_unit",
        "classes",
        "class_aliases",
        "bounds",
        "resolution",
        "objects_per_cell",
        "intensity_max",
        "use_intensity",
        "height_slices",
        "height_slice_m",
        "train_frame_stride",
        "split_mode",
        "random_split",
    )
    training_keys = (
        "node_channels",
        "backbone_channels",
        "node_blocks",
        "backbone_blocks",
        "fusion",
        "output_stride",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "mixed_precision",
        "amp_dtype",
        "modes",
        "node_counts",
        "augmentation",
        "min_target_points",
        "repeat_factor_threshold",
        "stale",
    )
    return {
        "model_version": MODEL_VERSION,
        "dataset": {
            **{key: d.get(key) for key in dataset_keys},
            "splits": {name: d["splits"][name] for name in ("train", "val")},
        },
        "training": {key: t.get(key) for key in training_keys},
        # Evaluation budgets, selectors, and serialization widths do not change
        # learned weights. Keeping them out permits post-training sweeps without
        # invalidating an otherwise compatible checkpoint.
        "adaptive_fusion": {
            "training_budgets": cfg.get("adaptive_fusion", {}).get("training_budgets")
        },
    }


def audit_split_isolation(samples, split_mode="sessions"):
    """Fail closed when any recording or source artifact crosses a split.

    Under ``random_frames`` (CoInfra's published protocol) sessions are shared
    across splits by design; every frame and source file must still belong to
    exactly one split.
    """
    owners = {}
    sessions = {}
    for sample in samples:
        split = sample["split"]
        session = sample["session"]
        previous = sessions.setdefault(session, split)
        if previous != split and split_mode != "random_frames":
            raise ValueError(
                f"Data leakage: session {session!r} occurs in {previous} and {split}"
            )
        sources = [("sample_id", sample["id"]), ("ground_truth", sample["gt"])]
        sources.extend(
            (f"cloud:{node}", value["cloud"]) for node, value in sample["nodes"].items()
        )
        for kind, value in sources:
            identity = value if kind == "sample_id" else str(Path(value).resolve())
            key = (kind, identity)
            earlier = owners.setdefault(key, split)
            if earlier != split:
                raise ValueError(
                    f"Data leakage: {kind} {identity!r} occurs in {earlier} and {split}"
                )
    return {
        "samples": len(samples),
        "sessions": len(sessions),
        "sources": len(owners),
        "split_mode": split_mode,
        "status": (
            "passed (frame-level random split; sessions shared across splits "
            "by design, as in the CoInfra benchmark)"
            if split_mode == "random_frames"
            else "passed"
        ),
    }


def checkpoint_compatible(state, cfg):
    saved = state.get("signature")
    current = signature(cfg)
    if saved == current:
        return True
    saved_cfg = state.get("config")
    return isinstance(saved_cfg, dict) and signature(saved_cfg) == current


def quarantine_seed_directory(seed_directory, reason):
    """Preserve stale seed artifacts and leave the seed ready for retraining."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantine_root = seed_directory.parent / "stale_checkpoints"
    quarantine_root.mkdir(parents=True, exist_ok=True)
    destination = quarantine_root / f"{seed_directory.name}_{timestamp}_{os.getpid()}"
    suffix = 1
    while destination.exists():
        destination = quarantine_root / (
            f"{seed_directory.name}_{timestamp}_{os.getpid()}_{suffix}"
        )
        suffix += 1
    seed_directory.replace(destination)
    print(
        f"[{utc_now()}] QUARANTINED {seed_directory} -> {destination} ({reason}); "
        "retraining this seed",
        flush=True,
    )
    return destination


def handle_incompatible_seed(cfg, seed_directory, reason):
    action = cfg["training"]["recovery"]["on_incompatible"]
    if action == "fail":
        raise ValueError(
            f"{seed_directory} {reason}; choose another output directory or set "
            "training.recovery.on_incompatible=quarantine"
        )
    return quarantine_seed_directory(seed_directory, reason)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluation_fingerprint(cfg, checkpoint):
    return hashlib.sha256(
        json.dumps(
            {
                "evaluation_version": EVALUATION_VERSION,
                "checkpoint_sha256": file_sha256(checkpoint),
                "dataset": signature(cfg)["dataset"],
                "evaluation": cfg["evaluation"],
                "adaptive_fusion": cfg["adaptive_fusion"],
                "latency_experiment": cfg["latency_experiment"],
                "deployment_assumptions": cfg["deployment_assumptions"],
                "mixed_precision": cfg["training"]["mixed_precision"],
                "amp_dtype": cfg["training"]["amp_dtype"],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def device_for(cfg):
    name = cfg['training']['device']
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else name
    )
    if device.type == "cuda":
        # BEV tensor shapes repeat heavily, so benchmarked kernels and TF32 improve
        # throughput without changing storage or checkpoint compatibility.
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        properties = torch.cuda.get_device_properties(device)
        print(
            f"Using GPU {properties.name} ({properties.total_memory / 1e9:.1f} GB); "
            f'train_batch={cfg["training"]["batch_size"]}, '
            f'eval_batch={cfg["evaluation"]["batch_size"]}, '
            f'amp={cfg["training"]["mixed_precision"]}',
            flush=True,
        )
    else:
        print(f"Using {device.type.upper()} (CUDA unavailable or disabled)", flush=True)
    return device


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


def preprocess(cfg):
    samples = discover(cfg)
    if cfg.get("diagnostic_overfit"):
        isolation = {
            "samples": len(samples),
            "status": "intentionally skipped for --overfit diagnostic",
        }
    else:
        isolation = audit_split_isolation(samples, cfg["dataset"]["split_mode"])
    validate_node_capacity(samples, cfg)
    counts = defaultdict(int)
    for sample in samples:
        counts[f'{sample["split"]}/{sample["weather"]}'] += 1
    for split in ("train", "val", "test"):
        if not any(s["split"] == split for s in samples):
            raise ValueError(f"No {split} samples after weather filters")
    output = Path(cfg["output"]["root"])
    write_json(
        output / "index.json",
        {"dataset_fingerprint": fingerprint(cfg), "samples": samples},
    )
    write_json(output / "resolved_config.json", cfg)
    write_json(output / "environment.json", environment())
    write_json(output / "inventory.json", dict(counts))
    write_json(output / "split_isolation.json", isolation)
    print(json.dumps(dict(counts), indent=2))
    return samples


def environment():
    """Software and hardware that produced the results, for latency context."""
    cuda = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda else None,
        "gpus": (
            [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if cuda
            else []
        ),
        "numpy": np.__version__,
    }


def indexed(cfg):
    path = Path(cfg['output']['root'])/'index.json'
    if not path.exists():
        return preprocess(cfg)
    value = json.loads(path.read_text())
    if value['dataset_fingerprint'] != fingerprint(cfg):
        raise ValueError('Dataset config changed. Rerun --stage preprocess to rebuild the index.')
    validate_node_capacity(value['samples'], cfg)
    return value['samples']


def validate_node_capacity(samples, cfg):
    """Ensure every test session can support requested node counts before training."""
    sessions = defaultdict(list)
    for sample in samples:
        if sample['split'] == 'test':
            sessions[sample['session']].append(sample)
    requested = max(cfg['evaluation']['node_counts'])
    for session, frames in sessions.items():
        common = set.intersection(*(set(frame['nodes']) for frame in frames))
        if len(common) < requested:
            raise ValueError(f'Test session {session} has {len(common)} common LiDAR nodes, but '
                             f'evaluation.node_counts requests {requested}; adjust the configuration')


def inspect(cfg):
    samples = preprocess(cfg)
    annotation_counts = {}
    class_inventory = {
        split: {label: 0 for label in cfg["dataset"]["classes"]}
        for split in ("train", "val", "test")
    }
    grid = output_shape(cfg["dataset"], cfg["training"]["output_stride"])
    crowded = None
    for sample in samples:
        boxes = annotations(sample["gt"], cfg["dataset"])
        needed = validate_target_centers(boxes, cfg["dataset"], sample["id"], grid)
        if needed and (crowded is None or needed[0] > crowded[0]):
            crowded = needed
        annotation_counts[sample["id"]] = len(boxes)
        for box in boxes:
            class_inventory[sample["split"]][box.label] += 1
    unvalidated = [
        label
        for label in cfg["dataset"]["classes"]
        if class_inventory["val"][label] == 0 and class_inventory["test"][label] > 0
    ]
    if unvalidated:
        warnings.warn(
            "The validation split has no ground truth for "
            f"{', '.join(unvalidated)}; checkpoint selection and operating-threshold "
            "tuning cannot see these classes, so they keep evaluation.score_threshold",
            RuntimeWarning,
        )
    slots = cfg["dataset"].get("objects_per_cell", 1)
    if crowded is not None and crowded[0] > slots:
        raise ValueError(
            f"More than {slots} same-class centers occupy one grid cell during "
            f"inspection; set dataset.objects_per_cell to at least {crowded[0]}. "
            f"Most crowded cell: {crowded[1]}"
        )
    report = []
    seen = set()
    intensity_sample = []
    sampler = np.random.default_rng(0)
    for sample in samples:
        key = (sample['session'],sample['slice'])
        if key in seen:
            continue
        seen.add(key)
        clouds = {node:load_cloud(value,cfg['dataset']['bounds']) for node,value in sample['nodes'].items()}
        intensities = np.concatenate(
            [cloud[:, 3] for cloud in clouds.values()] or [np.zeros(0, np.float32)]
        )
        entry = {
            "sample": sample["id"],
            "points_in_bounds": {n: len(p) for n, p in clouds.items()},
            "boxes_in_bounds": annotation_counts[sample["id"]],
        }
        if len(intensities):
            entry["intensity_p50"] = float(np.quantile(intensities, 0.5))
            entry["intensity_p99"] = float(np.quantile(intensities, 0.99))
            intensity_sample.append(
                sampler.choice(
                    intensities, min(len(intensities), 20_000), replace=False
                )
            )
        report.append(entry)
        if not any(len(p) for p in clouds.values()):
            raise ValueError(
                f'All transformed points outside configured bounds at {sample["id"]}'
            )
    validate_intensity_scale(intensity_sample, cfg["dataset"])
    write_json(Path(cfg["output"]["root"]) / "inspection.json", report)
    write_json(Path(cfg["output"]["root"]) / "class_inventory.json", class_inventory)
    print(
        f"Validated annotations in {len(samples)} frames and point transforms in "
        f"{len(report)} representative slices; see inspection.json and class_inventory.json"
    )


def inputs(sample, nodes, cfg):
    return [load_cloud(sample['nodes'][n],cfg['dataset']['bounds']) for n in nodes]


def validate_target_centers(boxes, cfg, sample_id, shape):
    """Find the most same-class centers sharing one prediction-grid cell.

    Uses the same binning as ``targets`` on the stride-reduced output grid, so
    inspection and training agree on what fits ``objects_per_cell``. Returns
    ``(count, description)`` for the most crowded cell, or None.
    """
    occupied = defaultdict(list)
    for box in boxes:
        cell = target_cell(box, cfg, shape)
        if cell is not None:
            occupied[cell[:3]].append(box)
    if not occupied:
        return None
    (label, ix, iy), colocated = max(occupied.items(), key=lambda item: len(item[1]))
    centers = ", ".join(f"({box.x:.3f}, {box.y:.3f})" for box in colocated)
    return len(colocated), (
        f"sample={sample_id!r}, class={cfg['classes'][label]!r}, "
        f"output cell=(x={ix}, y={iy}), centers={centers}"
    )


def validate_intensity_scale(samples, cfg):
    """Fail when the intensity channel is uninformative or grossly mis-scaled."""
    if not cfg.get("use_intensity", True) or not samples:
        return
    values = np.concatenate(samples)
    p1, p99 = (float(value) for value in np.quantile(values, [0.01, 0.99]))
    if p99 - p1 <= 1e-6 * max(1.0, abs(p99)):
        raise ValueError(
            f"Point intensity is constant (p1={p1:g}, p99={p99:g}), so the intensity "
            "BEV channel carries no information; set dataset.use_intensity: false"
        )
    maximum = cfg.get("intensity_max")
    if not maximum:
        return
    if p99 > maximum or p99 < 0.01 * maximum:
        raise ValueError(
            f"Point intensity p99={p99:g} (p50={float(np.quantile(values, 0.5)):g}) "
            f"does not match dataset.intensity_max={maximum:g}; set intensity_max "
            "to the sensor's full-scale intensity"
        )


def array_inputs(clouds, cfg, mode):
    if mode == "early":
        return rasterize(np.concatenate(clouds), cfg["dataset"])[None]
    return np.stack([rasterize(p, cfg["dataset"]) for p in clouds])


def device_tensor(array, device):
    value = torch.from_numpy(array)
    if device.type == "cuda":
        value = value.pin_memory()
    return value.to(device, non_blocking=device.type == "cuda")


def repeat_factors(cfg, training):
    """Per-frame repeat factors for class-balanced sampling (LVIS).

    ``r = max(1, max_c sqrt(t / f_c))`` over the classes ``c`` in the frame,
    where ``f_c`` is the fraction of training frames containing ``c`` and
    ``t`` is ``training.repeat_factor_threshold``. Frames with rare classes
    are seen more often without reweighting the loss.
    """
    threshold = cfg["training"]["repeat_factor_threshold"]
    if not threshold:
        return {sample["id"]: 1.0 for sample in training}
    path = Path(cfg["output"]["root"]) / "repeat_factors.json"
    key = {"fingerprint": fingerprint(cfg), "threshold": threshold}
    if path.exists():
        cached = json.loads(path.read_text())
        if cached.get("key") == key:
            return cached["factors"]
    present = {
        sample["id"]: {box.label for box in annotations(sample["gt"], cfg["dataset"])}
        for sample in training
    }
    frequency = {
        label: sum(label in labels for labels in present.values()) / len(present)
        for label in cfg["dataset"]["classes"]
    }
    factors = {
        sample_id: max(
            [1.0] + [math.sqrt(threshold / frequency[label]) for label in labels]
        )
        for sample_id, labels in present.items()
    }
    write_json(
        path, {"key": key, "class_frame_frequency": frequency, "factors": factors}
    )
    return factors


def repeated_order(training, factors, rng):
    """Each frame ``floor(r)`` times plus once more with probability ``r % 1``."""
    order = []
    for sample in training:
        factor = factors[sample["id"]]
        whole = int(factor)
        fraction = factor - whole
        order.extend(
            [sample] * (whole + int(bool(fraction) and rng.random() < fraction))
        )
    return order


def slice_histories(samples):
    """Frames of each recording slice in time order, for stale-message lookup."""
    histories = defaultdict(list)
    for sample in sorted(samples, key=lambda item: float(item["timestamp"])):
        histories[(sample["session"], sample["slice"])].append(sample)
    return histories


def training_plan(order, cfg, rng, batch_size=None, histories=None):
    """Batches of ``(sample, mode, nodes, augmentation, sources)`` plans.

    Node 0 of each plan is the receiver. With ``training.stale.probability``,
    every collaborator of an intermediate or adaptive sample is replaced, with
    that probability, by one of its own earlier frames from ``histories``, as
    a message delayed by the network would be; ``sources`` maps such nodes to
    the earlier sample.
    """
    stale = cfg["training"]["stale"]
    buckets = defaultdict(list)
    for sample in order:
        available = sorted(sample["nodes"])
        counts = [n for n in cfg["training"]["node_counts"] if n <= len(available)]
        if not counts:
            raise ValueError("No training node count fits available data")
        mode = rng.choice(cfg["training"]["modes"])
        nodes = rng.sample(available, 1 if mode == "single" else rng.choice(counts))
        model_mode = "early" if mode == "single" else mode
        if model_mode == "adaptive":
            budget = rng.choice(cfg["adaptive_fusion"]["training_budgets"])
            model_mode = f"adaptive@{budget:g}"
        augmentation = sample_parameters(rng, cfg["training"]["augmentation"])
        sources = {}
        if model_mode != "early" and len(nodes) > 1 and stale["probability"]:
            history = histories[(sample["session"], sample["slice"])]
            for node in nodes[1:]:
                if rng.random() >= stale["probability"]:
                    continue
                age = rng.randint(1, stale["max_age_frames"])
                earlier = past_frame(history, sample, age)
                if earlier is not None and node in earlier["nodes"]:
                    sources[node] = earlier
        buckets[(model_mode, len(nodes))].append(
            (sample, model_mode, nodes, augmentation, sources)
        )
    batches = []
    size = batch_size or cfg["training"]["batch_size"]
    for plans in buckets.values():
        batches.extend(
            plans[start : start + size] for start in range(0, len(plans), size)
        )
    rng.shuffle(batches)
    return batches


def support_counts(clouds, boxes, cfg):
    """Points from all of ``clouds`` inside each box."""
    bounds = cfg["dataset"]["bounds"]
    counts = np.zeros(len(boxes), dtype=np.int64)
    for cloud in clouds:
        counts += box_point_counts(cloud, boxes, bounds)
    return counts


def supported_boxes(boxes, counts, minimum):
    return [box for box, count in zip(boxes, counts) if count >= minimum]


def visibility_scope(boxes, counts, cfg):
    """Split boxes into scored (visible) and ignored truths for these nodes."""
    minimum = cfg["evaluation"]["min_visible_points"]
    visible = [box for box, count in zip(boxes, counts) if count >= minimum]
    ignored = [box for box, count in zip(boxes, counts) if count < minimum]
    return visible, ignored


def prepare_training_example(plan, cfg):
    """``(bev, target, ages)`` for one plan; ``ages`` are message ages in s."""
    sample, mode, nodes, augmentation, sources = plan
    bounds = cfg["dataset"]["bounds"]
    clouds = [
        load_cloud(sources.get(node, sample)["nodes"][node], bounds) for node in nodes
    ]
    ages = [
        (
            float(sample["timestamp"]) - float(sources[node]["timestamp"])
            if node in sources
            else 0.0
        )
        for node in nodes
    ]
    boxes = annotations(sample["gt"], cfg["dataset"])
    minimum = cfg["training"]["min_target_points"]
    if minimum:
        # The global ROI labels objects no chosen node can see. Supervising
        # those would teach the detector to hallucinate from priors, so a
        # box is a target only with enough points from the chosen nodes.
        boxes = supported_boxes(boxes, support_counts(clouds, boxes, cfg), minimum)
    shape = output_shape(cfg["dataset"], cfg["training"]["output_stride"])
    if augmentation is not None:
        moved_clouds, moved_boxes = augment_scene(
            clouds, boxes, augmentation, cfg["dataset"]
        )
        try:
            target = targets(moved_boxes, cfg["dataset"], shape, sample_id=sample["id"])
        except TargetCollisionError:
            # Inspection proved the original labels fit; only the rotation or
            # scale crowded this cell, so train this sample unaugmented.
            augmentation = None
        else:
            clouds = moved_clouds
    if augmentation is None:
        target = targets(boxes, cfg["dataset"], shape, sample_id=sample["id"])
    bev = array_inputs(clouds, cfg, mode)
    if mode == "early":
        # The merged cloud is one fresh node.
        bev, ages = bev[0], [0.0]
    return bev, target, np.asarray(ages, dtype=np.float32)


def stack_training_examples(examples):
    return {
        "bev": np.stack([example[0] for example in examples]),
        "target": tuple(
            np.stack([example[1][index] for example in examples]) for index in range(3)
        ),
        "ages": np.stack([example[2] for example in examples]),
    }


def submit_training_batch(executor, plans, cfg):
    def prepare_batch():
        return stack_training_examples(
            [prepare_training_example(plan, cfg) for plan in plans]
        )

    # Stack the large dense targets on a preprocessing worker instead of
    # blocking the GPU-owning training thread.
    return [executor.submit(prepare_batch)]


def slice_training_batch(batch, start, stop):
    return {
        "bev": batch["bev"][start:stop],
        "target": tuple(value[start:stop] for value in batch["target"]),
        "ages": batch["ages"][start:stop],
    }


def cuda_oom(error, device):
    return device.type == "cuda" and (
        isinstance(error, torch.OutOfMemoryError)
        or "out of memory" in str(error).lower()
    )


def release_cuda_memory(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()


def nonfinite_tensor_summary(tensors):
    """Compact diagnostics for tensors implicated in a nonfinite loss."""
    bad = []
    for name, value in tensors:
        invalid = ~torch.isfinite(value)
        if invalid.any():
            bad.append(f"{name}={int(invalid.sum())}/{value.numel()}")
    return ",".join(bad) or "none"


def train_step(model, optimizer, scaler, examples, mode, cfg, device, context):
    """One optimizer step on a stacked batch.

    The batch tensors and autograd graph live only in this frame, so a CUDA OOM
    raised here frees them with its traceback before the caller retries.
    """
    bev = device_tensor(examples["bev"], device)
    target = tuple(device_tensor(value, device) for value in examples["target"])
    ages = device_tensor(examples["ages"], device)
    optimizer.zero_grad(set_to_none=True)
    amp_enabled = mixed_precision(cfg, device)
    with autocast(cfg, device):
        outputs = model(bev, mode, ages)
        value = loss(outputs, target, cell_size(cfg))
    if not torch.isfinite(value) and amp_enabled:
        # fp16 can still overflow on an unusual batch (bf16 cannot). Retry
        # only that batch in fp32, without disabling AMP globally. Drop the
        # AMP graph first; keeping it alive through the fp32 pass needs
        # memory for both graphs at once.
        outputs = value = None
        optimizer.zero_grad(set_to_none=True)
        with autocast(cfg, device, enabled=False):
            outputs = model(bev, mode, ages)
            value = loss(outputs, target, cell_size(cfg))
        if torch.isfinite(value):
            print(
                f"[{utc_now()}] AMP nonfinite loss recovered in fp32 "
                f"({context}, mode={mode})",
                flush=True,
            )
    if not torch.isfinite(value):
        tensors = [("bev", bev)]
        tensors.extend((f"target[{index}]", item) for index, item in enumerate(target))
        tensors.extend((f"output[{index}]", item) for index, item in enumerate(outputs))
        raise ValueError(
            "Nonfinite training loss "
            f"({context}, mode={mode}, amp={amp_enabled}, "
            f"nonfinite={nonfinite_tensor_summary(tensors)})"
        )
    scaler.scale(value).backward()
    scaler.unscale_(optimizer)
    try:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        scaler.step(optimizer)
    finally:
        # GradScaler records a per-optimizer STEPPED state before invoking the
        # optimizer. Clear that state so an OOM inside optimizer.step does not
        # make the smaller retry fail with "step() has already been called".
        scaler.update()
    return [float(value.detach())] * len(examples["bev"])


def train_examples(
    model,
    optimizer,
    scaler,
    examples,
    mode,
    cfg,
    device,
    context,
    runtime_batches,
    retries,
):
    """Train the largest batch that fits, recursively splitting on CUDA OOM."""
    if isinstance(examples, list):
        examples = stack_training_examples(examples)
    try:
        return train_step(
            model, optimizer, scaler, examples, mode, cfg, device, context
        )
    except (torch.OutOfMemoryError, RuntimeError) as error:
        if not cuda_oom(error, device):
            raise
        oom_message = str(error)
    # The failed attempt's tensors and graph left scope with the traceback, so
    # each parent frame of the split recursion holds only CPU arrays.
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    release_cuda_memory(device)
    example_count = len(examples["bev"])
    if example_count == 1:
        if retries > 0:
            wait_for_resources(cfg, device, f"{context} single-example CUDA OOM")
            return train_examples(
                model,
                optimizer,
                scaler,
                examples,
                mode,
                cfg,
                device,
                context,
                runtime_batches,
                retries - 1,
            )
        raise ValueError(
            f"CUDA {mode} OOM retries exhausted ({context}). Reduce batch_size, "
            "resolution, hidden_channels, or node_counts, or set training.device=cpu. "
            f"Original error: {oom_message}"
        )
    midpoint = example_count // 2
    runtime_batches["training"] = min(runtime_batches["training"], midpoint)
    print(
        f"[{utc_now()}] CUDA OOM ({context}); reducing batch "
        f"{example_count} -> {midpoint}+{example_count-midpoint}",
        flush=True,
    )
    return train_examples(
        model,
        optimizer,
        scaler,
        slice_training_batch(examples, 0, midpoint),
        mode,
        cfg,
        device,
        context,
        runtime_batches,
        retries,
    ) + train_examples(
        model,
        optimizer,
        scaler,
        slice_training_batch(examples, midpoint, example_count),
        mode,
        cfg,
        device,
        context,
        runtime_batches,
        retries,
    )


def cell_size(cfg):
    return grid_cell_size(cfg["dataset"], cfg["training"]["output_stride"])


def prepare_validation_example(sample, cfg):
    """Inputs, scored/ignored truths, and targets for the all-node and
    single-node validation views.

    Returns ``(early, per_node, all_node_truths, all_node_target,
    single_node, single_node_truths, single_node_target)``. Each truths entry
    is a ``(visible, ignored)`` pair for the nodes of that view.
    """
    nodes = sorted(sample["nodes"])
    clouds = inputs(sample, nodes, cfg)
    early = array_inputs(clouds, cfg, "early")[0]
    per_node = array_inputs(clouds, cfg, "intermediate")
    gt = annotations(sample["gt"], cfg["dataset"])
    bounds = cfg["dataset"]["bounds"]
    counts = np.stack([box_point_counts(cloud, gt, bounds) for cloud in clouds])
    shape = output_shape(cfg["dataset"], cfg["training"]["output_stride"])
    minimum = cfg["training"]["min_target_points"]
    # The single-node view rotates deterministically through every node.
    single_node = stable_random_seed("validation-single", sample["id"]) % len(nodes)
    views = []
    for view_counts in (counts.sum(axis=0), counts[single_node]):
        target = targets(
            supported_boxes(gt, view_counts, minimum),
            cfg["dataset"],
            shape,
            sample_id=sample["id"],
        )
        views.append((visibility_scope(gt, view_counts, cfg), target))
    (early_truths, early_target), (single_truths, single_target) = views
    return (
        early,
        per_node,
        early_truths,
        early_target,
        single_node,
        single_truths,
        single_target,
    )


def submit_validation_batch(executor, samples, cfg):
    return [
        executor.submit(prepare_validation_example, sample, cfg) for sample in samples
    ]


def prefetched_batches(executor, batches, submit, cfg, depth=3):
    """Prepare a bounded number of batches while the current batch runs on GPU."""
    batches = iter(batches)
    pending = deque()
    for _ in range(depth):
        try:
            batch = next(batches)
        except StopIteration:
            break
        pending.append((batch, submit(executor, batch, cfg)))
    while pending:
        batch, futures = pending.popleft()
        try:
            following = next(batches)
        except StopIteration:
            following = None
        if following is not None:
            pending.append((following, submit(executor, following, cfg)))
        yield batch, [future.result() for future in futures]


def batch_outputs(outputs, index):
    return outputs[0][index : index + 1], outputs[1][index : index + 1]


def make_model(cfg, device):
    t = cfg["training"]
    return DRIVEDetector(
        cfg["dataset"]["classes"],
        bev_channels(cfg["dataset"]),
        cfg["dataset"].get("objects_per_cell", 1),
        node_channels=t["node_channels"],
        backbone_channels=t["backbone_channels"],
        fusion=t["fusion"],
        node_blocks=t["node_blocks"],
        backbone_blocks=tuple(t["backbone_blocks"]),
        output_stride=t["output_stride"],
    ).to(device)


def mixed_precision(cfg, device):
    return device.type == "cuda" and cfg["training"]["mixed_precision"]


def amp_dtype(cfg, device):
    """bf16 keeps fp32's exponent range, so activations cannot overflow as in fp16."""
    choice = cfg["training"]["amp_dtype"]
    if choice == "auto":
        bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
        return torch.bfloat16 if bf16 else torch.float16
    return getattr(torch, choice)


def autocast(cfg, device, enabled=True):
    return torch.autocast(
        device_type=device.type,
        dtype=amp_dtype(cfg, device),
        enabled=enabled and mixed_precision(cfg, device),
    )


def gradient_scaling(cfg, device):
    """Only fp16 needs loss scaling; bf16 has fp32's range."""
    return mixed_precision(cfg, device) and amp_dtype(cfg, device) == torch.float16


def train(cfg, overwrite=False):
    samples = indexed(cfg)
    training = [s for s in samples if s['split']=='train']
    validation = [s for s in samples if s['split']=='val']
    output, device = Path(cfg['output']['root']), device_for(cfg)
    wait_for_resources(cfg, device, 'before training')
    runtime_batches = {
        "training": cfg["training"]["batch_size"],
        "evaluation": cfg["evaluation"]["batch_size"],
    }
    factors = repeat_factors(cfg, training)
    histories = slice_histories(training)
    for seed in cfg['training']['seeds']:
        path = output/f'seed_{seed}'/'best.pt'
        latest = path.parent/'latest.pt'
        completion = path.parent/'complete.json'
        if path.exists() and completion.exists() and not overwrite:
            completion_state = json.loads(completion.read_text())
            state = torch.load(path,map_location='cpu',weights_only=True)
            if not checkpoint_compatible(state, cfg):
                handle_incompatible_seed(
                    cfg, path.parent, "contains a checkpoint from a different config"
                )
            else:
                completed_epochs = int(completion_state.get("epochs", 0))
                if completed_epochs >= cfg["training"][
                    "epochs"
                ] or completion_state.get("stopped_early", False):
                    print(
                        f"Skipping existing checkpoint {path}; --overwrite retrains it"
                    )
                    continue
                if not latest.exists():
                    raise ValueError(
                        f"Cannot extend seed {seed}: missing recovery checkpoint {latest}"
                    )
                print(
                    f'Extending seed={seed} from {completed_epochs} to {cfg["training"]["epochs"]} epochs'
                )
        if latest.exists() and not overwrite:
            recovery_state = torch.load(latest, map_location="cpu", weights_only=True)
            if not checkpoint_compatible(recovery_state, cfg):
                handle_incompatible_seed(
                    cfg,
                    path.parent,
                    "contains a recovery checkpoint from a different config",
                )
        if overwrite:
            for stale in list(path.parent.glob('epoch_*.pt')) + list(path.parent.glob('epoch_*.json')):
                stale.unlink()
            for stale in (completion, path, latest):
                if stale.exists():
                    stale.unlink()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        rng = random.Random(seed)
        model = make_model(cfg, device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["training"]["learning_rate"],
            weight_decay=cfg["training"]["weight_decay"],
        )
        scaler = torch.amp.GradScaler("cuda", enabled=gradient_scaling(cfg, device))
        scheduler_cfg = cfg["training"]["scheduler"]
        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=scheduler_cfg["factor"],
                patience=scheduler_cfg["patience"],
                min_lr=scheduler_cfg["minimum_learning_rate"],
            )
            if scheduler_cfg["enabled"]
            else None
        )
        best_score, history, start_epoch = -float("inf"), [], 0
        patience_reference, epochs_without_improvement = -float("inf"), 0
        if (
            not overwrite
            and latest.exists()
            and (
                not completion.exists()
                or int(json.loads(completion.read_text()).get("epochs", 0))
                < cfg["training"]["epochs"]
            )
        ):
            if not cfg['training']['recovery']['resume']:
                raise ValueError(f'Incomplete training for seed {seed}; enable training.recovery.resume or use --overwrite')
            state = torch.load(latest, map_location=device, weights_only=True)
            if not checkpoint_compatible(state, cfg):
                raise ValueError(f"{latest} belongs to a different configuration")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            if state.get("scaler") is not None:
                scaler.load_state_dict(state["scaler"])
            start_epoch = int(state["epoch"])
            best_score = float(state["best_score"])
            history = state["history"]
            patience_reference, epochs_without_improvement = patience_progress(
                history, cfg["training"]["early_stopping_min_delta"]
            )
            if scheduler is not None:
                if state.get("scheduler") is not None:
                    scheduler.load_state_dict(state["scheduler"])
                else:
                    # Backward-compatible continuation from checkpoints created
                    # before scheduler state was recorded.
                    for historical_row in history:
                        scheduler.step(float(historical_row["selection_score"]))
            rng.setstate(state["python_rng_state"])
            torch.set_rng_state(state["torch_rng_state"].cpu())
            if device.type == "cuda" and state.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state(state["cuda_rng_state"].cpu(), device)
            print(
                f"[{utc_now()}] RESUMING seed={seed} after epoch={start_epoch} from {latest}",
                flush=True,
            )
        elif not overwrite and path.exists() and not completion.exists():
            raise ValueError(f'Incomplete training for seed {seed} has no recovery checkpoint; use --overwrite')
        started = time.perf_counter()
        for epoch in range(start_epoch, cfg['training']['epochs']):
            wait_for_resources(cfg, device, f'before seed={seed} epoch={epoch+1}')
            model.train()
            order = repeated_order(training, factors, rng)
            rng.shuffle(order)
            train_losses = []
            batches = training_plan(
                order, cfg, rng, runtime_batches["training"], histories
            )
            if not batches:
                raise ValueError("Training split contains no samples")
            workers = cfg["training"]["preprocessing_workers"]
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for plans, examples in prefetched_batches(
                    executor, batches, submit_training_batch, cfg
                ):
                    mode = plans[0][1]
                    prepared_batch = examples[0]
                    train_losses.extend(
                        train_examples(
                            model,
                            optimizer,
                            scaler,
                            prepared_batch,
                            mode,
                            cfg,
                            device,
                            f"seed={seed} epoch={epoch+1}",
                            runtime_batches,
                            cfg["training"]["recovery"]["cuda_oom_retries"],
                        )
                    )
            model.eval()
            val, val_map, val_recall, validation_details = validation_metrics(
                model, validation, cfg, device, runtime_batches
            )
            if not all(np.isfinite(v) for v in (val, val_map, val_recall)):
                raise ValueError('Nonfinite validation metric')
            score = selection_score(cfg, val, val_map, val_recall)
            row = {
                "timestamp_utc": utc_now(),
                "seed": seed,
                "epoch": epoch + 1,
                "train_loss": float(np.mean(train_losses)),
                "val_loss": val,
                "val_map": val_map,
                "val_recall": val_recall,
                "val_loss_quality": 1.0 / (1.0 + val),
                "selection_score": score,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "effective_batch_size": runtime_batches["training"],
                "elapsed_seconds": time.perf_counter() - started,
            } | validation_details
            history.append(row)
            path.parent.mkdir(parents=True, exist_ok=True)
            candidate = path.parent/f'epoch_{epoch+1:04d}.pt'
            temporary = candidate.with_suffix('.tmp')
            state = {'model':model.state_dict(),'signature':signature(cfg),'config':cfg,
                     'epoch':epoch+1,'seed':seed,'validation_loss':val,
                     'validation_map':val_map,'validation_recall':val_recall,
                     'selection_score':score}
            torch.save(state, temporary)
            temporary.replace(candidate)
            write_json(candidate.with_suffix('.json'), {k:state[k] for k in
                       ('epoch','seed','validation_loss','validation_map','validation_recall','selection_score')})
            retained = prune_checkpoints(path.parent, cfg['training']['checkpointing']['keep_best'])
            if score > best_score:
                best_score = score
                best_tmp = path.with_suffix('.tmp')
                shutil.copyfile(candidate, best_tmp)
                best_tmp.replace(path)
            # The patience reference moves only on gains above min_delta, so a
            # resumed run (rebuilt from history) stops at the same epoch as an
            # uninterrupted one.
            if score > patience_reference + cfg["training"]["early_stopping_min_delta"]:
                patience_reference, epochs_without_improvement = score, 0
            else:
                epochs_without_improvement += 1
            if scheduler is not None:
                scheduler.step(score)
            metrics_path = path.parent/f'training_metrics_seed_{seed}.csv'
            write_csv(metrics_path, history)
            if cfg["output"]["plots"]:
                plot_training(path.parent / f"training_curves_seed_{seed}.png", history)
            kept_epochs = ",".join(str(item[1]) for item in retained)
            primary_strategy = cfg["evaluation"]["strategies"][0]
            class_metrics = {}
            for metric in ("ap", "precision", "recall"):
                class_metrics[metric] = ",".join(
                    f"{label}:{row.get(f'val_{metric}_{primary_strategy}_{label}', 0) or 0:.3f}"
                    for label in cfg["dataset"]["classes"]
                )
            print(
                f'seed={seed} epoch={epoch+1} train={row["train_loss"]:.4f} '
                f"val={val:.4f} mAP={val_map:.4f} "
                f'(single={row["val_map_single"]:.4f} '
                f'cooperative={row["val_map_cooperative"]:.4f}) '
                f'precision={row["val_precision"]:.4f} '
                f'recall={val_recall:.4f} F1={row["val_f1"]:.4f} '
                f"score={score:.4f} {primary_strategy} IoU-level-0 "
                f"AP=[{class_metrics['ap']}] "
                f"P=[{class_metrics['precision']}] "
                f"R=[{class_metrics['recall']}] "
                f"kept=[{kept_epochs}]",
                flush=True,
            )
            should_stop = (
                cfg["training"]["early_stopping_patience"] > 0
                and epoch + 1 >= cfg["training"]["minimum_epochs"]
                and epochs_without_improvement
                >= cfg["training"]["early_stopping_patience"]
            )
            if (epoch + 1) % cfg["training"]["recovery"][
                "save_every_epochs"
            ] == 0 or should_stop:
                recovery_state = state | {
                    "optimizer": optimizer.state_dict(),
                    "scaler": (
                        scaler.state_dict() if gradient_scaling(cfg, device) else None
                    ),
                    "best_score": best_score,
                    "history": history,
                    "scheduler": (
                        scheduler.state_dict() if scheduler is not None else None
                    ),
                    "epochs_without_improvement": epochs_without_improvement,
                    "python_rng_state": rng.getstate(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": (
                        torch.cuda.get_rng_state(device)
                        if device.type == "cuda"
                        else None
                    ),
                }
                atomic_torch_save(recovery_state, latest)
            if should_stop:
                print(
                    f"Early stopping seed={seed} at epoch={epoch+1}; "
                    f"no selection-score improvement greater than "
                    f"{cfg['training']['early_stopping_min_delta']:g} for "
                    f"{epochs_without_improvement} epochs",
                    flush=True,
                )
                break
        best_state = torch.load(path, map_location='cpu', weights_only=True)
        completed_epochs = int(history[-1]["epoch"]) if history else start_epoch
        write_json(
            completion,
            {
                "timestamp_utc": utc_now(),
                "epochs": completed_epochs,
                "target_epochs": cfg["training"]["epochs"],
                "stopped_early": completed_epochs < cfg["training"]["epochs"],
                "best_epoch": best_state["epoch"],
                "best_validation_loss": best_state["validation_loss"],
                "best_validation_map": best_state["validation_map"],
                "best_validation_recall": best_state["validation_recall"],
                "best_selection_score": best_state["selection_score"],
            },
        )


def choose_groups(nodes, settings):
    nodes = sorted(nodes)
    rng = random.Random(settings['subset_seed'])
    groups = []
    for count in settings['node_counts']:
        if count > len(nodes):
            raise ValueError(f'Requested {count} nodes but only {len(nodes)} common nodes. Adjust evaluation.node_counts.')
        choices = list(itertools.combinations(nodes,count))
        if count != 1 and len(choices)>settings['subsets_per_count']:
            choices = sorted(rng.sample(choices,settings['subsets_per_count']))
        if settings.get("receiver_policy", "all") == "all" and count > 1:
            choices = [
                (receiver,) + tuple(node for node in choice if node != receiver)
                for choice in choices
                for receiver in choice
            ]
        groups.extend(choices)
    return groups


def payload(array):
    return len(zlib.compress(np.asarray(array,dtype='<f4').tobytes(),level=1))


def feature_payload(array, cfg):
    """Compressed dense feature bytes at the adaptive value precision."""
    dtype = "<f2" if cfg["adaptive_fusion"]["value_bytes"] == 2 else "<f4"
    return len(zlib.compress(np.asarray(array, dtype=dtype).tobytes(), level=1))


def adaptive_payload(messages, selected, cfg):
    """Serialize each collaborator's selected cells: indices and the cell values.

    ``messages`` [N, C, h, w] holds every node's transmitted channels (node 0
    is the receiver and sends nothing); ``selected`` [N-1, h, w] marks the
    cells each collaborator sends.
    """
    total = 0
    index_dtype = "<u2" if cfg["adaptive_fusion"]["index_bytes"] == 2 else "<u4"
    value_dtype = "<f2" if cfg["adaptive_fusion"]["value_bytes"] == 2 else "<f4"
    for node, mask in zip(messages[1:], selected):
        device_indices = mask.flatten().nonzero(as_tuple=False).flatten()
        indices = device_indices.detach().cpu().numpy().astype(index_dtype, copy=False)
        values = (
            node.flatten(1)[:, device_indices]
            .T.detach()
            .float()
            .cpu()
            .numpy()
            .astype(value_dtype, copy=False)
        )
        total += len(zlib.compress(indices.tobytes() + values.tobytes(), level=1))
    return total


def late_payload(boxes, cfg):
    """Compressed bytes of the local boxes a node sends at the operating point.

    Lower-scoring candidates are kept locally for the AP ranking but would not
    be transmitted by a deployed node, so they do not count toward payload.
    """
    threshold = cfg["evaluation"]["score_threshold"]
    return payload(
        np.array(
            [
                list(box.to_array())
                + [cfg["dataset"]["classes"].index(box.label), box.score]
                for box in boxes
                if box.score >= threshold
            ],
            dtype=np.float32,
        ).reshape(-1, 9)
    )


def adaptive_selector(strategy):
    base = strategy.split("@", 1)[0]
    return base.split("_", 1)[1] if "_" in base else "learned"


def stable_random_seed(*parts):
    digest = hashlib.sha256("\0".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def run_inference(
    model,
    cloud_batches,
    cfg,
    device,
    strategy,
    raw_payloads,
    random_seeds,
    prepared_bevs,
    preprocessing_overheads_ms,
):
    """One inference attempt; infer_batch owns CUDA OOM recovery."""
    base_strategy, _, budget_text = strategy.partition("@")
    preprocessing_start = time.perf_counter()
    arrays = (
        prepared_bevs
        if prepared_bevs is not None
        else [array_inputs(clouds, cfg, strategy) for clouds in cloud_batches]
    )
    if base_strategy == "early":
        arrays = [value[0] for value in arrays]
    bev = device_tensor(np.stack(arrays), device)
    sync(device)
    preprocessing_ms = (
        (time.perf_counter() - preprocessing_start) * 1000 / len(cloud_batches)
    )
    if preprocessing_overheads_ms is not None:
        preprocessing_ms += float(np.mean(preprocessing_overheads_ms))
    start = time.perf_counter()
    serialization_ms = 0.0
    with torch.no_grad(), autocast(cfg, device):
        if base_strategy == "early":
            outputs = model(bev, "early")
            predictions = [
                decode(batch_outputs(outputs, i), cfg)
                for i in range(len(cloud_batches))
            ]
            representations = (
                raw_payloads
                if raw_payloads is not None
                else [sum(payload(p) for p in clouds[1:]) for clouds in cloud_batches]
            )
        else:
            batch, nodes = bev.shape[:2]
            features, density = model.encode(bev)
            # A collaborator's message carries its features and point density.
            messages = torch.cat((features, density.to(features.dtype)), dim=2)
            if base_strategy == "intermediate":
                outputs = model.detect(model.fuse(features, density))
                predictions = [
                    decode(batch_outputs(outputs, i), cfg) for i in range(batch)
                ]
                serialization_start = time.perf_counter()
                representations = [
                    sum(
                        feature_payload(message, cfg)
                        for message in item[1:].detach().float().cpu().numpy()
                    )
                    for item in messages
                ]
                serialization_ms = (
                    (time.perf_counter() - serialization_start)
                    * 1000
                    / len(cloud_batches)
                )
            elif base_strategy.startswith("adaptive"):
                budget = float(
                    budget_text or cfg["adaptive_fusion"]["evaluation_budgets"][0]
                )
                selector = adaptive_selector(strategy)
                generators = None
                if selector == "random" and random_seeds is not None:
                    generators = []
                    for random_seed in random_seeds:
                        generator = torch.Generator(device=device)
                        generator.manual_seed(random_seed)
                        generators.append(generator)
                available, _ = model.select(
                    features, density, budget, selector, generators=generators
                )
                outputs = model.detect(model.fuse(features, density, available))
                predictions = [
                    decode(batch_outputs(outputs, i), cfg) for i in range(batch)
                ]
                serialization_start = time.perf_counter()
                representations = [
                    adaptive_payload(messages[item], available[item, 1:], cfg)
                    for item in range(batch)
                ]
                serialization_ms = (
                    (time.perf_counter() - serialization_start)
                    * 1000
                    / len(cloud_batches)
                )
            else:
                outputs = model.detect(
                    model.fuse(
                        features.flatten(0, 1)[:, None], density.flatten(0, 1)[:, None]
                    )
                )
                predictions, representations = [], []
                serialization_seconds = 0.0
                for item in range(batch):
                    local = [
                        decode(batch_outputs(outputs, item * nodes + node), cfg)
                        for node in range(nodes)
                    ]
                    predictions.append(
                        nms(
                            [box for group in local for box in group],
                            cfg["evaluation"]["nms_iou"],
                            cfg["evaluation"]["late_max_detections"],
                        )
                    )
                    serialization_start = time.perf_counter()
                    representations.append(
                        sum(late_payload(group, cfg) for group in local[1:])
                    )
                    serialization_seconds += time.perf_counter() - serialization_start
                serialization_ms = serialization_seconds * 1000 / len(cloud_batches)
    sync(device)
    combined_ms = (time.perf_counter() - start) * 1000 / len(cloud_batches)
    inference_ms = max(0.0, combined_ms - serialization_ms)
    return list(
        zip(
            predictions,
            [inference_ms] * len(predictions),
            representations,
            [preprocessing_ms] * len(predictions),
            [serialization_ms] * len(predictions),
        )
    )


def infer_batch(
    model,
    cloud_batches,
    cfg,
    device,
    strategy,
    raw_payloads=None,
    retries=None,
    random_seeds=None,
    prepared_bevs=None,
    preprocessing_overheads_ms=None,
):
    """Batched inference with transparent CUDA OOM batch reduction."""
    if retries is None:
        retries = cfg["training"]["recovery"]["cuda_oom_retries"]
    try:
        return run_inference(
            model,
            cloud_batches,
            cfg,
            device,
            strategy,
            raw_payloads,
            random_seeds,
            prepared_bevs,
            preprocessing_overheads_ms,
        )
    except (torch.OutOfMemoryError, RuntimeError) as error:
        if not cuda_oom(error, device):
            raise
        oom_message = str(error)
    gc.collect()
    release_cuda_memory(device)
    if len(cloud_batches) == 1:
        if retries > 0:
            wait_for_resources(
                cfg, device, f"single-frame {strategy} evaluation CUDA OOM"
            )
            return infer_batch(
                model,
                cloud_batches,
                cfg,
                device,
                strategy,
                raw_payloads,
                retries - 1,
                random_seeds,
                prepared_bevs,
                preprocessing_overheads_ms,
            )
        raise ValueError(
            f"CUDA cannot fit one {strategy} evaluation frame after resource "
            "retries; reduce resolution, hidden_channels, or node_counts, or "
            f"use training.device=cpu. Original error: {oom_message}"
        )
    midpoint = len(cloud_batches) // 2
    print(
        f"[{utc_now()}] CUDA OOM during {strategy} evaluation; reducing batch "
        f"{len(cloud_batches)} -> {midpoint}+{len(cloud_batches)-midpoint}",
        flush=True,
    )
    left_payloads = raw_payloads[:midpoint] if raw_payloads is not None else None
    right_payloads = raw_payloads[midpoint:] if raw_payloads is not None else None
    left_seeds = random_seeds[:midpoint] if random_seeds is not None else None
    right_seeds = random_seeds[midpoint:] if random_seeds is not None else None
    left_bevs = prepared_bevs[:midpoint] if prepared_bevs is not None else None
    right_bevs = prepared_bevs[midpoint:] if prepared_bevs is not None else None
    left_overheads = (
        preprocessing_overheads_ms[:midpoint]
        if preprocessing_overheads_ms is not None
        else None
    )
    right_overheads = (
        preprocessing_overheads_ms[midpoint:]
        if preprocessing_overheads_ms is not None
        else None
    )
    return infer_batch(
        model,
        cloud_batches[:midpoint],
        cfg,
        device,
        strategy,
        left_payloads,
        retries,
        left_seeds,
        left_bevs,
        left_overheads,
    ) + infer_batch(
        model,
        cloud_batches[midpoint:],
        cfg,
        device,
        strategy,
        right_payloads,
        retries,
        right_seeds,
        right_bevs,
        right_overheads,
    )


def network_rows(cfg, base, cloud_bytes, representation_bytes):
    """Analytical latency rows.

    Edge processing runs at the receiving node, so only collaborators send
    ``representation_bytes``. Cloud processing needs every participating
    node, the receiver included, to upload its raw points (``cloud_bytes``).
    """
    net = cfg["deployment_assumptions"]
    frame_period_ms = cfg["latency_experiment"]["frame_period_ms"]
    base_strategy = base["strategy"].split("@", 1)[0]
    if base_strategy.startswith("adaptive"):
        base_strategy = "adaptive"
    measured_keys = ("preprocessing_ms_mean", "inference_decode_ms_mean")
    rows = []
    for placement in ("edge", "cloud"):
        transmitted = representation_bytes if placement == "edge" else cloud_bytes
        serialization_key = (
            "serialization_ms_mean"
            if placement == "edge"
            else "cloud_serialization_ms_mean"
        )
        measured = (
            sum(base[key] for key in (*measured_keys, serialization_key))
            if all(key in base for key in (*measured_keys, serialization_key))
            else None
        )
        for bandwidth in net["bandwidths_mbps"]:
            transmit_ms = transmitted * 8 / (bandwidth * 1e6) * 1000
            fixed = net["sensing_ms"] + net[placement + "_rtt_ms"]
            total = fixed + net["processing_ms"][placement][base_strategy] + transmit_ms
            rows.append(
                base
                | {
                    "placement": placement,
                    "bandwidth_mbps": bandwidth,
                    "assumption_source": net["source"],
                    "payload_bytes": transmitted,
                    # Link rate needed to deliver every frame within one period.
                    "required_bandwidth_mbps": transmitted * 8 / frame_period_ms / 1000,
                    "transmission_ms_model": transmit_ms,
                    "end_to_end_ms_model": total,
                    # Replaces the assumed processing constant with the measured
                    # preprocessing, inference+decode, and serialization times.
                    "end_to_end_ms_measured": (
                        None if measured is None else fixed + measured + transmit_ms
                    ),
                    "link_keeps_up": transmit_ms < frame_period_ms,
                }
            )
    return rows


def finite_burst_arrivals(frame_count, frame_period_ms, burst_multiplier):
    """Finite bursts with recovery gaps and the original long-run frame rate."""
    if burst_multiplier == 1:
        return [index * frame_period_ms for index in range(frame_count)]
    return [
        (index // burst_multiplier) * burst_multiplier * frame_period_ms
        + (index % burst_multiplier) * frame_period_ms / burst_multiplier
        for index in range(frame_count)
    ]


def simulate_fifo_latency(
    service_ms, arrival_interval_ms, fixed_ms, deadline_ms, arrival_times_ms=None
):
    """Return D/D/1-style FIFO latency samples for a measured service trace."""
    available = 0.0
    latencies, waits = [], []
    arrivals = (
        arrival_times_ms
        if arrival_times_ms is not None
        else [index * arrival_interval_ms for index in range(len(service_ms))]
    )
    for arrival, service in zip(arrivals, service_ms):
        start = max(arrival, available)
        wait = start - arrival
        available = start + service
        waits.append(wait)
        latencies.append(fixed_ms + wait + service)
    values = np.asarray(latencies, dtype=float)
    return {
        "frames": len(values),
        "latency_mean_ms": float(values.mean()),
        "latency_p50_ms": float(np.quantile(values, 0.50)),
        "latency_p95_ms": float(np.quantile(values, 0.95)),
        "latency_p99_ms": float(np.quantile(values, 0.99)),
        "latency_max_ms": float(values.max()),
        "queue_wait_mean_ms": float(np.mean(waits)),
        "queue_wait_p99_ms": float(np.quantile(waits, 0.99)),
        "deadline_miss_rate": float(np.mean(values > deadline_ms)),
    }


TRACE_KEYS = ("seed", "session", "weather", "nodes", "node_count", "strategy")
TRACE_FIELDS = (
    *TRACE_KEYS[:3],
    "sample",
    *TRACE_KEYS[3:],
    "load_ms",
    "preprocessing_ms",
    "inference_decode_ms",
    "serialization_ms",
    "cloud_serialization_ms",
    "local_pipeline_ms",
    "cloud_raw_bytes",
    "representation_bytes",
)


def latency_tail_rows(cfg, grouped):
    """FIFO tail-latency summaries for measured per-frame service traces.

    ``grouped`` maps a ``TRACE_KEYS`` tuple to per-frame tuples of
    ``(inference_decode_ms, serialization_ms, cloud_serialization_ms,
    representation_bytes, cloud_raw_bytes)``.
    """
    summaries = []
    net, experiment = cfg["deployment_assumptions"], cfg["latency_experiment"]
    for key, rows in sorted(grouped.items()):
        for placement in ("edge", "cloud"):
            for bandwidth in net["bandwidths_mbps"]:
                service = []
                for compute, serialization, cloud_ms, sent, uploaded in rows:
                    payload_bytes = sent if placement == "edge" else uploaded
                    tx = payload_bytes * 8 / (bandwidth * 1e6) * 1000
                    # Replay measured receiver compute rather than a configured
                    # constant. Loading and sender preprocessing remain outside
                    # the shared receiver queue.
                    service.append(
                        compute
                        + (serialization if placement == "edge" else cloud_ms)
                        + tx
                    )
                for burst in experiment["burst_multipliers"]:
                    arrivals = finite_burst_arrivals(
                        len(service), experiment["frame_period_ms"], burst
                    )
                    for deadline in experiment["deadlines_ms"]:
                        result = simulate_fifo_latency(
                            service,
                            experiment["frame_period_ms"],
                            net["sensing_ms"] + net[placement + "_rtt_ms"],
                            deadline,
                            arrival_times_ms=arrivals,
                        )
                        summaries.append(
                            dict(zip(TRACE_KEYS, key))
                            | {
                                "placement": placement,
                                "bandwidth_mbps": bandwidth,
                                "burst_multiplier": burst,
                                "arrival_interval_ms": experiment["frame_period_ms"],
                                "within_burst_interval_ms": (
                                    experiment["frame_period_ms"] / burst
                                ),
                                "utilization": float(np.mean(service))
                                / experiment["frame_period_ms"],
                                # At utilization >= 1 the queue grows without
                                # bound; tail values then reflect trace length.
                                "queue_stable": float(np.mean(service))
                                < experiment["frame_period_ms"],
                                "deadline_ms": deadline,
                                **result,
                            }
                        )
    return summaries


def concatenate_csv(parts, destination):
    """Stream CSV parts into one file without loading them all into memory."""
    fieldnames = []
    for part in parts:
        with part.open(newline="") as stream:
            for name in next(csv.reader(stream), []):
                if name not in fieldnames:
                    fieldnames.append(name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
        for part in parts:
            with part.open(newline="") as stream:
                writer.writerows(csv.DictReader(stream))
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(destination)


def evaluation_strategies(cfg):
    strategies = []
    for configured in cfg["evaluation"]["strategies"]:
        if configured == "adaptive":
            budgets = cfg["adaptive_fusion"]["evaluation_budgets"]
            strategies.extend(f"adaptive@{budget:g}" for budget in budgets)
            strategies.extend(
                f"adaptive_{selector}@{budget:g}"
                for selector in cfg["adaptive_fusion"]["ablations"]
                for budget in budgets
            )
        else:
            strategies.append(configured)
    return strategies


def evaluation_runs(nodes, strategies):
    """Computations for one node group, each with the strategy labels it serves.

    Receiver rotation matters only to asymmetric adaptive fusion, so other
    strategies run once per unordered subset. With one node every strategy
    reduces to the same single-node detector, so it runs once and is reported
    under every strategy label.
    """
    labels = [
        strategy
        for strategy in strategies
        if strategy.startswith("adaptive") or tuple(nodes) == tuple(sorted(nodes))
    ]
    if len(nodes) == 1:
        return [("early", labels)]
    return [(label, [label]) for label in labels]


DETECTION_FIELDS = (
    *TRACE_KEYS[:3],
    "sample",
    *TRACE_KEYS[3:],
    "record",
    "class",
    "status",
    "score",
    "iou",
    "x",
    "y",
    "z",
    "length",
    "width",
    "height",
    "yaw",
    "support_points",
    "nearest_node_m",
)


def nearest_node_distance(box, origins):
    if not origins or any(origin is None for origin in origins):
        return None
    return float(min(math.hypot(box.x - o[0], box.y - o[1]) for o in origins))


def detection_rows(records, base, entries):
    """Rows for the per-detection dump from ``evaluate(records=...)`` output."""
    for record in records:
        _, gt, counts, sample_id, origins = entries[record["frame"]]
        box = record["box"]
        support = None
        if record["record"] == "ground_truth":
            support = next(
                (int(count) for truth, count in zip(gt, counts) if truth is box), None
            )
        yield base | {
            "sample": sample_id,
            "record": record["record"],
            "class": record["class"],
            "status": record["status"],
            "score": record["score"],
            "iou": record["iou"],
            "x": box.x,
            "y": box.y,
            "z": box.z,
            "length": box.length,
            "width": box.width,
            "height": box.height,
            "yaw": box.yaw,
            "support_points": support,
            "nearest_node_m": nearest_node_distance(box, origins),
        }


def evaluate_session(
    model,
    cfg,
    device,
    seed,
    session,
    frames,
    warmed,
    trace_writer,
    score_thresholds=None,
    detection_writer=None,
):
    """Evaluate one test session; stream traces and return metrics, costs, tails.

    Every subset is scored on two ground-truth scopes: ``full`` (all boxes in
    the fixed region, the system view) and ``visible`` (boxes with at least
    ``evaluation.min_visible_points`` points from the subset's nodes; the rest
    are ignored), which separates fusion quality from added coverage.
    """
    strategies = evaluation_strategies(cfg)
    bounds = cfg["dataset"]["bounds"]
    if score_thresholds is None:
        score_thresholds = cfg["evaluation"]["score_threshold"]
    common = set.intersection(*(set(f["nodes"]) for f in frames))
    groups = choose_groups(common, cfg["evaluation"])
    buckets, costs, service = defaultdict(list), defaultdict(list), defaultdict(list)
    batch_size = cfg["latency_experiment"]["profile_batch_size"]
    for start in range(0, len(frames), batch_size):
        frame_chunk = frames[start : start + batch_size]
        load_start = time.perf_counter()
        loaded = [
            (
                annotations(sample["gt"], cfg["dataset"]),
                {n: load_cloud(sample["nodes"][n], bounds) for n in common},
            )
            for sample in frame_chunk
        ]
        load_ms = (time.perf_counter() - load_start) * 1000 / len(frame_chunk)
        prepared = []
        for sample, (gt, cloud_by_node) in zip(frame_chunk, loaded):
            raw, raw_ms, bev, bev_ms = {}, {}, {}, {}
            for node, cloud in cloud_by_node.items():
                started = time.perf_counter()
                raw[node] = payload(cloud)
                raw_ms[node] = (time.perf_counter() - started) * 1000
                started = time.perf_counter()
                bev[node] = rasterize(cloud, cfg["dataset"])
                bev_ms[node] = (time.perf_counter() - started) * 1000
            # Untimed: point support of every box per node, for the visible scope.
            support = {
                node: box_point_counts(cloud, gt, bounds)
                for node, cloud in cloud_by_node.items()
            }
            origins = {
                node: node_origin(sample["nodes"][node]) for node in cloud_by_node
            }
            prepared.append(
                (gt, cloud_by_node, raw, raw_ms, bev, bev_ms, support, origins)
            )
        for nodes in groups:
            cloud_batches = [[item[1][n] for n in nodes] for item in prepared]
            # Edge early fusion: collaborators send raw points to the receiver.
            collaborator_raw = [sum(item[2][n] for n in nodes[1:]) for item in prepared]
            # Cloud placement: every participating node uploads raw points.
            cloud_raw = [sum(item[2][n] for n in nodes) for item in prepared]
            cloud_ms = [sum(item[3][n] for n in nodes) for item in prepared]
            for strategy, labels in evaluation_runs(nodes, strategies):
                random_seeds = [
                    stable_random_seed(seed, sample["id"], *nodes, strategy)
                    for sample in frame_chunk
                ]
                prepared_bevs = preprocessing_overheads = None
                if strategy != "early" or len(nodes) == 1:
                    prepared_bevs = [
                        np.stack([item[4][node] for node in nodes]) for item in prepared
                    ]
                    preprocessing_overheads = [
                        sum(item[5][node] for node in nodes) for item in prepared
                    ]
                warmup_key = (strategy, len(nodes), len(cloud_batches))
                if warmup_key not in warmed:
                    infer_batch(
                        model,
                        cloud_batches[:1],
                        cfg,
                        device,
                        strategy,
                        collaborator_raw[:1],
                        random_seeds=random_seeds[:1],
                        prepared_bevs=(
                            prepared_bevs[:1] if prepared_bevs is not None else None
                        ),
                        preprocessing_overheads_ms=(
                            preprocessing_overheads[:1]
                            if preprocessing_overheads is not None
                            else None
                        ),
                    )
                    warmed.add(warmup_key)
                results = infer_batch(
                    model,
                    cloud_batches,
                    cfg,
                    device,
                    strategy,
                    collaborator_raw,
                    random_seeds=random_seeds,
                    prepared_bevs=prepared_bevs,
                    preprocessing_overheads_ms=preprocessing_overheads,
                )
                for sample, item, uploaded, upload_ms, result in zip(
                    frame_chunk, prepared, cloud_raw, cloud_ms, results
                ):
                    preds, elapsed, representation, preprocessing, serialization = (
                        result
                    )
                    group_support = sum(item[6][n] for n in nodes)
                    group_origins = [item[7][n] for n in nodes]
                    for label in labels:
                        buckets[(nodes, label)].append(
                            (preds, item[0], group_support, sample["id"], group_origins)
                        )
                        costs[(nodes, label)].append(
                            (
                                elapsed,
                                uploaded,
                                representation,
                                preprocessing,
                                serialization,
                                upload_ms,
                            )
                        )
                        key = (
                            seed,
                            session,
                            sample["weather"],
                            ",".join(nodes),
                            len(nodes),
                            label,
                        )
                        service[key].append(
                            (
                                elapsed,
                                serialization,
                                upload_ms,
                                representation,
                                uploaded,
                            )
                        )
                        trace_writer.writerow(
                            dict(zip(TRACE_KEYS, key))
                            | {
                                "sample": sample["id"],
                                "load_ms": load_ms,
                                "preprocessing_ms": preprocessing,
                                "inference_decode_ms": elapsed,
                                "serialization_ms": serialization,
                                "cloud_serialization_ms": upload_ms,
                                "local_pipeline_ms": load_ms
                                + preprocessing
                                + elapsed
                                + serialization,
                                "cloud_raw_bytes": uploaded,
                                "representation_bytes": representation,
                            }
                        )
    metrics, cost_rows, comparison_rows = [], [], []
    comparison = cfg["evaluation"]["comparison"]
    for (nodes, strategy), values in buckets.items():
        base = {
            "seed": seed,
            "session": session,
            "weather": frames[0]["weather"],
            "nodes": ",".join(nodes),
            "node_count": len(nodes),
            "strategy": strategy,
        }
        records = [] if detection_writer is not None else None
        scoped = {
            "full": [(preds, gt) for preds, gt, *_ in values],
            "visible": [
                (preds, *visibility_scope(gt, support, cfg))
                for preds, gt, support, *_ in values
            ],
        }
        for scope in GT_SCOPES:
            metrics.extend(
                base | {"gt_scope": scope} | row
                for row in evaluate(
                    scoped[scope],
                    cfg["dataset"]["classes"],
                    cfg["evaluation"]["iou_thresholds"],
                    score_thresholds,
                    cfg["evaluation"]["class_iou_thresholds"],
                    records=records if scope == "full" else None,
                )
            )
        if records:
            detection_writer.writerows(detection_rows(records, base, values))
        if comparison is not None:
            # The published benchmark's metric (merged classes, BEV IoU), so
            # results can be set beside its table.
            for scope in GT_SCOPES:
                grouped = group_frames(
                    scoped[scope],
                    comparison["groups"],
                    cfg["evaluation"]["nms_iou"],
                    cfg["evaluation"]["late_max_detections"],
                )
                comparison_rows.extend(
                    base | {"gt_scope": scope, "iou_type": comparison["iou_type"]} | row
                    for row in evaluate(
                        grouped,
                        list(comparison["groups"]),
                        [comparison["iou_threshold"]],
                        iou_type=comparison["iou_type"],
                    )
                )
        measured = np.array(costs[(nodes, strategy)])
        cost_base = base | {
            "frames": len(values),
            "inference_decode_ms_mean": float(measured[:, 0].mean()),
            "inference_decode_ms_p95": float(np.quantile(measured[:, 0], 0.95)),
            "preprocessing_ms_mean": float(measured[:, 3].mean()),
            "serialization_ms_mean": float(measured[:, 4].mean()),
            "cloud_serialization_ms_mean": float(measured[:, 5].mean()),
        }
        cost_rows.extend(
            network_rows(
                cfg,
                cost_base,
                float(measured[:, 1].mean()),
                float(measured[:, 2].mean()),
            )
        )
    print(
        f"Evaluated seed={seed} session={session}, frames={len(frames)}, "
        f"subsets={len(groups)}",
        flush=True,
    )
    return metrics, cost_rows, latency_tail_rows(cfg, service), comparison_rows


def operating_thresholds(model, cfg, device, validation, seed_directory, signature):
    """Per-class operating score thresholds, tuned on validation when enabled.

    Thresholds maximize F1 at the primary IoU level over every validation view
    (visible scope). Classes absent from validation keep
    ``evaluation.score_threshold``. Results are cached per evaluation signature.
    """
    settings = cfg["evaluation"]
    if not settings["tune_score_thresholds"]:
        return settings["score_threshold"]
    path = seed_directory / "operating_thresholds.json"
    if path.exists():
        try:
            cached = json.loads(path.read_text())
        except (OSError, ValueError):
            cached = {}
        if cached.get("signature") == signature:
            return cached["thresholds"]
    _, frames = validation_frames(
        model, validation, cfg, device, settings["batch_size"]
    )
    pooled = [frame for view in frames.values() for frame in view]
    thresholds = f1_optimal_thresholds(
        pooled,
        cfg["dataset"]["classes"],
        settings["iou_thresholds"],
        settings["class_iou_thresholds"],
        fallback=settings["score_threshold"],
        minimum=settings["ap_score_threshold"],
    )
    write_json(
        path,
        {
            "signature": signature,
            "criterion": "maximum F1 at the primary IoU level on validation (visible scope)",
            "fallback": settings["score_threshold"],
            "thresholds": thresholds,
        },
    )
    print(f"Operating thresholds for {seed_directory.name}: {thresholds}", flush=True)
    return thresholds


def run_evaluation(cfg):
    indexed_samples = indexed(cfg)
    samples = [s for s in indexed_samples if s["split"] == "test"]
    validation = [s for s in indexed_samples if s["split"] == "val"]
    save_detections = cfg["evaluation"]["save_detections"]
    sessions = defaultdict(list)
    for sample in samples:
        sessions[sample['session']].append(sample)
    output, device = Path(cfg['output']['root']),device_for(cfg)
    all_metrics, all_costs, all_comparison, trace_parts, tail_parts = [], [], [], [], []
    for seed in cfg['training']['seeds']:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        checkpoint = output/f'seed_{seed}'/'best.pt'
        if not checkpoint.exists():
            raise FileNotFoundError(f'Train first; missing {checkpoint}')
        state = torch.load(checkpoint,map_location=device,weights_only=True)
        if not checkpoint_compatible(state, cfg):
            raise ValueError("Checkpoint data/model configuration mismatch")
        evaluation_signature = evaluation_fingerprint(cfg, checkpoint)
        model = make_model(cfg, device)
        model.load_state_dict(state["model"])
        model.eval()
        thresholds = operating_thresholds(
            model, cfg, device, validation, checkpoint.parent, evaluation_signature
        )
        warmed = set()
        for session, frames in sessions.items():
            part_directory = output / "eval_parts" / f"seed_{seed}"
            part_path = part_directory / f"{session}.json"
            trace_path = part_directory / f"{session}.traces.csv"
            tail_path = part_directory / f"{session}.latency_tail.csv"
            detection_path = part_directory / f"{session}.detections.csv.gz"
            if (
                part_path.exists()
                and trace_path.exists()
                and tail_path.exists()
                and (detection_path.exists() or not save_detections)
            ):
                try:
                    part = json.loads(part_path.read_text())
                except (OSError, ValueError) as error:
                    print(
                        f"Ignoring unreadable evaluation part {part_path}: {error}",
                        flush=True,
                    )
                else:
                    if part.get("signature") == evaluation_signature:
                        all_metrics.extend(part["metrics"])
                        all_costs.extend(part["costs"])
                        all_comparison.extend(part["comparison"])
                        trace_parts.append(trace_path)
                        tail_parts.append(tail_path)
                        print(
                            f"Resumed evaluation seed={seed} session={session}",
                            flush=True,
                        )
                        continue
            # Traces are streamed to disk: a full sweep produces millions of rows.
            part_directory.mkdir(parents=True, exist_ok=True)
            temporary = trace_path.with_name(f".{trace_path.name}.{os.getpid()}.tmp")
            detection_temporary = detection_path.with_name(
                f".{detection_path.name}.{os.getpid()}.tmp"
            )
            with temporary.open("w", newline="") as stream, (
                gzip.open(detection_temporary, "wt", newline="", compresslevel=6)
                if save_detections
                else open(os.devnull, "w")
            ) as detection_stream:
                writer = csv.DictWriter(stream, fieldnames=TRACE_FIELDS)
                writer.writeheader()
                detection_writer = None
                if save_detections:
                    detection_writer = csv.DictWriter(
                        detection_stream, fieldnames=DETECTION_FIELDS
                    )
                    detection_writer.writeheader()
                metrics, costs, tails, comparison = evaluate_session(
                    model,
                    cfg,
                    device,
                    seed,
                    session,
                    frames,
                    warmed,
                    writer,
                    thresholds,
                    detection_writer,
                )
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(trace_path)
            if save_detections:
                detection_temporary.replace(detection_path)
            write_csv(tail_path, tails)
            # The JSON part is written last; it marks the session as complete.
            write_json(
                part_path,
                {
                    "signature": evaluation_signature,
                    "seed": seed,
                    "session": session,
                    "metrics": metrics,
                    "costs": costs,
                    "comparison": comparison,
                },
            )
            all_metrics.extend(metrics)
            all_costs.extend(costs)
            all_comparison.extend(comparison)
            trace_parts.append(trace_path)
            tail_parts.append(tail_path)
    write_csv(output / "accuracy.csv", all_metrics)
    if cfg["evaluation"]["comparison"] is not None:
        write_csv(output / "coinfra_comparison.csv", all_comparison)
    write_csv(output / "latency_communication.csv", all_costs)
    concatenate_csv(trace_parts, output / "latency_trace.csv")
    concatenate_csv(tail_parts, output / "latency_tail_summary.csv")
    return all_metrics


def deadline_sequences(cfg):
    """Consecutive native-rate frames from the first slices of each test session."""
    x = cfg["deadline_experiment"]
    sequence_cfg = copy.deepcopy(cfg)
    d = sequence_cfg["dataset"]
    d["splits"] = {"train": [], "val": [], "test": list(d["splits"]["test"])}
    d["split_mode"] = "sessions"
    d["frame_stride"] = d["train_frame_stride"] = 1
    d["max_frames_per_slice"] = x["frames_per_slice"]
    slices = defaultdict(list)
    for sample in discover(sequence_cfg):
        slices[(sample["session"], sample["slice"])].append(sample)
    chosen, per_session = {}, defaultdict(int)
    for key in sorted(slices):
        if per_session[key[0]] < x["slices_per_session"]:
            per_session[key[0]] += 1
            chosen[key] = sorted(slices[key], key=lambda item: float(item["timestamp"]))
    return chosen


class DeadlineState:
    """Links, estimators, controllers, and delivered messages of one run."""

    def __init__(self, cfg, key, collaborators, trace, controller):
        x = cfg["deadline_experiment"]
        period = cfg["latency_experiment"]["frame_period_ms"]
        rtt = cfg["deployment_assumptions"]["edge_rtt_ms"]
        self.links, self.estimators, self.logs, self.controllers = {}, {}, {}, {}
        self.inbox = {node: [] for node in collaborators}
        for node in collaborators:
            seed = stable_random_seed("deadline-trace", trace["name"], *key, node)
            self.links[node] = Link(make_trace(trace, seed, period), rtt)
            self.estimators[node] = PastOnlyLinkEstimator(
                DEADLINE_INITIAL_MBPS, x["estimator_alpha"]
            )
            self.logs[node] = OutcomeLog()
            self.controllers[node] = controller() if controller else None
        # Per collaborator: (frame, budget, age in s) of the message fused
        # this frame, or None.
        self.use = {}


def deadline_frame_cache(model, cfg, device, sample, nodes, budgets):
    """Everything any controller may send or fuse for one frame."""
    d = cfg["dataset"]
    bounds = d["bounds"]
    clouds = [load_cloud(sample["nodes"][node], bounds) for node in nodes]
    gt = annotations(sample["gt"], d)
    support = sum(box_point_counts(cloud, gt, bounds) for cloud in clouds)
    bev = device_tensor(
        np.stack([rasterize(cloud, d) for cloud in clouds])[None], device
    )
    with torch.no_grad(), autocast(cfg, device):
        features, density = model.encode(bev)
        utility = model.utility(features[0, 1:]).float().sigmoid().flatten(1)
        ranking = utility.argsort(dim=1, descending=True)
        cumulative = utility.gather(1, ranking).cumsum(dim=1)
        own = model.detect(model.fuse(features[0][:, None], density[0][:, None]))
    cells = ranking.shape[1]
    share = [
        {
            budget: float(row[math.ceil(budget * cells) - 1] / row[-1])
            for budget in budgets
        }
        for row in cumulative
    ]
    boxes = [decode(batch_outputs(own, index), cfg) for index in range(len(nodes))]
    threshold = cfg["evaluation"]["score_threshold"]
    return {
        "timestamp": float(sample["timestamp"]),
        "features": features[0],
        "density": density[0],
        "ranking": ranking,
        "share": share,
        "clouds": clouds,
        "boxes": boxes,
        # Raw points as float32 x, y, z, intensity; boxes as nine float32.
        "early_bytes": [len(cloud) * 16 for cloud in clouds[1:]],
        "late_bytes": [
            max(16, 36 * sum(box.score >= threshold for box in local))
            for local in boxes[1:]
        ],
        "scope": visibility_scope(gt, support, cfg),
        "cells": cells,
    }


def run_deadline_sequence(model, cfg, device, seed, key, frames, predictions, totals):
    """Replay one recording slice under every trace, deadline, and controller.

    Appends each frame's ``(detections, visible, ignored)`` to
    ``predictions[(seed, trace, deadline, controller)]`` and adds message
    statistics to ``totals`` under the same key.
    """
    x = cfg["deadline_experiment"]
    n = cfg["deployment_assumptions"]
    period = cfg["latency_experiment"]["frame_period_ms"]
    budgets = cfg["adaptive_fusion"]["evaluation_budgets"]
    adaptive = cfg["adaptive_fusion"]
    common = sorted(set.intersection(*(set(frame["nodes"]) for frame in frames)))
    if len(common) < 2:
        return
    receiver = common[stable_random_seed("deadline-receiver", *key) % len(common)]
    nodes = [receiver] + [node for node in common if node != receiver]
    collaborators = nodes[1:]
    factories = feature_controllers(
        x["controllers"],
        budgets,
        x["tradeoffs"],
        x["target_miss_rate"],
        x["max_miss_probability"],
    )
    labels = list(factories) + [
        name for name in ("dense", "early", "late") if name in x["controllers"]
    ]
    kinds = {
        label: (
            label
            if label in ("early", "late")
            else "intermediate" if label == "dense" else "adaptive"
        )
        for label in labels
    }
    states = {
        (trace["name"], deadline, label): DeadlineState(
            cfg, key, collaborators, trace, factories.get(label)
        )
        for trace in x["traces"]
        for deadline in x["deadlines_ms"]
        for label in labels
    }
    cache = {}
    for t, sample in enumerate(frames):
        cache[t] = deadline_frame_cache(model, cfg, device, sample, nodes, budgets)
        for stale in [f for f in cache if t - f > x["max_age_frames"]]:
            del cache[stale]
        current = cache[t]
        # Every sent cell carries its features and its point density.
        channels = current["features"].shape[1] + 1
        feature_jobs = []
        for (trace_name, deadline, label), state in states.items():
            kind = kinds[label]
            frame_key = (seed, trace_name, deadline, label)
            stats = totals[frame_key]
            fusion_ms = t * period + deadline - n["processing_ms"]["edge"][kind]
            ready_ms = t * period + n["sensing_ms"]
            for index, node in enumerate(collaborators):
                link, estimator = state.links[node], state.estimators[node]
                controller = state.controllers[node]
                if t:
                    # One-frame-old capacity report from the node's 5G modem.
                    estimator.observe_rate(t - 1, link.trace.bandwidth(t - 1))
                for *_, missed in state.logs[node].release(ready_ms):
                    if controller is not None:
                        controller.outcome(True, missed)
                if controller is not None:
                    controller.frame()
                queue = link.queue_ms(ready_ms)
                horizon = fusion_ms - ready_ms
                window = horizon - link.rtt_ms / 2 - queue
                if kind == "adaptive":
                    options = [Option(0.0, 0, 0.0)] + [
                        Option(
                            budget,
                            sparse_bytes(
                                current["cells"],
                                budget,
                                channels,
                                adaptive["index_bytes"],
                                adaptive["value_bytes"],
                            ),
                            current["share"][index][budget],
                        )
                        for budget in budgets
                    ]
                    choice = controller.choose(
                        options,
                        {
                            "estimator": estimator,
                            "window_ms": window,
                            "horizon_ms": horizon,
                            "fixed_ms": link.rtt_ms / 2,
                            "queue_ms": queue,
                            "frame": t,
                            "cells": current["cells"],
                            "channels": channels,
                            "index_bytes": adaptive["index_bytes"],
                            "value_bytes": adaptive["value_bytes"],
                            "true_mbps": link.trace.bandwidth(t),
                        },
                    )
                    budget, size = choice.budget, choice.message_bytes
                elif kind == "intermediate":
                    budget = 1.0
                    size = dense_bytes(
                        current["cells"], channels, adaptive["value_bytes"]
                    )
                else:
                    budget = 1.0
                    size = current[f"{kind}_bytes"][index]
                stats["collaborator_frames"] += 1
                stats["budget"] += budget
                if budget > 0:
                    arrival, duration = link.send(t, ready_ms, size)
                    missed = arrival > fusion_ms
                    state.logs[node].record(
                        link.busy_until_ms, t, size, duration, missed
                    )
                    state.inbox[node].append((t, arrival, budget))
                    stats["sent"] += 1
                    stats["missed"] += missed
                    stats["bytes"] += size
                usable = [
                    message
                    for message in state.inbox[node]
                    if message[1] <= fusion_ms and t - message[0] <= x["max_age_frames"]
                ]
                state.inbox[node] = [
                    message
                    for message in state.inbox[node]
                    if t - message[0] <= x["max_age_frames"]
                ]
                latest = max(usable, default=None)
                if latest is None:
                    stats["none"] += 1
                    state.use[node] = None
                    continue
                age = current["timestamp"] - cache[latest[0]]["timestamp"]
                stats["fresh" if latest[0] == t else "stale"] += 1
                stats["age"] += age
                state.use[node] = (latest[0], latest[2], age)
            if kind in ("adaptive", "intermediate"):
                feature_jobs.append((frame_key, state))
            elif kind == "early":
                merged = [current["clouds"][0]]
                for index, node in enumerate(collaborators):
                    use = state.use[node]
                    if use is not None:
                        merged.append(cache[use[0]]["clouds"][index + 1])
                bev = device_tensor(
                    rasterize(np.concatenate(merged), cfg["dataset"])[None], device
                )
                with torch.no_grad(), autocast(cfg, device):
                    outputs = model(bev, "early")
                predictions[frame_key].append((decode(outputs, cfg), *current["scope"]))
            else:
                boxes = list(current["boxes"][0])
                for index, node in enumerate(collaborators):
                    use = state.use[node]
                    if use is not None:
                        boxes.extend(cache[use[0]]["boxes"][index + 1])
                merged_boxes = nms(
                    boxes,
                    cfg["evaluation"]["nms_iou"],
                    cfg["evaluation"]["late_max_detections"],
                )
                predictions[frame_key].append((merged_boxes, *current["scope"]))
        for start in range(0, len(feature_jobs), cfg["evaluation"]["batch_size"]):
            chunk = feature_jobs[start : start + cfg["evaluation"]["batch_size"]]
            stacked = [
                deadline_fusion_inputs(current, cache, collaborators, state)
                for _, state in chunk
            ]
            with torch.no_grad(), autocast(cfg, device):
                outputs = model.detect(
                    model.fuse(*(torch.stack(part) for part in zip(*stacked)))
                )
            for index, (frame_key, _) in enumerate(chunk):
                predictions[frame_key].append(
                    (decode(batch_outputs(outputs, index), cfg), *current["scope"])
                )


def deadline_fusion_inputs(current, cache, collaborators, state):
    """Receiver-side features, density, availability, and ages for one run."""
    features = [current["features"][0]]
    density = [current["density"][0]]
    receiver = torch.ones_like(current["density"][0][0], dtype=torch.bool)
    available, ages = [receiver], [0.0]
    for index, node in enumerate(collaborators):
        use = state.use[node]
        if use is None:
            features.append(torch.zeros_like(current["features"][0]))
            density.append(torch.zeros_like(current["density"][0]))
            available.append(torch.zeros_like(receiver))
            ages.append(0.0)
            continue
        frame, budget, age = use
        source = cache[frame]
        features.append(source["features"][index + 1])
        density.append(source["density"][index + 1])
        mask = torch.zeros(receiver.numel(), dtype=torch.bool, device=receiver.device)
        mask[source["ranking"][index][: math.ceil(budget * source["cells"])]] = True
        available.append(mask.reshape(receiver.shape))
        ages.append(age)
    return (
        torch.stack(features),
        torch.stack(density),
        torch.stack(available),
        torch.tensor(ages, device=receiver.device),
    )


def run_deadline_evaluation(cfg):
    """Deadline-constrained cooperation on consecutive frames of test sessions."""
    x = cfg["deadline_experiment"]
    output = Path(cfg["output"]["root"])
    if not x["enabled"]:
        return []
    sequences = deadline_sequences(cfg)
    device = device_for(cfg)
    predictions = defaultdict(list)
    totals = defaultdict(lambda: defaultdict(float))
    for seed in cfg["training"]["seeds"]:
        checkpoint = output / f"seed_{seed}" / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Train first; missing {checkpoint}")
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if not checkpoint_compatible(state, cfg):
            raise ValueError("Checkpoint data/model configuration mismatch")
        torch.manual_seed(seed)
        model = make_model(cfg, device)
        model.load_state_dict(state["model"])
        model.eval()
        for key, frames in sequences.items():
            run_deadline_sequence(
                model, cfg, device, seed, key, frames, predictions, totals
            )
            print(
                f"Deadline experiment seed={seed} session={key[0]} slice={key[1]} "
                f"frames={len(frames)}",
                flush=True,
            )
    classes = cfg["dataset"]["classes"]
    rows = []
    for key, frames in predictions.items():
        seed, trace_name, deadline, label = key
        stats = totals[key]
        metrics = [
            row
            for row in evaluate(
                frames,
                classes,
                cfg["evaluation"]["iou_thresholds"],
                class_thresholds=cfg["evaluation"]["class_iou_thresholds"],
            )
            if row["iou_level"] == 0
        ]
        scored = [row for row in metrics if row["ap_3d"] is not None]
        used = stats["fresh"] + stats["stale"]
        collaborator_frames = max(stats["collaborator_frames"], 1)
        rows.append(
            {
                "seed": seed,
                "trace": trace_name,
                "deadline_ms": deadline,
                "controller": label,
                "frames": len(frames),
                "map": _mean([row["ap_3d"] for row in scored]),
                **{f"ap_{row['class']}": row["ap_3d"] for row in metrics},
                "mean_kb_per_collaborator_frame": stats["bytes"]
                / collaborator_frames
                / 1000,
                "mean_budget": stats["budget"] / collaborator_frames,
                "miss_rate_sent": (
                    stats["missed"] / stats["sent"] if stats["sent"] else 0.0
                ),
                "miss_rate_frames": stats["missed"] / collaborator_frames,
                "fresh_fraction": stats["fresh"] / collaborator_frames,
                "stale_fraction": stats["stale"] / collaborator_frames,
                "none_fraction": stats["none"] / collaborator_frames,
                "mean_age_s": stats["age"] / used if used else 0.0,
            }
        )
    write_csv(output / "deadline_results.csv", rows)
    return rows


def range_bin(distance):
    if distance in ("", None):
        return "unknown"
    distance = float(distance)
    edges = (0.0, *RANGE_BINS_M, float("inf"))
    for low, high in zip(edges, edges[1:]):
        if distance < high:
            return f"{low:g}-{high:g}m" if math.isfinite(high) else f">={low:g}m"
    return "unknown"


def detection_diagnostics(cfg, output):
    """Range breakdown and false-positive map from per-detection dumps.

    Uses the full-ROI scope at the first IoU level. Distance is from the box
    centre to the nearest participating node.
    """
    parts = sorted(output.glob("eval_parts/seed_*/*.detections.csv.gz"))
    if not parts:
        return []
    counts = defaultdict(lambda: defaultdict(int))
    bounds = cfg["dataset"]["bounds"]
    edges_x = np.linspace(bounds[0], bounds[3], 101)
    edges_y = np.linspace(bounds[1], bounds[4], 101)
    false_positives = defaultdict(lambda: np.zeros((100, 100)))
    for part in parts:
        with gzip.open(part, "rt", newline="") as stream:
            for row in csv.DictReader(stream):
                key = (
                    row["weather"],
                    int(row["node_count"]),
                    row["strategy"],
                    row["class"],
                    range_bin(row["nearest_node_m"]),
                )
                counts[key][f'{row["record"]}:{row["status"]}'] += 1
                if row["record"] == "prediction" and row["status"] == "fp":
                    ix = np.searchsorted(edges_x, float(row["x"]), side="right") - 1
                    iy = np.searchsorted(edges_y, float(row["y"]), side="right") - 1
                    if 0 <= ix < 100 and 0 <= iy < 100:
                        false_positives[int(row["node_count"])][iy, ix] += 1
    rows = []
    for key, values in sorted(counts.items()):
        tp = values["prediction:tp"]
        fp = values["prediction:fp"]
        matched = values["ground_truth:matched"]
        missed = values["ground_truth:missed"]
        rows.append(
            dict(zip(("weather", "node_count", "strategy", "class", "range"), key))
            | {
                "true_positives": tp,
                "false_positives": fp,
                "ignored_predictions": values["prediction:ignored"],
                "matched_ground_truth": matched,
                "ground_truth": matched + missed,
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": matched / (matched + missed) if matched + missed else None,
            }
        )
    write_csv(output / "range_breakdown.csv", rows)
    if cfg["output"]["plots"] and false_positives:
        try:
            import matplotlib
        except ImportError:
            return rows
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        node_counts = sorted(false_positives)
        figure, axes = plt.subplots(
            1, len(node_counts), figsize=(4.5 * len(node_counts), 4.5), squeeze=False
        )
        for axis, count in zip(axes[0], node_counts):
            image = axis.imshow(
                np.log1p(false_positives[count]),
                origin="lower",
                extent=(bounds[0], bounds[3], bounds[1], bounds[4]),
                cmap="magma",
            )
            axis.set(
                title=f"False positives, {count} node(s)",
                xlabel="x (m)",
                ylabel="y (m)",
            )
            figure.colorbar(image, ax=axis, label="log(1 + count)")
        figure.tight_layout()
        figure.savefig(output / "false_positive_map.png", dpi=140)
        plt.close(figure)
    return rows


def report_context_sections(cfg, output):
    """Validation coverage, operating thresholds, and sensor point counts."""
    lines = []
    inventory_path = output / "class_inventory.json"
    if inventory_path.exists():
        inventory = json.loads(inventory_path.read_text())
        missing = [
            label
            for label in cfg["dataset"]["classes"]
            if inventory.get("val", {}).get(label, 0) == 0
            and inventory.get("test", {}).get(label, 0) > 0
        ]
        lines.extend(["## Validation coverage", ""])
        lines.append(
            "Classes with test but no validation ground truth (unseen by checkpoint "
            "selection and threshold tuning): " + (", ".join(missing) or "none") + "."
        )
        lines.append("")
    threshold_rows = []
    for seed in cfg["training"]["seeds"]:
        path = output / f"seed_{seed}" / "operating_thresholds.json"
        if path.exists():
            threshold_rows.append((seed, json.loads(path.read_text())["thresholds"]))
    if threshold_rows:
        classes = cfg["dataset"]["classes"]
        lines.extend(
            [
                "## Operating thresholds",
                "",
                "| Seed | " + " | ".join(classes) + " |",
                "|---:|" + "---:|" * len(classes),
            ]
        )
        for seed, values in threshold_rows:
            lines.append(
                f"| {seed} | "
                + " | ".join(
                    f"{values.get(label, float('nan')):.3f}" for label in classes
                )
                + " |"
            )
        lines.append("")
    inspection_path = output / "inspection.json"
    if inspection_path.exists():
        per_session = defaultdict(lambda: defaultdict(list))
        for entry in json.loads(inspection_path.read_text()):
            session = entry["sample"].split("/", 1)[0]
            for node, count in entry["points_in_bounds"].items():
                per_session[session][node].append(count)
        nodes = sorted(
            {node for values in per_session.values() for node in values},
            key=lambda value: (len(value), value),
        )
        if nodes:
            lines.extend(
                [
                    "## Sensor point counts",
                    "",
                    "Mean in-bounds points per node (thousands) over inspected slices; "
                    "a drop flags a degraded or obstructed sensor.",
                    "",
                    "| Session | " + " | ".join(nodes) + " |",
                    "|---|" + "---:|" * len(nodes),
                ]
            )
            for session, values in sorted(per_session.items()):
                lines.append(
                    f"| {session} | "
                    + " | ".join(
                        (
                            f"{np.mean(values[node]) / 1000:.1f}"
                            if values.get(node)
                            else "n/a"
                        )
                        for node in nodes
                    )
                    + " |"
                )
            lines.append("")
    return lines


def latency_feasibility_section(cfg, costs):
    """Required link rate and measured end-to-end latency per configuration."""
    groups = defaultdict(list)
    for row in costs:
        strategy = row["strategy"].split("@", 1)[0].split("_", 1)[0]
        groups[(strategy, int(row["node_count"]), row["placement"])].append(row)
    if not groups:
        return []
    bandwidths = sorted({float(row["bandwidth_mbps"]) for row in costs})
    lines = [
        "",
        "## Latency feasibility",
        "",
        f'Required link rate delivers each frame within one {cfg["latency_experiment"]["frame_period_ms"]:g} ms period; '
        "configurations above the link rate are infeasible (their queue grows without bound). "
        "See latency_tail_summary.csv (`queue_stable`) for deadline-miss rates.",
        "",
        "| Fusion | Nodes | Placement | Payload (kB) | Required (Mbps) | "
        + " | ".join(f"E2E @ {b:g} Mbps (ms)" for b in bandwidths)
        + " |",
        "|---|---:|---|---:|---:|" + "---:|" * len(bandwidths),
    ]

    def value(row, key):
        return float(row[key]) if row.get(key) not in ("", None) else float("nan")

    period_ms = cfg["latency_experiment"]["frame_period_ms"]
    for (strategy, count, placement), rows in sorted(groups.items()):
        payload_bytes = np.mean([value(row, "payload_bytes") for row in rows])
        payload_kb = payload_bytes / 1000
        required = payload_bytes * 8 / period_ms / 1000
        cells = []
        for bandwidth in bandwidths:
            selected = [r for r in rows if float(r["bandwidth_mbps"]) == bandwidth]
            latency = np.mean([value(r, "end_to_end_ms_measured") for r in selected])
            text = f"{latency:.0f}" if math.isfinite(latency) else "n/a"
            cells.append(text if required < bandwidth else f"infeasible ({text})")
        lines.append(
            f"| {strategy} | {count} | {placement} | {payload_kb:.1f} | {required:.1f} | "
            + " | ".join(cells)
            + " |"
        )
    return lines


def range_section(ranges):
    pooled = defaultdict(lambda: defaultdict(int))
    for row in ranges:
        key = (row["node_count"], row["strategy"], row["range"])
        for name in (
            "true_positives",
            "false_positives",
            "matched_ground_truth",
            "ground_truth",
        ):
            pooled[key][name] += int(row[name])
    lines = [
        "",
        "## Accuracy by range (full scope, IoU level 0, all classes)",
        "",
        "Range is from the box centre to the nearest participating node. "
        "Per-class rows are in range_breakdown.csv.",
        "",
        "| Nodes | Fusion | Range | Precision | Recall | False positives | Ground truth |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for (count, strategy, distance), values in sorted(pooled.items()):
        tp, fp = values["true_positives"], values["false_positives"]
        matched, gt = values["matched_ground_truth"], values["ground_truth"]
        precision = f"{tp / (tp + fp):.3f}" if tp + fp else "n/a"
        recall = f"{matched / gt:.3f}" if gt else "n/a"
        lines.append(
            f"| {count} | {strategy} | {distance} | {precision} | {recall} | {fp} | {gt} |"
        )
    return lines


def comparison_section(cfg, output):
    """Grouped BEV AP at the largest node count beside the published table."""
    path = output / "coinfra_comparison.csv"
    if cfg["evaluation"]["comparison"] is None or not path.is_file():
        return []
    with path.open() as stream:
        rows = [row for row in csv.DictReader(stream) if row["ap_3d"] not in ("", None)]
    if not rows:
        return []
    groups = list(cfg["evaluation"]["comparison"]["groups"])
    largest = max(int(row["node_count"]) for row in rows)
    pooled = defaultdict(list)
    for row in rows:
        count, strategy = int(row["node_count"]), row["strategy"]
        # One node is the same detector under every label; selector ablations
        # appear in the rate-accuracy tables.
        if count == 1 and strategy == "early":
            strategy = "single node"
        elif count != largest or count == 1 or "_" in strategy.split("@")[0]:
            continue
        pooled[(row["gt_scope"], count, strategy, row["class"])].append(
            float(row["ap_3d"])
        )
    protocol = (
        "the same random frame split as the published table"
        if cfg["dataset"]["split_mode"] == "random_frames"
        else "session-held-out splits, which the published random frame split "
        "does not test; compare with configs/coinfra_random_split.yaml"
    )
    iou = cfg["evaluation"]["comparison"]
    lines = [
        "",
        "## Comparison with the CoInfra benchmark metric",
        "",
        f"AP@{iou['iou_threshold']:g} with {iou['iou_type'].upper()} IoU over merged "
        f"classes ({', '.join(f'{k} = {'+'.join(v)}' for k, v in iou['groups'].items())}), "
        f"mean over sessions and seeds. These results use {protocol}. Published "
        "LiDAR rows are from arXiv 2507.02245v3, Table 5.",
        "",
        "| Scope | Nodes | Fusion | " + " | ".join(groups) + " |",
        "|---|---:|---|" + "---:|" * len(groups),
    ]
    keys = sorted({key[:3] for key in pooled})
    for scope, count, strategy in keys:
        values = [
            (
                f"{np.mean(pooled[(scope, count, strategy, group)]):.3f}"
                if pooled.get((scope, count, strategy, group))
                else "n/a"
            )
            for group in groups
        ]
        lines.append(f"| {scope} | {count} | {strategy} | " + " | ".join(values) + " |")
    for name, values in COINFRA_PUBLISHED_AP.items():
        cells = [
            f"{values[group]:.3f}" if group in values else "n/a" for group in groups
        ]
        lines.append(f"| published | 8 | CoInfra {name} | " + " | ".join(cells) + " |")
    return lines


def deadline_section(cfg, output):
    """Mean over seeds of each controller under each trace and deadline."""
    path = output / "deadline_results.csv"
    if not cfg["deadline_experiment"]["enabled"] or not path.is_file():
        return []
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    pooled = defaultdict(list)
    for row in rows:
        pooled[(row["trace"], float(row["deadline_ms"]), row["controller"])].append(row)
    x = cfg["deadline_experiment"]
    lines = [
        "",
        "## Deadline-constrained cooperation",
        "",
        f"Consecutive 10 Hz test frames ({x['frames_per_slice']} per slice). Each "
        "collaborator decides every frame what to send over its own uplink trace; "
        "a message that misses the receiver's fusion deadline is fused later as "
        f"a stale message for up to {x['max_age_frames']} frames. DRIVE targets a "
        f"late-message rate of {x['target_miss_rate']:g} per frame; each tradeoff V "
        "is one point of its utility-versus-miss-rate curve. The oracle knows the "
        "true throughput. mAP uses the visible scope at the primary IoU level.",
        "",
        "| Trace | Deadline (ms) | Controller | mAP | kB per collaborator-frame | "
        "Late per frame | Fresh | Stale | None | Mean age (s) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (trace, deadline, controller), group in sorted(pooled.items()):

        def mean(name):
            values = [float(row[name]) for row in group if row[name] not in ("", None)]
            return float(np.mean(values)) if values else float("nan")

        lines.append(
            f"| {trace} | {deadline:g} | {controller} | {mean('map'):.4f} | "
            f"{mean('mean_kb_per_collaborator_frame'):.1f} | "
            f"{mean('miss_rate_frames'):.3f} | {mean('fresh_fraction'):.2f} | "
            f"{mean('stale_fraction'):.2f} | {mean('none_fraction'):.2f} | "
            f"{mean('mean_age_s'):.2f} |"
        )
    return lines


def report(cfg):
    output = Path(cfg['output']['root'])
    with (output/'accuracy.csv').open() as stream:
        accuracy = list(csv.DictReader(stream))
    for row in accuracy:
        # Results from before ground-truth scopes were scored on the full ROI.
        if not row.get("gt_scope"):
            row["gt_scope"] = "full"
    with (output/'latency_communication.csv').open() as stream:
        costs = list(csv.DictReader(stream))

    def number(row, key):
        return float(row[key]) if row[key] not in ('', None) else None

    metric_names = ('ap_3d','precision','recall','f1','mean_matched_iou_3d',
                    'mean_center_error_m','mean_yaw_error_deg','mean_size_iou')
    # Thresholds may differ by class, so groups across classes use the IoU level.
    dimensions = (
        "gt_scope",
        "weather",
        "node_count",
        "strategy",
        "class",
        "iou_level",
        "iou",
    )
    aggregates = defaultdict(list)
    for row in accuracy:
        if number(row, 'ap_3d') is not None:
            aggregates[tuple(row[k] for k in dimensions)].append(row)
    summary = []
    for key, rows in sorted(aggregates.items()):
        values = {name:[number(row,name) for row in rows if number(row,name) is not None]
                  for name in metric_names}
        result = dict(zip(dimensions,key))
        labels = {'ap_3d':'ap','mean_matched_iou_3d':'matched_iou_3d',
                  'mean_center_error_m':'center_error_m','mean_yaw_error_deg':'yaw_error_deg',
                  'mean_size_iou':'size_iou'}
        for name, data in values.items():
            label = labels.get(name,name)
            result[f'mean_{label}'] = float(np.mean(data)) if data else None
            result[f'std_{label}'] = float(np.std(data)) if data else None
        result['observations'] = len(rows)
        result['matched_observations'] = len(values['mean_matched_iou_3d'])
        summary.append(result)
    write_csv(output/'accuracy_summary.csv',summary)

    overview_groups = defaultdict(list)
    overview_dimensions = ("gt_scope", "weather", "node_count", "strategy", "iou_level")
    for row in accuracy:
        if number(row, 'ap_3d') is not None:
            overview_groups[tuple(row[k] for k in overview_dimensions)].append(row)
    overview = []
    for key, rows in sorted(overview_groups.items()):
        tp = sum(int(row["true_positives"]) for row in rows)
        fp = sum(int(row["false_positives"]) for row in rows)
        fn = sum(int(row["false_negatives"]) for row in rows)
        macro = lambda name: float(np.mean([number(row, name) for row in rows]))
        micro_precision = tp / (tp + fp) if tp + fp else 0.0
        micro_recall = tp / (tp + fn) if tp + fn else 0.0
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if micro_precision + micro_recall
            else 0.0
        )
        matched = [
            row for row in rows if number(row, "mean_matched_iou_3d") is not None
        ]
        matched_count = sum(int(row["true_positives"]) for row in matched)
        weighted = lambda name: (
            sum(number(row, name) * int(row["true_positives"]) for row in matched)
            / matched_count
            if matched_count
            else None
        )
        overview.append(
            dict(zip(overview_dimensions, key))
            | {
                "map_3d_macro": macro("ap_3d"),
                "precision_macro": macro("precision"),
                "recall_macro": macro("recall"),
                "f1_macro": macro("f1"),
                "precision_micro": micro_precision,
                "recall_micro": micro_recall,
                "f1_micro": micro_f1,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "mean_matched_iou_3d": weighted("mean_matched_iou_3d"),
                "mean_center_error_m": weighted("mean_center_error_m"),
                "mean_yaw_error_deg": weighted("mean_yaw_error_deg"),
                "mean_size_iou": weighted("mean_size_iou"),
                "class_observations": len(rows),
                "iou_thresholds": ";".join(
                    sorted({f"{row['class']}:{row['iou']}" for row in rows})
                ),
            }
        )
    write_csv(output / "metrics_overview.csv", overview)

    comparison_dimensions = (
        "gt_scope",
        "seed",
        "session",
        "weather",
        "strategy",
        "class",
        "iou_level",
        "iou",
    )
    single_nodes = defaultdict(dict)
    multi_subsets = defaultdict(list)
    for row in accuracy:
        ap, recall = number(row, 'ap_3d'), number(row, 'recall')
        if ap is None:
            continue
        key = tuple(row[k] for k in comparison_dimensions)
        nodes = tuple(sorted(filter(None, row["nodes"].split(","))))
        if int(row["node_count"]) == 1:
            single_nodes[key][nodes[0]] = (ap, recall)
        else:
            # Receiver rotations share one unordered participant set. Average
            # those rotations before comparing with the same participating nodes.
            multi_subsets[(key, nodes)].append((ap, recall))
    gains = []
    for (key, nodes), values in sorted(multi_subsets.items()):
        baselines = single_nodes.get(key, {})
        if any(node not in baselines for node in nodes):
            continue
        baseline_ap = [baselines[node][0] for node in nodes]
        baseline_recall = [baselines[node][1] for node in nodes]
        aps, recalls = zip(*values)
        mean_ap, mean_recall = float(np.mean(aps)), float(np.mean(recalls))
        baseline_ap_mean = float(np.mean(baseline_ap))
        baseline_recall_mean = float(np.mean(baseline_recall))
        gains.append(
            dict(zip(comparison_dimensions, key))
            | {
                "nodes": ",".join(nodes),
                "node_count": len(nodes),
                "receiver_rotations": len(values),
                "multi_node_ap_mean": mean_ap,
                "single_node_ap_mean": baseline_ap_mean,
                "single_node_ap_best": max(baseline_ap),
                "ap_gain_over_mean_single": mean_ap - baseline_ap_mean,
                "ap_gain_over_best_single": mean_ap - max(baseline_ap),
                "multi_node_recall_mean": mean_recall,
                "single_node_recall_mean": baseline_recall_mean,
                "single_node_recall_best": max(baseline_recall),
                "recall_gain_over_mean_single": mean_recall - baseline_recall_mean,
                "recall_gain_over_best_single": mean_recall - max(baseline_recall),
            }
        )
    write_csv(output/'cooperation_gains.csv',gains)

    weather_groups = defaultdict(list)
    for row in gains:
        key = tuple(row[k] for k in dimensions)
        weather_groups[key].append(row)
    weather_summary = []
    for key, values in sorted(weather_groups.items()):
        ap = [v['ap_gain_over_mean_single'] for v in values]
        recall = [v['recall_gain_over_mean_single'] for v in values]
        best = [v["ap_gain_over_best_single"] for v in values]
        worst = min(values, key=lambda v: v["ap_gain_over_best_single"])
        weather_summary.append(
            dict(zip(dimensions, key))
            | {
                "mean_ap_gain": float(np.mean(ap)),
                "std_ap_gain": float(np.std(ap)),
                "ap_positive_rate": float(np.mean(np.asarray(ap) > 0)),
                "mean_recall_gain": float(np.mean(recall)),
                "std_recall_gain": float(np.std(recall)),
                "recall_positive_rate": float(np.mean(np.asarray(recall) > 0)),
                # Against the best participating node, and the worst paired case,
                # so a session where cooperation hurts is not averaged away.
                "mean_ap_gain_over_best": float(np.mean(best)),
                "min_ap_gain_over_best": float(worst["ap_gain_over_best_single"]),
                "min_ap_gain_session": worst["session"],
                "min_ap_gain_nodes": worst["nodes"],
                "negative_over_best": int(np.sum(np.asarray(best) < 0)),
                "paired_observations": len(values),
            }
        )
    write_csv(output / "weather_cooperation_summary.csv", weather_summary)

    paired = defaultdict(lambda: defaultdict(list))
    for row in accuracy:
        ap, recall = number(row, "ap_3d"), number(row, "recall")
        if ap is not None:
            key = tuple(row[k] for k in comparison_dimensions)
            paired[key][int(row["node_count"])].append((ap, recall))

    scaling = []
    for key, counts in sorted(paired.items()):
        ordered = sorted(counts)
        for previous, current in zip(ordered, ordered[1:]):
            previous_ap = float(np.mean([v[0] for v in counts[previous]]))
            current_ap = float(np.mean([v[0] for v in counts[current]]))
            previous_recall = float(np.mean([v[1] for v in counts[previous]]))
            current_recall = float(np.mean([v[1] for v in counts[current]]))
            scaling.append(
                dict(zip(comparison_dimensions, key))
                | {
                    "from_node_count": previous,
                    "to_node_count": current,
                    "ap_change": current_ap - previous_ap,
                    "recall_change": current_recall - previous_recall,
                }
            )
    write_csv(output / "node_scaling.csv", scaling)

    costs_by_run = defaultdict(list)
    run_keys = ('seed','session','weather','nodes','node_count','strategy')
    for row in costs:
        costs_by_run[tuple(row[k] for k in run_keys)].append(row)
    tradeoffs = []
    for row in accuracy:
        for cost in costs_by_run[tuple(row[k] for k in run_keys)]:
            tradeoffs.append(
                {k: row[k] for k in run_keys}
                | {
                    "gt_scope": row["gt_scope"],
                    "class": row["class"],
                    "iou_level": row["iou_level"],
                    "iou": row["iou"],
                    "ap_3d": row["ap_3d"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1": row["f1"],
                    "placement": cost["placement"],
                    "bandwidth_mbps": cost["bandwidth_mbps"],
                    "payload_bytes": cost["payload_bytes"],
                    "inference_decode_ms_mean": cost["inference_decode_ms_mean"],
                    "inference_decode_ms_p95": cost["inference_decode_ms_p95"],
                    "preprocessing_ms_mean": cost["preprocessing_ms_mean"],
                    "serialization_ms_mean": cost["serialization_ms_mean"],
                    "transmission_ms_assumed": cost["transmission_ms_model"],
                    "end_to_end_ms_assumed": cost["end_to_end_ms_model"],
                    "end_to_end_ms_measured": cost.get("end_to_end_ms_measured"),
                    "required_bandwidth_mbps": cost.get("required_bandwidth_mbps"),
                }
            )
    write_csv(output / "accuracy_latency_tradeoffs.csv", tradeoffs)

    adaptive_groups = defaultdict(list)
    for row in tradeoffs:
        if (
            row["strategy"].startswith("adaptive")
            and row["placement"] == "edge"
            and float(row["bandwidth_mbps"])
            == max(cfg["deployment_assumptions"]["bandwidths_mbps"])
            and row["ap_3d"] not in ("", None)
        ):
            budget = float(row["strategy"].split("@", 1)[1])
            selector = adaptive_selector(row["strategy"])
            key = tuple(
                row[k]
                for k in (
                    "gt_scope",
                    "weather",
                    "node_count",
                    "class",
                    "iou_level",
                    "iou",
                    "strategy",
                )
            ) + (selector,)
            adaptive_groups[key].append(
                (float(row["ap_3d"]), float(row["payload_bytes"]))
            )
    adaptive_rate = []
    for key, values in sorted(adaptive_groups.items()):
        scope, weather, node_count, label, level, iou, strategy, selector = key
        adaptive_rate.append(
            {
                "gt_scope": scope,
                "weather": weather,
                "node_count": node_count,
                "class": label,
                "iou_level": level,
                "iou": iou,
                "strategy": strategy,
                "selector": selector,
                "budget_fraction": float(strategy.split("@", 1)[1]),
                "mean_ap": float(np.mean([value[0] for value in values])),
                "std_ap": float(np.std([value[0] for value in values])),
                "mean_payload_bytes": float(np.mean([value[1] for value in values])),
                "observations": len(values),
            }
        )
    write_csv(output / "adaptive_rate_accuracy.csv", adaptive_rate)

    auc_dimensions = (
        "gt_scope",
        "weather",
        "node_count",
        "class",
        "iou_level",
        "iou",
        "selector",
    )
    auc_groups = defaultdict(list)
    for row in adaptive_rate:
        key = tuple(row[k] for k in auc_dimensions)
        auc_groups[key].append((row["budget_fraction"], row["mean_ap"]))
    adaptive_auc = []
    for key, values in sorted(auc_groups.items()):
        points = sorted(set(values))
        if len(points) < 2:
            continue
        x = np.asarray([point[0] for point in points], dtype=float)
        y = np.asarray([point[1] for point in points], dtype=float)
        log_x = np.log(x)
        normalized_auc = float(np.trapezoid(y, log_x) / (log_x[-1] - log_x[0]))
        adaptive_auc.append(
            dict(zip(auc_dimensions, key))
            | {
                "normalized_log_budget_ap_auc": normalized_auc,
                "minimum_budget": float(x[0]),
                "maximum_budget": float(x[-1]),
                "operating_points": len(points),
            }
        )
    write_csv(output / "adaptive_rate_accuracy_auc.csv", adaptive_auc)

    ranges = detection_diagnostics(cfg, output)
    if cfg['output']['plots'] and summary:
        try:
            import matplotlib
        except ImportError:
            print('Report plot skipped: install matplotlib from requirements.txt', flush=True)
            matplotlib = None
        if matplotlib is not None:
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            primary_iou = cfg["evaluation"]["iou_thresholds"][0]
            adaptive_primary = (
                f'adaptive@{max(cfg["adaptive_fusion"]["evaluation_budgets"]):g}'
            )
            plotted_strategies = {"early", "intermediate", "late", adaptive_primary}
            grouped = defaultdict(list)
            for row in summary:
                if int(row["iou_level"]) == 0 and row["strategy"] in plotted_strategies:
                    label = (
                        "adaptive"
                        if row["strategy"] == adaptive_primary
                        else row["strategy"]
                    )
                    grouped[(label, row["gt_scope"])].append(
                        (int(row["node_count"]), float(row["mean_ap"]))
                    )
            figure, axis = plt.subplots(figsize=(10, 6))
            counts_seen = set()
            for (strategy, scope), values in sorted(grouped.items()):
                by_count = defaultdict(list)
                for count, value in values:
                    by_count[count].append(value)
                x = sorted(by_count)
                counts_seen.update(x)
                y = [float(np.mean(by_count[count])) for count in x]
                axis.plot(
                    x,
                    y,
                    marker="o",
                    linestyle="-" if scope == "full" else "--",
                    label=f"{strategy} ({scope} GT)",
                )
            # Node counts double, so a log2 axis spaces them evenly.
            axis.set_xscale("log", base=2)
            axis.set_xticks(sorted(counts_seen), [str(c) for c in sorted(counts_seen)])
            axis.set(
                xlabel="node count",
                ylabel="macro mean AP",
                title=(
                    f"Test AP by node count and fusion (IoU level {primary_iou:g}; "
                    "class overrides applied)"
                ),
            )
            axis.set_ylim(0, 1)
            axis.legend(fontsize=8, ncol=2)
            axis.grid(alpha=.25)
            figure.tight_layout()
            figure.savefig(output/'accuracy_by_nodes.png', dpi=160)
            plt.close(figure)
            primary_gains = [
                row for row in weather_summary if int(row["iou_level"]) == 0
            ]
            if primary_gains:
                scopes = sorted({row["gt_scope"] for row in primary_gains})
                figure, axes = plt.subplots(
                    1, len(scopes), figsize=(7 * len(scopes), 5), squeeze=False
                )
                for axis, scope in zip(axes[0], scopes):
                    grouped = defaultdict(list)
                    for row in primary_gains:
                        if row["gt_scope"] == scope:
                            grouped[
                                (
                                    row["weather"],
                                    row["strategy"],
                                    int(row["node_count"]),
                                )
                            ].append(float(row["mean_ap_gain_over_best"]))
                    labels = sorted({key[:2] for key in grouped})
                    counts = sorted({key[2] for key in grouped})
                    width = 0.8 / max(1, len(counts))
                    for offset, count in enumerate(counts):
                        axis.bar(
                            np.arange(len(labels)) + offset * width,
                            [
                                float(np.mean(grouped.get(label + (count,), [np.nan])))
                                for label in labels
                            ],
                            width,
                            label=f"{count} nodes",
                        )
                    axis.set_xticks(
                        np.arange(len(labels)) + width * (len(counts) - 1) / 2,
                        ["/".join(label) for label in labels],
                        rotation=45,
                    )
                    axis.axhline(0, color="black", linewidth=0.8)
                    axis.set(
                        ylabel="mean AP gain over best member node",
                        title=f"Cooperation gain, IoU level 0 ({scope} GT)",
                    )
                    axis.legend(fontsize=8)
                figure.tight_layout()
                figure.savefig(output/'cooperation_gain_by_weather.png', dpi=160)
                plt.close(figure)

            figure, axis = plt.subplots(figsize=(9, 5))
            latency_groups = defaultdict(list)
            for row in costs:
                strategy = row["strategy"].split("@", 1)[0].split("_", 1)[0]
                measured = row.get("end_to_end_ms_measured")
                latency_groups[
                    (row["placement"], strategy, float(row["bandwidth_mbps"]))
                ].append(float(measured if measured else row["end_to_end_ms_model"]))
            for placement, strategy in sorted(
                {(key[0], key[1]) for key in latency_groups}
            ):
                points = sorted(
                    (bandwidth, float(np.mean(values)))
                    for (name, method, bandwidth), values in latency_groups.items()
                    if name == placement and method == strategy
                )
                if points:
                    axis.plot(
                        [p[0] for p in points],
                        [p[1] for p in points],
                        marker="o",
                        label=f"{placement}/{strategy}",
                    )
            axis.set(
                xlabel="assumed bandwidth (Mbps)",
                ylabel="mean end-to-end latency (ms)",
                title="End-to-end latency: measured compute, assumed sensing/RTT/link",
                xscale="log",
                yscale="log",
            )
            axis.legend()
            axis.grid(alpha=.25)
            figure.tight_layout()
            figure.savefig(output/'latency_by_bandwidth.png', dpi=160)
            plt.close(figure)
    advercity = cfg["dataset"].get("type") == "advercity"
    primary = (
        "Primary question: how much do additional simulated vehicle and roadside LiDAR agents improve detection?"
        if advercity
        else "Primary question: how much do additional real roadside LiDAR nodes improve detection?"
    )
    secondary = (
        "Secondary question: how does that cooperation gain vary across controlled simulated weather categories?"
        if advercity
        else "Secondary question: does that cooperation gain persist across recorded weather conditions?"
    )
    weather_limit = (
        "Adver-City weather is simulated and categorical; no numerical meteorological visibility is inferred."
        if advercity
        else "Weather sessions differ in traffic/date; weather comparisons alone do not establish a causal weather effect."
    )
    lines = [
        "# ECCO cooperation-first experiment report",
        "",
        primary,
        secondary,
        "",
        *(
            [
                "**Smoke-run notice:** this report validates the end-to-end pipeline on a tiny "
                "sample. Its accuracy and cooperation metrics are not experimental results.",
                "",
            ]
            if cfg["dataset"].get("smoke", {}).get("enabled")
            else []
        ),
        "Accuracy: rotated 3D, globally confidence-ranked AP within each session/subset/seed, using candidates retained at the configured AP score threshold.",
        (
            "Precision, recall, F1, and localization diagnostics use per-class operating score thresholds tuned for maximum F1 on validation (see Operating thresholds), after class-wise NMS."
            if cfg["evaluation"]["tune_score_thresholds"]
            else "Precision, recall, F1, and localization diagnostics use the configured operating score threshold after class-wise NMS."
        ),
        (
            "Detection scores are IoU-rectified: heat^(1-a) * predicted IoU^a with "
            f'a={cfg["evaluation"]["iou_score_weight"]:g}.'
        ),
        (
            "Ground-truth scopes: `full` scores every box in the fixed global ROI (system coverage); "
            f'`visible` scores boxes with at least {cfg["evaluation"]["min_visible_points"]} points from the '
            "subset's own nodes and ignores the rest, isolating fusion quality from added coverage."
        ),
        "Matched-object diagnostics include 3D IoU, center error, yaw error, and dimension IoU.",
        "All aggregates are descriptive means, not confidence intervals or causal weather estimates.",
        "Single-node rows cover every common real node. The fixed global ROI/GT is unchanged across counts in the full scope.",
        "Measured inference+decode uses batched node encoding and includes decode/NMS; excludes loading/rasterization/serialization.",
        "With one node every strategy reduces to the same detector; it is computed once and reported under each strategy label.",
        (
            "IoU levels: "
            + ", ".join(f"{value:g}" for value in cfg["evaluation"]["iou_thresholds"])
            + "; per-class overrides: "
            + (
                ", ".join(
                    f"{label} {'/'.join(f'{v:g}' for v in values)}"
                    for label, values in cfg["evaluation"][
                        "class_iou_thresholds"
                    ].items()
                )
                or "none"
            )
            + "."
        ),
        "End-to-end latency adds measured preprocessing, inference+decode, and serialization to assumed sensing, RTT, and link rate (`end_to_end_ms_measured`); `end_to_end_ms_model` keeps the assumed processing constants.",
        "Early fusion sends float32 raw points; dense intermediate features and adaptive sparse messages use the configured value width (sparse messages add index width); late fusion sends float32 boxes at the operating score threshold. All payloads use zlib level 1.",
        (
            "Adaptive fusion rotates every participating node as the dense local reference and budgets collaborator cells only."
            if cfg["evaluation"].get("receiver_policy") == "all"
            else "Adaptive fusion uses the first sorted participating node as the dense local reference and budgets collaborator cells only."
        ),
        "Cloud placement uploads raw cropped points from every participating node, receiver included; edge sends collaborator raw points/features/local predictions according to strategy.",
        f'Assumption source: {cfg["deployment_assumptions"]["source"]}.',
        weather_limit,
        "cooperation_gains.csv uses paired comparisons within scope/session/seed/class/strategy/IoU.",
        "",
    ]
    lines.extend(report_context_sections(cfg, output))
    lines.extend(
        [
            "## Cooperation gain by weather (IoU level 0)",
            "",
            "Gain over mean and over best member node; the worst paired case keeps a "
            "session where cooperation hurts visible.",
            "",
            "| Scope | Weather | Nodes | Fusion | Class | IoU | AP gain (mean) | AP gain (best) "
            "| Worst AP gain (best) | Worst session | Recall gain | AP positive rate |",
            "|---|---|---:|---|---|---:|---:|---:|---:|---|---:|---:|",
        ]
    )
    for row in weather_summary:
        if int(row["iou_level"]) != 0:
            continue
        lines.append(
            f'| {row["gt_scope"]} | {row["weather"]} | {row["node_count"]} | '
            f'{row["strategy"]} | {row["class"]} | {row["iou"]} | '
            f'{row["mean_ap_gain"]:.4f} | {row["mean_ap_gain_over_best"]:.4f} | '
            f'{row["min_ap_gain_over_best"]:.4f} | {row["min_ap_gain_session"]} | '
            f'{row["mean_recall_gain"]:.4f} | {row["ap_positive_rate"]:.2f} |'
        )
    harmful = sorted(
        (row for row in gains if row["ap_gain_over_best_single"] < -0.02),
        key=lambda row: row["ap_gain_over_best_single"],
    )
    lines.extend(
        [
            "",
            "## Cases where cooperation reduced AP",
            "",
            "Paired subsets whose AP is more than 0.02 below their best member node "
            f"({len(harmful)} cases; at most 40 shown, worst first).",
            "",
        ]
    )
    if harmful:
        lines.extend(
            [
                "| Scope | Session | Nodes | Fusion | Class | IoU | Subset AP | Best member AP | Gain |",
                "|---|---|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in harmful[:40]:
            lines.append(
                f'| {row["gt_scope"]} | {row["session"]} | {row["nodes"]} | '
                f'{row["strategy"]} | {row["class"]} | {row["iou"]} | '
                f'{row["multi_node_ap_mean"]:.4f} | {row["single_node_ap_best"]:.4f} | '
                f'{row["ap_gain_over_best_single"]:.4f} |'
            )
    else:
        lines.append("None.")
    lines.extend(latency_feasibility_section(cfg, costs))
    if ranges:
        lines.extend(range_section(ranges))
    lines.extend(comparison_section(cfg, output))
    lines.extend(deadline_section(cfg, output))
    lines.extend(
        [
            "",
            "## Absolute accuracy",
            "",
            "| Scope | Weather | Nodes | Fusion | Class | IoU | Mean AP | Precision | Recall | F1 | Center error (m) |",
            "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary:
        center = (
            "n/a"
            if row["mean_center_error_m"] is None
            else f'{row["mean_center_error_m"]:.3f}'
        )
        lines.append(
            f'| {row["gt_scope"]} | {row["weather"]} | {row["node_count"]} | {row["strategy"]} | '
            f'{row["class"]} | {row["iou"]} | {row["mean_ap"]:.4f} | '
            f'{row["mean_precision"]:.4f} | {row["mean_recall"]:.4f} | '
            f'{row["mean_f1"]:.4f} | {center} |'
        )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(f'Report: {output / "report.md"}')


def verify_results(cfg):
    """Fail the unattended run if required artifacts or metric invariants are invalid."""
    output = Path(cfg["output"]["root"])
    required = [
        "index.json",
        "inventory.json",
        "class_inventory.json",
        "split_isolation.json",
        "inspection.json",
        "resolved_config.json",
        "environment.json",
        "accuracy.csv",
        "accuracy_summary.csv",
        "metrics_overview.csv",
        "latency_communication.csv",
        "latency_trace.csv",
        "latency_tail_summary.csv",
        "accuracy_latency_tradeoffs.csv",
        "report.md",
    ]
    allowed_empty = [
        "cooperation_gains.csv",
        "weather_cooperation_summary.csv",
        "node_scaling.csv",
        "adaptive_rate_accuracy.csv",
        "adaptive_rate_accuracy_auc.csv",
    ]
    checks, errors = {}, []
    for name in required:
        path = output/name
        checks[f'artifact:{name}'] = path.is_file() and path.stat().st_size > 0
        if not checks[f'artifact:{name}']:
            errors.append(f'missing or empty artifact: {name}')
    for name in allowed_empty:
        path = output/name
        checks[f'artifact:{name}'] = path.is_file()
        if not checks[f'artifact:{name}']:
            errors.append(f'missing artifact: {name}')
    optional = []
    if cfg["deadline_experiment"]["enabled"]:
        optional.append("deadline_results.csv")
    if cfg["evaluation"]["comparison"] is not None:
        optional.append("coinfra_comparison.csv")
    for name in optional:
        path = output / name
        checks[f"artifact:{name}"] = path.is_file() and path.stat().st_size > 0
        if not checks[f"artifact:{name}"]:
            errors.append(f"missing or empty artifact: {name}")
    if cfg["evaluation"]["save_detections"]:
        path = output / "range_breakdown.csv"
        checks["artifact:range_breakdown.csv"] = path.is_file()
        if not path.is_file():
            errors.append("missing artifact: range_breakdown.csv")
    for seed in cfg["training"]["seeds"]:
        names = [
            "best.pt",
            "latest.pt",
            "complete.json",
            f"training_metrics_seed_{seed}.csv",
        ]
        if cfg["evaluation"]["tune_score_thresholds"]:
            names.append("operating_thresholds.json")
        for name in names:
            path = output/f'seed_{seed}'/name
            checks[f'seed_{seed}:{name}'] = path.is_file() and path.stat().st_size > 0
            if not checks[f'seed_{seed}:{name}']:
                errors.append(f'missing or empty seed artifact: seed_{seed}/{name}')
    metric_rows = []
    if (output/'accuracy.csv').is_file():
        with (output/'accuracy.csv').open() as stream:
            metric_rows = list(csv.DictReader(stream))
    checks['accuracy_rows_present'] = bool(metric_rows)
    if not metric_rows:
        errors.append('accuracy.csv contains no metric rows')
    scopes = {row.get("gt_scope") for row in metric_rows}
    checks["gt_scopes_present"] = not metric_rows or scopes == set(GT_SCOPES)
    if not checks["gt_scopes_present"]:
        errors.append(
            f"accuracy.csv ground-truth scopes {sorted(map(str, scopes))} != {list(GT_SCOPES)}"
        )
    bounded = (
        "ap_3d",
        "precision",
        "recall",
        "f1",
        "mean_matched_iou_3d",
        "mean_size_iou",
    )
    for line, row in enumerate(metric_rows, start=2):
        for name in bounded:
            value = row.get(name)
            if value not in ("", None) and (
                not math.isfinite(float(value)) or not 0 <= float(value) <= 1
            ):
                errors.append(f"accuracy.csv line {line}: invalid {name}={value}")
        tp, fp, fn = (
            int(row[name])
            for name in ("true_positives", "false_positives", "false_negatives")
        )
        if tp + fp != int(row["predictions"]) or tp + fn != int(row["ground_truth"]):
            errors.append(f"accuracy.csv line {line}: TP/FP/FN totals are inconsistent")
        if int(row["predictions"]) > int(
            row.get("ranked_predictions", row["predictions"])
        ):
            errors.append(
                f"accuracy.csv line {line}: operating predictions exceed ranked predictions"
            )
        for name in ('mean_center_error_m','mean_yaw_error_deg'):
            value = row.get(name)
            if value not in ('',None) and (not math.isfinite(float(value)) or float(value) < 0):
                errors.append(f'accuracy.csv line {line}: invalid {name}={value}')
    checks['metric_invariants'] = not errors
    result = {'timestamp_utc':utc_now(),'status':'passed' if not errors else 'failed',
              'accuracy_rows':len(metric_rows),'checks':checks,'errors':errors}
    write_json(output/'verification.json',result)
    if errors:
        raise ValueError(
            "Result verification failed; see verification.json: "
            + "; ".join(errors[:5])
        )
    print(f'Result verification passed: {output / "verification.json"}', flush=True)
    return result
