"""ECCO preprocessing, training, inference, evaluation, and reporting pipeline."""
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
import platform
import shutil
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np
import torch
from src.data.api import annotations, discover, load_cloud, rasterize
from src.detection.bev import BEVDetector, targets, loss, decode
from src.metrics.evaluation import evaluate, nms

EVALUATION_VERSION = 2


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


def validate_examples(model, examples, cfg, device, retries):
    """Validate one prepared batch, splitting only that batch after CUDA OOM."""
    try:
        early = device_tensor(np.stack([item[0] for item in examples]), device)
        per_node = device_tensor(np.stack([item[1] for item in examples]), device)
        target = tuple(
            device_tensor(np.stack([item[3][i] for item in examples]), device)
            for i in range(3)
        )
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=mixed_precision(cfg, device),
        ):
            early_outputs = model(early)
            batch_loss = float(loss(early_outputs, target))
            batch_size, nodes, channels, height, width = per_node.shape
            features = model.encoder(
                per_node.reshape(batch_size * nodes, channels, height, width)
            ).reshape(batch_size, nodes, -1, height, width)
            intermediate = (
                model.heads(features.amax(dim=1))
                if "intermediate" in cfg["evaluation"]["strategies"]
                else None
            )
            adaptive = None
            if "adaptive" in cfg["evaluation"]["strategies"]:
                validation_budget = cfg["adaptive_fusion"]["evaluation_budgets"][0]
                fused, _ = model.adaptive_features(features, validation_budget)
                adaptive = model.heads(fused)
            late = (
                model.heads(features.reshape(batch_size * nodes, *features.shape[2:]))
                if "late" in cfg["evaluation"]["strategies"]
                else None
            )
        frames = {strategy: [] for strategy in cfg["evaluation"]["strategies"]}
        for index, item in enumerate(examples):
            if "early" in frames:
                frames["early"].append(
                    (decode(batch_outputs(early_outputs, index), cfg), item[2])
                )
            if intermediate is not None:
                frames["intermediate"].append(
                    (decode(batch_outputs(intermediate, index), cfg), item[2])
                )
            if adaptive is not None:
                frames["adaptive"].append(
                    (decode(batch_outputs(adaptive, index), cfg), item[2])
                )
            if late is not None:
                offset = index * nodes
                local = [
                    decode(batch_outputs(late, offset + node), cfg)
                    for node in range(nodes)
                ]
                frames["late"].append(
                    (
                        nms(
                            [box for group in local for box in group],
                            cfg["evaluation"]["nms_iou"],
                            cfg["evaluation"]["late_max_detections"],
                        ),
                        item[2],
                    )
                )
        return [batch_loss] * len(examples), frames
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
            strategy: left[1][strategy] + right[1][strategy]
            for strategy in cfg["evaluation"]["strategies"]
        }
    if retries > 0:
        wait_for_resources(cfg, device, "single-example validation CUDA OOM")
        return validate_examples(model, examples, cfg, device, retries - 1)
    raise ValueError(
        "CUDA cannot fit one validation example after resource retries; reduce "
        "resolution, hidden_channels, or node_counts, or use training.device=cpu. "
        f"Original error: {oom_message}"
    )


def _validation_metrics(model, samples, cfg, device, batch_size):
    losses = []
    frames = {strategy: [] for strategy in cfg['evaluation']['strategies']}
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
            for strategy in frames:
                frames[strategy].extend(batch_frames[strategy])
    details, all_aps, all_precisions, all_recalls, all_f1s = {}, [], [], [], []
    for strategy, strategy_frames in frames.items():
        rows = evaluate(strategy_frames, cfg['dataset']['classes'], cfg['evaluation']['iou_thresholds'])
        aps = [r['ap_3d'] for r in rows if r['ap_3d'] is not None]
        precisions = [r['precision'] for r in rows if r['precision'] is not None]
        recalls = [r['recall'] for r in rows if r['recall'] is not None]
        f1s = [r['f1'] for r in rows if r['f1'] is not None]
        details[f'val_map_{strategy}'] = float(np.mean(aps)) if aps else 0.0
        details[f'val_precision_{strategy}'] = float(np.mean(precisions)) if precisions else 0.0
        details[f'val_recall_{strategy}'] = float(np.mean(recalls)) if recalls else 0.0
        details[f'val_f1_{strategy}'] = float(np.mean(f1s)) if f1s else 0.0
        all_aps.extend(aps)
        all_precisions.extend(precisions)
        all_recalls.extend(recalls)
        all_f1s.extend(f1s)
    val_loss = float(np.mean(losses))
    details['val_precision'] = float(np.mean(all_precisions)) if all_precisions else 0.0
    details['val_f1'] = float(np.mean(all_f1s)) if all_f1s else 0.0
    return (val_loss, float(np.mean(all_aps)) if all_aps else 0.0,
            float(np.mean(all_recalls)) if all_recalls else 0.0, details)


def validation_metrics(model, samples, cfg, device, runtime_batches=None, retries=None):
    """Validate in batches; OOM recovery is local to each failing batch."""
    if runtime_batches is None:
        runtime_batches = {"evaluation": cfg["evaluation"]["batch_size"]}
    return _validation_metrics(
        model, samples, cfg, device, runtime_batches["evaluation"]
    )


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
        gpu_free = None
        if device.type == 'cuda':
            physical_free, _ = torch.cuda.mem_get_info(device)
            reclaimable = max(
                0,
                torch.cuda.memory_reserved(device)
                - torch.cuda.memory_allocated(device),
            )
            gpu_free = physical_free + reclaimable
        disk_ok = disk_free >= disk_required
        gpu_ok = gpu_free is None or gpu_free >= gpu_required
        if disk_ok and gpu_ok:
            return
        elapsed = time.monotonic() - started
        timeout = settings['timeout_minutes'] * 60
        details = f'disk_free={disk_free/1e9:.2f}GB'
        if gpu_free is not None:
            details += f' gpu_free={gpu_free/1e9:.2f}GB'
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
    )
    training_keys = (
        "hidden_channels",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "mixed_precision",
        "modes",
        "node_counts",
    )
    return {
        "model_version": 2,
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


def audit_split_isolation(samples):
    """Fail closed when any recording or source artifact crosses a split."""
    owners = {}
    sessions = {}
    for sample in samples:
        split = sample["split"]
        session = sample["session"]
        previous = sessions.setdefault(session, split)
        if previous != split:
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
        "status": "passed",
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
    isolation = audit_split_isolation(samples)
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
    write_json(
        output / "environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "numpy": np.__version__,
        },
    )
    write_json(output / "inventory.json", dict(counts))
    write_json(output / "split_isolation.json", isolation)
    print(json.dumps(dict(counts), indent=2))
    return samples


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
    for sample in samples:
        boxes = annotations(sample["gt"], cfg["dataset"])
        validate_target_centers(boxes, cfg["dataset"], sample["id"])
        annotation_counts[sample["id"]] = len(boxes)
    report = []
    seen = set()
    for sample in samples:
        key = (sample['session'],sample['slice'])
        if key in seen:
            continue
        seen.add(key)
        clouds = {node:load_cloud(value,cfg['dataset']['bounds']) for node,value in sample['nodes'].items()}
        report.append({'sample':sample['id'],'points_in_bounds':{n:len(p) for n,p in clouds.items()},
                       'boxes_in_bounds':annotation_counts[sample['id']]})
        if not any(len(p) for p in clouds.values()):
            raise ValueError(f'All transformed points outside configured bounds at {sample["id"]}')
    write_json(Path(cfg['output']['root'])/'inspection.json',report)
    print(f'Validated annotations in {len(samples)} frames and point transforms in '
          f'{len(report)} representative slices; see inspection.json')


def inputs(sample, nodes, cfg):
    return [load_cloud(sample['nodes'][n],cfg['dataset']['bounds']) for n in nodes]


def validate_target_centers(boxes, cfg, sample_id):
    """Detect unrepresentable same-class centers without allocating dense targets."""
    classes, bounds, resolution = cfg["classes"], cfg["bounds"], cfg["resolution"]
    slots = cfg.get("objects_per_cell", 1)
    occupied = {}
    for box in boxes:
        ix = int((box.x - bounds[0]) / resolution)
        iy = int((box.y - bounds[1]) / resolution)
        key = classes.index(box.label), iy, ix
        colocated = occupied.setdefault(key, [])
        if len(colocated) >= slots:
            previous = colocated[0]
            raise ValueError(
                f"More than {slots} same-class centers occupy one grid cell during inspection: "
                f"sample={sample_id!r}, class={box.label!r}, cell=(x={ix}, y={iy}), "
                f"resolution={resolution}, first_center=({previous.x:.6f}, {previous.y:.6f}), "
                f"second_center=({box.x:.6f}, {box.y:.6f})"
            )
        colocated.append(box)


def tensor_inputs(clouds, cfg, device, mode):
    bevs = array_inputs(clouds, cfg, mode)
    return device_tensor(bevs, device)


def array_inputs(clouds, cfg, mode):
    if mode == "early":
        return rasterize(np.concatenate(clouds), cfg["dataset"])[None]
    return np.stack([rasterize(p, cfg["dataset"]) for p in clouds])


def device_tensor(array, device):
    value = torch.from_numpy(array)
    if device.type == "cuda":
        value = value.pin_memory()
    return value.to(device, non_blocking=device.type == "cuda")


def training_plan(order, cfg, rng, batch_size=None):
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
        buckets[(model_mode, len(nodes))].append((sample, model_mode, nodes))
    batches = []
    size = batch_size or cfg["training"]["batch_size"]
    for plans in buckets.values():
        batches.extend(
            plans[start : start + size] for start in range(0, len(plans), size)
        )
    rng.shuffle(batches)
    return batches


def prepare_training_example(plan, cfg):
    sample, mode, nodes = plan
    bev = array_inputs(inputs(sample, nodes, cfg), cfg, mode)
    if mode == "early":
        bev = bev[0]
    target = targets(
        annotations(sample["gt"], cfg["dataset"]),
        cfg["dataset"],
        bev.shape[-2:],
        sample_id=sample["id"],
    )
    return bev, target


def submit_training_batch(executor, plans, cfg):
    def prepare_batch():
        examples = [prepare_training_example(plan, cfg) for plan in plans]
        return {
            "bev": np.stack([example[0] for example in examples]),
            "target": tuple(
                np.stack([example[1][index] for example in examples])
                for index in range(3)
            ),
        }

    # Stack the large dense targets on a preprocessing worker instead of
    # blocking the GPU-owning training thread.
    return [executor.submit(prepare_batch)]


def slice_training_batch(batch, start, stop):
    return {
        "bev": batch["bev"][start:stop],
        "target": tuple(value[start:stop] for value in batch["target"]),
    }


def cuda_oom(error, device):
    return device.type == "cuda" and (
        isinstance(error, torch.OutOfMemoryError)
        or "out of memory" in str(error).lower()
    )


def release_cuda_memory(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()


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
        examples = {
            "bev": np.stack([example[0] for example in examples]),
            "target": tuple(
                np.stack([example[1][index] for example in examples])
                for index in range(3)
            ),
        }
    scaler_needs_update = False
    try:
        bev = device_tensor(examples["bev"], device)
        target = tuple(device_tensor(value, device) for value in examples["target"])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=mixed_precision(cfg, device),
        ):
            value = loss(model(bev, mode), target)
        if not torch.isfinite(value):
            raise ValueError("Nonfinite training loss")
        scaler.scale(value).backward()
        scaler.unscale_(optimizer)
        scaler_needs_update = True
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        scaler.step(optimizer)
        scaler.update()
        scaler_needs_update = False
        return [float(value.detach())] * len(examples["bev"])
    except (torch.OutOfMemoryError, RuntimeError) as error:
        if not cuda_oom(error, device):
            raise
        oom_message = str(error)
    optimizer.zero_grad(set_to_none=True)
    if scaler_needs_update:
        # GradScaler records a per-optimizer STEPPED state before invoking the
        # optimizer. Clear that state so an OOM inside optimizer.step does not
        # make the smaller retry fail with "step() has already been called".
        scaler.update()
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


def prepare_validation_example(sample, cfg):
    clouds = inputs(sample, sorted(sample["nodes"]), cfg)
    early = array_inputs(clouds, cfg, "early")[0]
    per_node = array_inputs(clouds, cfg, "intermediate")
    gt = annotations(sample["gt"], cfg["dataset"])
    target = targets(gt, cfg["dataset"], early.shape[-2:], sample_id=sample["id"])
    return early, per_node, gt, target


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


def make_model(cfg,device):
    return BEVDetector(
        cfg["dataset"]["classes"],
        cfg["training"]["hidden_channels"],
        cfg["dataset"].get("objects_per_cell", 1),
    ).to(device)


def mixed_precision(cfg, device):
    return device.type == "cuda" and cfg["training"]["mixed_precision"]


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
    for seed in cfg['training']['seeds']:
        path = output/f'seed_{seed}'/'best.pt'
        latest = path.parent/'latest.pt'
        completion = path.parent/'complete.json'
        if path.exists() and completion.exists() and not overwrite:
            state = torch.load(path,map_location='cpu',weights_only=True)
            if not checkpoint_compatible(state, cfg):
                handle_incompatible_seed(
                    cfg, path.parent, "contains a checkpoint from a different config"
                )
            else:
                completed_epochs = int(
                    json.loads(completion.read_text()).get("epochs", 0)
                )
                if completed_epochs >= cfg["training"]["epochs"]:
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
        scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision(cfg, device))
        best_score, history, start_epoch = -float("inf"), [], 0
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
            order = training.copy()
            rng.shuffle(order)
            train_losses = []
            batches = training_plan(order, cfg, rng, runtime_batches["training"])
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
            metrics_path = path.parent/f'training_metrics_seed_{seed}.csv'
            write_csv(metrics_path, history)
            if cfg["output"]["plots"]:
                plot_training(path.parent / f"training_curves_seed_{seed}.png", history)
            kept_epochs = ",".join(str(item[1]) for item in retained)
            print(
                f'seed={seed} epoch={epoch+1} train={row["train_loss"]:.4f} '
                f'val={val:.4f} mAP={val_map:.4f} precision={row["val_precision"]:.4f} '
                f'recall={val_recall:.4f} F1={row["val_f1"]:.4f} '
                f"score={score:.4f} kept=[{kept_epochs}]",
                flush=True,
            )
            if (epoch + 1) % cfg["training"]["recovery"]["save_every_epochs"] == 0:
                recovery_state = state | {
                    "optimizer": optimizer.state_dict(),
                    "scaler": (
                        scaler.state_dict() if mixed_precision(cfg, device) else None
                    ),
                    "best_score": best_score,
                    "history": history,
                    "python_rng_state": rng.getstate(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": (
                        torch.cuda.get_rng_state(device)
                        if device.type == "cuda"
                        else None
                    ),
                }
                atomic_torch_save(recovery_state, latest)
        best_state = torch.load(path, map_location='cpu', weights_only=True)
        write_json(completion,{'timestamp_utc':utc_now(),'epochs':cfg['training']['epochs'],
                   'best_epoch':best_state['epoch'],'best_validation_loss':best_state['validation_loss'],
                   'best_validation_map':best_state['validation_map'],
                   'best_validation_recall':best_state['validation_recall'],
                   'best_selection_score':best_state['selection_score']})


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


def adaptive_payload(feature_nodes, selected, cfg):
    """Serialize selected collaborator cells as int32 indices and fp16 values."""
    total = 0
    index_dtype = "<u2" if cfg["adaptive_fusion"]["index_bytes"] == 2 else "<u4"
    value_dtype = "<f2" if cfg["adaptive_fusion"]["value_bytes"] == 2 else "<f4"
    for node, mask in zip(feature_nodes[1:], selected):
        device_indices = mask.flatten().nonzero(as_tuple=False).flatten()
        indices = device_indices.detach().cpu().numpy().astype(index_dtype, copy=False)
        values = (
            node.flatten(1)[:, device_indices]
            .T.detach()
            .cpu()
            .numpy()
            .astype(value_dtype, copy=False)
        )
        total += len(zlib.compress(indices.tobytes() + values.tobytes(), level=1))
    return total


def adaptive_selector(strategy):
    base = strategy.split("@", 1)[0]
    return base.split("_", 1)[1] if "_" in base else "learned"


def stable_random_seed(*parts):
    digest = hashlib.sha256("\0".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def infer(model, clouds, cfg, device, strategy):
    base_strategy, _, budget_text = strategy.partition("@")
    bev = tensor_inputs(clouds,cfg,device,strategy)
    sync(device)
    start = time.perf_counter()
    with torch.no_grad():
        features = model.encoder(bev)
        if base_strategy == "intermediate":
            output = model.heads(features.amax(dim=0, keepdim=True))
            boxes = decode(output, cfg)
        elif base_strategy.startswith("adaptive"):
            budget = float(
                budget_text or cfg["adaptive_fusion"]["evaluation_budgets"][0]
            )
            fused, selected = model.adaptive_features(
                features[None], budget, adaptive_selector(strategy)
            )
            boxes = decode(model.heads(fused), cfg)
        elif base_strategy == "late":
            local = [decode(model.heads(f[None]), cfg) for f in features]
            boxes = nms(
                [b for group in local for b in group],
                cfg["evaluation"]["nms_iou"],
                cfg["evaluation"]["late_max_detections"],
            )
        else:
            boxes = decode(model.heads(features),cfg)
    sync(device)
    elapsed_ms = (time.perf_counter() - start) * 1000
    if base_strategy == "intermediate":
        representation = sum(
            feature_payload(f, cfg) for f in features[1:].detach().cpu().numpy()
        )
    elif base_strategy.startswith("adaptive"):
        representation = adaptive_payload(features, selected[0], cfg)
    elif base_strategy == "late":
        representation = sum(
            payload(
                np.array(
                    [
                        list(b.to_array())
                        + [cfg["dataset"]["classes"].index(b.label), b.score]
                        for b in group
                    ],
                    dtype=np.float32,
                ).reshape(-1, 9)
            )
            for group in local[1:]
        )
    else:
        representation = sum(payload(p) for p in clouds[1:])
    return boxes,elapsed_ms,representation


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
    base_strategy, _, budget_text = strategy.partition("@")
    try:
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
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=mixed_precision(cfg, device),
        ):
            if base_strategy == "early":
                outputs = model.heads(model.encoder(bev))
                predictions = [
                    decode(batch_outputs(outputs, i), cfg)
                    for i in range(len(cloud_batches))
                ]
                representations = (
                    raw_payloads
                    if raw_payloads is not None
                    else [
                        sum(payload(p) for p in clouds[1:]) for clouds in cloud_batches
                    ]
                )
            else:
                batch, nodes, channels, height, width = bev.shape
                features = model.encoder(
                    bev.reshape(batch * nodes, channels, height, width)
                ).reshape(batch, nodes, -1, height, width)
                if base_strategy == "intermediate":
                    outputs = model.heads(features.amax(dim=1))
                    predictions = [
                        decode(batch_outputs(outputs, i), cfg) for i in range(batch)
                    ]
                    serialization_start = time.perf_counter()
                    representations = [
                        sum(
                            feature_payload(feature, cfg)
                            for feature in item[1:].detach().cpu().numpy()
                        )
                        for item in features
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
                    fused, selected = model.adaptive_features(
                        features, budget, selector, generators=generators
                    )
                    outputs = model.heads(fused)
                    predictions = [
                        decode(batch_outputs(outputs, i), cfg) for i in range(batch)
                    ]
                    serialization_start = time.perf_counter()
                    representations = [
                        adaptive_payload(features[item], selected[item], cfg)
                        for item in range(batch)
                    ]
                    serialization_ms = (
                        (time.perf_counter() - serialization_start)
                        * 1000
                        / len(cloud_batches)
                    )
                else:
                    outputs = model.heads(
                        features.reshape(batch * nodes, *features.shape[2:])
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
                            sum(
                                payload(
                                    np.array(
                                        [
                                            list(box.to_array())
                                            + [
                                                cfg["dataset"]["classes"].index(
                                                    box.label
                                                ),
                                                box.score,
                                            ]
                                            for box in group
                                        ],
                                        dtype=np.float32,
                                    ).reshape(-1, 9)
                                )
                                for group in local[1:]
                            )
                        )
                        serialization_seconds += (
                            time.perf_counter() - serialization_start
                        )
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


def network_rows(cfg, base, raw_bytes, representation_bytes):
    net = cfg["deployment_assumptions"]
    base_strategy = base["strategy"].split("@", 1)[0]
    if base_strategy.startswith("adaptive"):
        base_strategy = "adaptive"
    rows = []
    for placement in ("edge", "cloud"):
        transmitted = representation_bytes if placement == "edge" else raw_bytes
        for bandwidth in net["bandwidths_mbps"]:
            transmit_ms = transmitted * 8 / (bandwidth * 1e6) * 1000
            total = (
                net["sensing_ms"]
                + net[placement + "_rtt_ms"]
                + net["processing_ms"][placement][base_strategy]
                + transmit_ms
            )
            rows.append(
                base
                | {
                    "placement": placement,
                    "bandwidth_mbps": bandwidth,
                    "assumption_source": net["source"],
                    "payload_bytes": transmitted,
                    "transmission_ms_model": transmit_ms,
                    "end_to_end_ms_model": total,
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


def write_latency_analysis(cfg, traces, output):
    write_csv(output / "latency_trace.csv", traces)
    grouped = defaultdict(list)
    keys = ("seed", "session", "weather", "nodes", "node_count", "strategy")
    for row in traces:
        grouped[tuple(row[key] for key in keys)].append(row)
    summaries = []
    net, experiment = cfg["deployment_assumptions"], cfg["latency_experiment"]
    for key, rows in sorted(grouped.items()):
        for placement in ("edge", "cloud"):
            for bandwidth in net["bandwidths_mbps"]:
                service = []
                for row in rows:
                    payload_bytes = (
                        float(row["representation_bytes"])
                        if placement == "edge"
                        else float(row["raw_bytes"])
                    )
                    tx = payload_bytes * 8 / (bandwidth * 1e6) * 1000
                    serialization = (
                        float(row["serialization_ms"])
                        if placement == "edge"
                        else float(row["raw_serialization_ms"])
                    )
                    # Replay measured receiver compute rather than a configured
                    # constant. Loading and sender preprocessing remain outside
                    # the shared receiver queue.
                    service.append(
                        float(row["inference_decode_ms"]) + serialization + tx
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
                            dict(zip(keys, key))
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
                                "deadline_ms": deadline,
                                **result,
                            }
                        )
    write_csv(output / "latency_tail_summary.csv", summaries)


def run_evaluation(cfg):
    samples = [s for s in indexed(cfg) if s['split']=='test']
    sessions = defaultdict(list)
    for sample in samples:
        sessions[sample['session']].append(sample)
    output, device = Path(cfg['output']['root']),device_for(cfg)
    all_metrics, all_costs, all_traces = [], [], []
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
        warmed = set()
        for session, frames in sessions.items():
            part_path = output / "eval_parts" / f"seed_{seed}" / f"{session}.json"
            if part_path.exists():
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
                        all_traces.extend(part.get("traces", []))
                        print(
                            f"Resumed evaluation seed={seed} session={session}",
                            flush=True,
                        )
                        continue
            metrics_start, costs_start, traces_start = (
                len(all_metrics),
                len(all_costs),
                len(all_traces),
            )
            common = set.intersection(*(set(f['nodes']) for f in frames))
            groups = choose_groups(common,cfg['evaluation'])
            buckets = defaultdict(list)
            costs = defaultdict(list)
            batch_size = cfg["latency_experiment"]["profile_batch_size"]
            for start in range(0, len(frames), batch_size):
                frame_chunk = frames[start : start + batch_size]
                load_start = time.perf_counter()
                prepared = []
                for sample in frame_chunk:
                    cloud_by_node = {
                        n: load_cloud(sample["nodes"][n], cfg["dataset"]["bounds"])
                        for n in common
                    }
                    prepared.append(
                        (
                            annotations(sample["gt"], cfg["dataset"]),
                            cloud_by_node,
                            None,
                            None,
                            None,
                        )
                    )
                load_ms = (time.perf_counter() - load_start) * 1000 / len(frame_chunk)
                raw_serialization_start = time.perf_counter()
                prepared = [
                    (
                        gt,
                        cloud_by_node,
                        {node: payload(cloud) for node, cloud in cloud_by_node.items()},
                        None,
                        None,
                    )
                    for gt, cloud_by_node, _, _, _ in prepared
                ]
                raw_serialization_ms = (
                    (time.perf_counter() - raw_serialization_start)
                    * 1000
                    / len(frame_chunk)
                )
                cached = []
                for gt, cloud_by_node, raw_by_node, _, _ in prepared:
                    bev_by_node, bev_ms_by_node = {}, {}
                    for node, cloud in cloud_by_node.items():
                        raster_start = time.perf_counter()
                        bev_by_node[node] = rasterize(cloud, cfg["dataset"])
                        bev_ms_by_node[node] = (
                            time.perf_counter() - raster_start
                        ) * 1000
                    cached.append(
                        (gt, cloud_by_node, raw_by_node, bev_by_node, bev_ms_by_node)
                    )
                prepared = cached
                for nodes in groups:
                    cloud_batches = [
                        [cloud_by_node[n] for n in nodes]
                        for _, cloud_by_node, _, _, _ in prepared
                    ]
                    raw_batch = [
                        sum(raw_by_node[node] for node in nodes[1:])
                        for _, _, raw_by_node, _, _ in prepared
                    ]
                    strategies = []
                    for configured in cfg["evaluation"]["strategies"]:
                        if configured == "adaptive":
                            strategies.extend(
                                f"adaptive@{budget:g}"
                                for budget in cfg["adaptive_fusion"][
                                    "evaluation_budgets"
                                ]
                            )
                            strategies.extend(
                                f"adaptive_{selector}@{budget:g}"
                                for selector in cfg["adaptive_fusion"]["ablations"]
                                for budget in cfg["adaptive_fusion"][
                                    "evaluation_budgets"
                                ]
                            )
                        else:
                            strategies.append(configured)
                    for strategy in strategies:
                        # Receiver rotation matters only to asymmetric adaptive
                        # fusion. Avoid repeating receiver-invariant baselines.
                        if not strategy.startswith("adaptive") and tuple(
                            nodes
                        ) != tuple(sorted(nodes)):
                            continue
                        key = (nodes, strategy)
                        random_seeds = [
                            stable_random_seed(seed, sample["id"], *nodes, strategy)
                            for sample in frame_chunk
                        ]
                        prepared_bevs = None
                        preprocessing_overheads = None
                        if not strategy.startswith("early"):
                            prepared_bevs = [
                                np.stack([bev_by_node[node] for node in nodes])
                                for _, _, _, bev_by_node, _ in prepared
                            ]
                            preprocessing_overheads = [
                                sum(bev_ms_by_node[node] for node in nodes)
                                for _, _, _, _, bev_ms_by_node in prepared
                            ]
                        warmup_key = (strategy, len(nodes), len(cloud_batches))
                        if warmup_key not in warmed:
                            infer_batch(
                                model,
                                cloud_batches[:1],
                                cfg,
                                device,
                                strategy,
                                raw_batch[:1],
                                random_seeds=random_seeds[:1],
                                prepared_bevs=(
                                    prepared_bevs[:1]
                                    if prepared_bevs is not None
                                    else None
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
                            raw_batch,
                            random_seeds=random_seeds,
                            prepared_bevs=prepared_bevs,
                            preprocessing_overheads_ms=preprocessing_overheads,
                        )
                        for sample, (gt, _, _, _, _), raw_bytes, result in zip(
                            frame_chunk, prepared, raw_batch, results
                        ):
                            (
                                preds,
                                elapsed,
                                representation,
                                preprocessing,
                                serialization,
                            ) = result
                            buckets[key].append((preds, gt))
                            costs[key].append(
                                (
                                    elapsed,
                                    raw_bytes,
                                    representation,
                                    preprocessing,
                                    serialization,
                                )
                            )
                            all_traces.append(
                                {
                                    "seed": seed,
                                    "session": session,
                                    "weather": sample["weather"],
                                    "sample": sample["id"],
                                    "nodes": ",".join(nodes),
                                    "node_count": len(nodes),
                                    "strategy": strategy,
                                    "load_ms": load_ms,
                                    "preprocessing_ms": preprocessing,
                                    "inference_decode_ms": elapsed,
                                    "serialization_ms": serialization,
                                    "raw_serialization_ms": raw_serialization_ms,
                                    "local_pipeline_ms": load_ms
                                    + preprocessing
                                    + elapsed
                                    + serialization,
                                    "raw_bytes": raw_bytes,
                                    "representation_bytes": representation,
                                }
                            )
            for (nodes, strategy), values in buckets.items():
                base = {
                    "seed": seed,
                    "session": session,
                    "weather": frames[0]["weather"],
                    "nodes": ",".join(nodes),
                    "node_count": len(nodes),
                    "strategy": strategy,
                }
                all_metrics.extend(
                    base | row
                    for row in evaluate(
                        values,
                        cfg["dataset"]["classes"],
                        cfg["evaluation"]["iou_thresholds"],
                    )
                )
                measured = np.array(costs[(nodes, strategy)])
                cost_base = base | {
                    "frames": len(values),
                    "inference_decode_ms_mean": float(measured[:, 0].mean()),
                    "inference_decode_ms_p95": float(np.quantile(measured[:, 0], 0.95)),
                    "preprocessing_ms_mean": float(measured[:, 3].mean()),
                    "serialization_ms_mean": float(measured[:, 4].mean()),
                }
                all_costs.extend(
                    network_rows(
                        cfg,
                        cost_base,
                        float(measured[:, 1].mean()),
                        float(measured[:, 2].mean()),
                    )
                )
            print(
                f"Evaluated seed={seed} session={session}, frames={len(frames)}, subsets={len(groups)}",
                flush=True,
            )
            write_json(
                part_path,
                {
                    "signature": evaluation_signature,
                    "seed": seed,
                    "session": session,
                    "metrics": all_metrics[metrics_start:],
                    "costs": all_costs[costs_start:],
                    "traces": all_traces[traces_start:],
                },
            )
    write_csv(output / "accuracy.csv", all_metrics)
    write_csv(output / "latency_communication.csv", all_costs)
    write_latency_analysis(cfg, all_traces, output)
    return all_metrics


def report(cfg):
    output = Path(cfg['output']['root'])
    with (output/'accuracy.csv').open() as stream:
        accuracy = list(csv.DictReader(stream))
    with (output/'latency_communication.csv').open() as stream:
        costs = list(csv.DictReader(stream))

    def number(row, key):
        return float(row[key]) if row[key] not in ('', None) else None

    metric_names = ('ap_3d','precision','recall','f1','mean_matched_iou_3d',
                    'mean_center_error_m','mean_yaw_error_deg','mean_size_iou')
    dimensions = ('weather','node_count','strategy','class','iou')
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
    overview_dimensions = ('weather','node_count','strategy','iou')
    for row in accuracy:
        if number(row, 'ap_3d') is not None:
            overview_groups[tuple(row[k] for k in overview_dimensions)].append(row)
    overview = []
    for key, rows in sorted(overview_groups.items()):
        tp = sum(int(row['true_positives']) for row in rows)
        fp = sum(int(row['false_positives']) for row in rows)
        fn = sum(int(row['false_negatives']) for row in rows)
        macro = lambda name: float(np.mean([number(row,name) for row in rows]))
        micro_precision = tp/(tp+fp) if tp+fp else 0.0
        micro_recall = tp/(tp+fn) if tp+fn else 0.0
        micro_f1 = (2*micro_precision*micro_recall/(micro_precision+micro_recall)
                    if micro_precision+micro_recall else 0.0)
        matched = [row for row in rows if number(row,'mean_matched_iou_3d') is not None]
        matched_count = sum(int(row['true_positives']) for row in matched)
        weighted = lambda name: (sum(number(row,name)*int(row['true_positives']) for row in matched)/matched_count
                                 if matched_count else None)
        overview.append(dict(zip(overview_dimensions,key)) | {
            'map_3d_macro':macro('ap_3d'),'precision_macro':macro('precision'),
            'recall_macro':macro('recall'),'f1_macro':macro('f1'),
            'precision_micro':micro_precision,'recall_micro':micro_recall,'f1_micro':micro_f1,
            'true_positives':tp,'false_positives':fp,'false_negatives':fn,
            'mean_matched_iou_3d':weighted('mean_matched_iou_3d'),
            'mean_center_error_m':weighted('mean_center_error_m'),
            'mean_yaw_error_deg':weighted('mean_yaw_error_deg'),
            'mean_size_iou':weighted('mean_size_iou'),'class_observations':len(rows)})
    write_csv(output/'metrics_overview.csv',overview)

    comparison_dimensions = ("seed", "session", "weather", "strategy", "class", "iou")
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
        key = tuple(row[k] for k in ('weather','node_count','strategy','class','iou'))
        weather_groups[key].append(row)
    weather_summary = []
    for key, values in sorted(weather_groups.items()):
        ap = [v['ap_gain_over_mean_single'] for v in values]
        recall = [v['recall_gain_over_mean_single'] for v in values]
        weather_summary.append(dict(zip(dimensions,key)) | {
            'mean_ap_gain':float(np.mean(ap)), 'std_ap_gain':float(np.std(ap)),
            'ap_positive_rate':float(np.mean(np.asarray(ap)>0)),
            'mean_recall_gain':float(np.mean(recall)), 'std_recall_gain':float(np.std(recall)),
            'recall_positive_rate':float(np.mean(np.asarray(recall)>0)),
            'paired_observations':len(values)})
    write_csv(output/'weather_cooperation_summary.csv',weather_summary)

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
            scaling.append(dict(zip(('seed','session','weather','strategy','class','iou'),key)) | {
                'from_node_count':previous, 'to_node_count':current,
                'ap_change':current_ap-previous_ap,
                'recall_change':current_recall-previous_recall})
    write_csv(output/'node_scaling.csv',scaling)

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
                    "class": row["class"],
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
                row[k] for k in ("weather", "node_count", "class", "iou", "strategy")
            ) + (selector,)
            adaptive_groups[key].append(
                (float(row["ap_3d"]), float(row["payload_bytes"]))
            )
    adaptive_rate = []
    for key, values in sorted(adaptive_groups.items()):
        weather, node_count, label, iou, strategy, selector = key
        adaptive_rate.append(
            {
                "weather": weather,
                "node_count": node_count,
                "class": label,
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

    auc_groups = defaultdict(list)
    for row in adaptive_rate:
        key = tuple(
            row[k] for k in ("weather", "node_count", "class", "iou", "selector")
        )
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
            dict(zip(("weather", "node_count", "class", "iou", "selector"), key))
            | {
                "normalized_log_budget_ap_auc": normalized_auc,
                "minimum_budget": float(x[0]),
                "maximum_budget": float(x[-1]),
                "operating_points": len(points),
            }
        )
    write_csv(output / "adaptive_rate_accuracy_auc.csv", adaptive_auc)

    if cfg['output']['plots'] and summary:
        try:
            import matplotlib
        except ImportError:
            print('Report plot skipped: install matplotlib from requirements.txt', flush=True)
            matplotlib = None
        if matplotlib is not None:
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            primary_iou = min(float(row["iou"]) for row in summary)
            adaptive_primary = (
                f'adaptive@{max(cfg["adaptive_fusion"]["evaluation_budgets"]):g}'
            )
            plotted_strategies = {"early", "intermediate", "late", adaptive_primary}
            grouped = defaultdict(list)
            for row in summary:
                if (
                    float(row["iou"]) == primary_iou
                    and row["strategy"] in plotted_strategies
                ):
                    label = (
                        "adaptive"
                        if row["strategy"] == adaptive_primary
                        else row["strategy"]
                    )
                    grouped[label].append(
                        (int(row["node_count"]), float(row["mean_ap"]))
                    )
            figure, axis = plt.subplots(figsize=(10, 6))
            for strategy, values in sorted(grouped.items()):
                by_count = defaultdict(list)
                for count, value in values:
                    by_count[count].append(value)
                x = sorted(by_count)
                y = [float(np.mean(by_count[count])) for count in x]
                axis.plot(x, y, marker="o", label=strategy)
            axis.set(
                xlabel="node count",
                ylabel="macro mean AP",
                title=f"Test AP by node count and fusion (IoU={primary_iou:g})",
            )
            axis.set_ylim(0, 1)
            axis.legend(fontsize=8, ncol=2)
            axis.grid(alpha=.25)
            figure.tight_layout()
            figure.savefig(output/'accuracy_by_nodes.png', dpi=160)
            plt.close(figure)
            if weather_summary:
                figure, axis = plt.subplots(figsize=(10, 6))
                grouped = defaultdict(list)
                for row in weather_summary:
                    grouped[(row['weather'], row['strategy'])].append(float(row['mean_ap_gain']))
                labels, values = [], []
                for key, group in sorted(grouped.items()):
                    labels.append('/'.join(key))
                    values.append(float(np.mean(group)))
                axis.bar(labels, values)
                axis.axhline(0, color='black', linewidth=.8)
                axis.set(ylabel='mean paired AP gain', title='Cooperation gain by weather and fusion')
                axis.tick_params(axis='x', rotation=45)
                figure.tight_layout()
                figure.savefig(output/'cooperation_gain_by_weather.png', dpi=160)
                plt.close(figure)

            figure, axis = plt.subplots(figsize=(9, 5))
            latency_groups = defaultdict(list)
            for row in costs:
                strategy = row["strategy"].split("@", 1)[0].split("_", 1)[0]
                latency_groups[
                    (row["placement"], strategy, float(row["bandwidth_mbps"]))
                ].append(float(row["end_to_end_ms_model"]))
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
                ylabel="mean assumed end-to-end latency (ms)",
                title="Analytical latency by placement and bandwidth",
                xscale="log",
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
        "Accuracy: rotated 3D, globally confidence-ranked AP within each session/subset/seed.",
        "Precision, recall, and F1 use the configured score threshold after class-wise NMS.",
        "Matched-object diagnostics include 3D IoU, center error, yaw error, and dimension IoU.",
        "All aggregates are descriptive means, not confidence intervals or causal weather estimates.",
        "Single-node rows cover every common real node. The fixed global ROI/GT is unchanged across counts.",
        "Measured inference+decode uses batched node encoding and includes decode/NMS; excludes loading/rasterization/serialization.",
        "End-to-end latency is an analytical estimate using configured literature-derived assumptions.",
        "Dense payloads use float32 and zlib level 1; adaptive sparse messages use configured index/value widths and zlib level 1.",
        (
            "Adaptive fusion rotates every participating node as the dense local reference and budgets collaborator cells only."
            if cfg["evaluation"].get("receiver_policy") == "all"
            else "Adaptive fusion uses the first sorted participating node as the dense local reference and budgets collaborator cells only."
        ),
        "Cloud sends raw cropped points; edge sends raw points/features/local predictions according to strategy.",
        f'Assumption source: {cfg["deployment_assumptions"]["source"]}.',
        weather_limit,
        "cooperation_gains.csv uses paired comparisons within session/seed/class/strategy/IoU.",
        "",
        "## Cooperation gain by weather",
        "",
        "| Weather | Nodes | Fusion | Class | IoU | AP gain | Recall gain | AP positive rate |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in weather_summary:
        lines.append(f'| {row["weather"]} | {row["node_count"]} | {row["strategy"]} | '
                     f'{row["class"]} | {row["iou"]} | {row["mean_ap_gain"]:.4f} | '
                     f'{row["mean_recall_gain"]:.4f} | {row["ap_positive_rate"]:.2f} |')
    lines.extend(['', '## Absolute accuracy', '',
                  '| Weather | Nodes | Fusion | Class | IoU | Mean AP | Precision | Recall | F1 | Center error (m) |',
                  '|---|---:|---|---|---:|---:|---:|---:|---:|---:|'])
    for row in summary:
        center = ('n/a' if row['mean_center_error_m'] is None else
                  f'{row["mean_center_error_m"]:.3f}')
        lines.append(f'| {row["weather"]} | {row["node_count"]} | {row["strategy"]} | '
                     f'{row["class"]} | {row["iou"]} | {row["mean_ap"]:.4f} | '
                     f'{row["mean_precision"]:.4f} | {row["mean_recall"]:.4f} | '
                     f'{row["mean_f1"]:.4f} | {center} |')
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print(f'Report: {output / "report.md"}')


def verify_results(cfg):
    """Fail the unattended run if required artifacts or metric invariants are invalid."""
    output = Path(cfg["output"]["root"])
    required = [
        "index.json",
        "inventory.json",
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
    for seed in cfg['training']['seeds']:
        for name in ('best.pt','latest.pt','complete.json',f'training_metrics_seed_{seed}.csv'):
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
    bounded = ('ap_3d','precision','recall','f1','mean_matched_iou_3d','mean_size_iou')
    for line, row in enumerate(metric_rows, start=2):
        for name in bounded:
            value = row.get(name)
            if value not in ('',None) and (not math.isfinite(float(value)) or not 0 <= float(value) <= 1):
                errors.append(f'accuracy.csv line {line}: invalid {name}={value}')
        tp, fp, fn = (int(row[name]) for name in ('true_positives','false_positives','false_negatives'))
        if tp+fp != int(row['predictions']) or tp+fn != int(row['ground_truth']):
            errors.append(f'accuracy.csv line {line}: TP/FP/FN totals are inconsistent')
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
