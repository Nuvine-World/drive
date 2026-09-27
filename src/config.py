"""Validated configuration with project-relative paths."""
import math
import os
from pathlib import Path
import yaml

PROJECT = Path(__file__).resolve().parents[1]


def resolve(path):
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (PROJECT / p).resolve()


def enable_smoke(cfg):
    """Apply a tiny, still split-disjoint, end-to-end experiment profile."""
    smoke = cfg['dataset']['smoke']
    smoke['enabled'] = True
    cfg['dataset']['splits'] = smoke['splits']
    cfg['training']['epochs'] = smoke['epochs']
    cfg['training']['seeds'] = cfg['training']['seeds'][:smoke['seeds']]
    # Keep the profile tiny while still exercising the cooperative code path.
    cfg["training"]["node_counts"] = [1, 2]
    cfg["evaluation"]["node_counts"] = [1, 2]
    output = Path(cfg["output"]["root"])
    if output.name != "smoke":
        cfg["output"]["root"] = str(output / "smoke")
    return cfg


def load_config(path):
    with open(path) as stream:
        cfg = yaml.safe_load(stream)
    d, t, e = cfg['dataset'], cfg['training'], cfg['evaluation']
    dataset_type = d.setdefault('type', 'coinfra')
    if dataset_type not in ('coinfra', 'advercity'):
        raise ValueError('dataset.type must be coinfra or advercity')
    root_environment = 'ADVERCITY_ROOT' if dataset_type == 'advercity' else 'COINFRA_ROOT'
    d['root'] = str(resolve(os.environ.get(root_environment, d['root'])))
    if dataset_type == 'advercity':
        acquisition = d.setdefault('acquisition', {})
        acquisition['work_dir'] = str(resolve(os.environ.get(
            'ADVERCITY_WORK_DIR', acquisition.get('work_dir', 'advercity_work'))))
        acquisition.setdefault('minimum_free_gb', 35)
        acquisition.setdefault('cleanup_archives', True)
        acquisition.setdefault('retain_frame_stride', 1)
        if acquisition['minimum_free_gb'] < 1 or acquisition['retain_frame_stride'] < 1:
            raise ValueError('Invalid Adver-City acquisition settings')
    cfg['output']['root'] = str(resolve(cfg['output']['root']))
    cfg['output'].setdefault('plots', True)
    smoke = d.setdefault('smoke', {'enabled': False})
    smoke.setdefault('enabled', False)
    smoke.setdefault('splits', d['splits'])
    smoke.setdefault('sessions_per_split', 1)
    smoke.setdefault('slices_per_session', 1)
    smoke.setdefault('frames_per_slice', 1)
    smoke.setdefault('epochs', 2)
    smoke.setdefault('seeds', 1)
    if smoke['enabled']:
        enable_smoke(cfg)
    b = d['bounds']
    if len(b) != 6 or not all(math.isfinite(v) for v in b) or any(b[i] >= b[i+3] for i in range(3)):
        raise ValueError('Invalid dataset bounds')
    if not math.isfinite(d['resolution']) or d['resolution'] <= 0:
        raise ValueError('resolution must be positive')
    if d['yaw_unit'] not in ('degrees', 'radians'):
        raise ValueError('yaw_unit must be degrees or radians')
    if d['frame_stride'] < 1 or d.get('max_timestamp_delta_s', 0) < 0:
        raise ValueError('Invalid stride or synchronization tolerance')
    if any(smoke[k] < 1 for k in ('sessions_per_split','slices_per_session','frames_per_slice','epochs','seeds')):
        raise ValueError('Smoke-mode limits must be positive')
    if not d['classes'] or len(d['classes']) != len(set(d['classes'])):
        raise ValueError('Classes must be nonempty and unique')
    seen = set()
    for split in ('train', 'val', 'test'):
        sessions = d['splits'][split]
        if not sessions or seen.intersection(sessions):
            raise ValueError('Splits must be nonempty and recording sessions must not overlap')
        seen.update(sessions)
    if not e["strategies"] or not set(e["strategies"]) <= {
        "early",
        "intermediate",
        "late",
    }:
        raise ValueError("Unknown fusion strategy")
    if not t["modes"] or not set(t["modes"]) <= {"early", "intermediate", "single"}:
        raise ValueError("Unknown training mode")
    if t["epochs"] < 1 or not t["seeds"] or t["learning_rate"] <= 0:
        raise ValueError("Invalid training configuration")
    t.setdefault("batch_size", 1)
    t.setdefault("preprocessing_workers", 1)
    t.setdefault("mixed_precision", False)
    t.setdefault("require_cuda", False)
    if (
        not isinstance(t["batch_size"], int)
        or isinstance(t["batch_size"], bool)
        or t["batch_size"] < 1
    ):
        raise ValueError("training.batch_size must be positive")
    if (
        not isinstance(t["preprocessing_workers"], int)
        or isinstance(t["preprocessing_workers"], bool)
        or t["preprocessing_workers"] < 1
    ):
        raise ValueError("training.preprocessing_workers must be positive")
    if not isinstance(t["mixed_precision"], bool):
        raise ValueError("training.mixed_precision must be boolean")
    if not isinstance(t["require_cuda"], bool):
        raise ValueError("training.require_cuda must be boolean")
    checkpointing = t.setdefault("checkpointing", {})
    if not isinstance(checkpointing, dict):
        raise ValueError('training.checkpointing must be a mapping')
    checkpointing.setdefault('keep_best', 3)
    checkpointing.setdefault('weights', {'val_loss_quality': .34, 'val_map': .33,
                                          'val_recall': .33})
    if (not isinstance(checkpointing['keep_best'], int) or
            isinstance(checkpointing['keep_best'], bool) or checkpointing['keep_best'] < 1):
        raise ValueError('training.checkpointing.keep_best must be positive')
    weights = checkpointing['weights']
    expected = {'val_loss_quality','val_map','val_recall'}
    if (not isinstance(weights, dict) or set(weights) != expected or
            any(not isinstance(v, (int,float)) or not math.isfinite(v) or v < 0
                for v in weights.values()) or sum(weights.values()) <= 0):
        raise ValueError(f'Checkpoint weights must be nonnegative and contain {sorted(expected)}')
    recovery = t.setdefault('recovery', {})
    defaults = {'resume': True, 'save_every_epochs': 1, 'wait_for_resources': True,
                'poll_seconds': 60, 'timeout_minutes': 0,
                'minimum_free_disk_gb': 5, 'minimum_free_gpu_gb': 2,
                'cuda_oom_retries': 3}
    for key, value in defaults.items():
        recovery.setdefault(key, value)
    if (
        recovery["save_every_epochs"] < 1
        or recovery["poll_seconds"] < 1
        or recovery["timeout_minutes"] < 0
    ):
        raise ValueError("Invalid training recovery interval")
    if (
        recovery["minimum_free_disk_gb"] < 0
        or recovery["minimum_free_gpu_gb"] < 0
        or recovery["cuda_oom_retries"] < 0
    ):
        raise ValueError("Recovery space and retry values must be nonnegative")
    if any(n < 1 for n in e["node_counts"] + t["node_counts"]):
        raise ValueError("Node counts must be positive")
    if e["subsets_per_count"] < 1 or e["max_detections"] < 1:
        raise ValueError("Invalid evaluation limits")
    e.setdefault("batch_size", t["batch_size"])
    if (
        not isinstance(e["batch_size"], int)
        or isinstance(e["batch_size"], bool)
        or e["batch_size"] < 1
    ):
        raise ValueError("evaluation.batch_size must be positive")
    if any(not 0 < x <= 1 for x in e["iou_thresholds"]):
        raise ValueError("Invalid IoU threshold")
    if not 0 <= e["score_threshold"] <= 1 or not 0 < e["nms_iou"] <= 1:
        raise ValueError("Invalid detection thresholds")
    n = cfg["deployment_assumptions"]
    if not isinstance(n.get("source"), str) or not n["source"].strip():
        raise ValueError(
            "deployment_assumptions.source must identify the assumption source"
        )
    if any(not math.isfinite(x) or x <= 0 for x in n["bandwidths_mbps"]):
        raise ValueError("Bandwidth must be positive")
    for p in ("edge", "cloud"):
        if n[p + "_rtt_ms"] < 0 or any(
            n["processing_ms"][p][s] < 0 for s in e["strategies"]
        ):
            raise ValueError("Latency cannot be negative")
    return cfg
