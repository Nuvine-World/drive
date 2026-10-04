"""Validated configuration with project-relative paths."""
import math
import os
from pathlib import Path
import yaml

PROJECT = Path(__file__).resolve().parents[1]


def validate_splits(splits):
    seen = set()
    for split in ("train", "val", "test"):
        sessions = splits[split]
        if not sessions or seen.intersection(sessions):
            raise ValueError(
                "Splits must be nonempty and recording sessions must not overlap"
            )
        seen.update(sessions)


def resolve(path):
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (PROJECT / p).resolve()


def enable_smoke(cfg):
    """Apply a tiny, still split-disjoint, end-to-end experiment profile."""
    smoke = cfg["dataset"]["smoke"]
    smoke["enabled"] = True
    cfg["dataset"]["splits"] = smoke["splits"]
    validate_splits(cfg["dataset"]["splits"])
    cfg["training"]["epochs"] = smoke["epochs"]
    cfg["training"]["minimum_epochs"] = smoke["epochs"]
    cfg["training"]["early_stopping_patience"] = 0
    cfg["training"].setdefault("scheduler", {})["enabled"] = False
    cfg["training"]["seeds"] = cfg["training"]["seeds"][: smoke["seeds"]]
    # Keep the profile tiny while still exercising the cooperative code path.
    cfg["training"]["node_counts"] = [1, 2]
    cfg["evaluation"]["node_counts"] = [1, 2]
    output = Path(cfg["output"]["root"])
    if output.name != "smoke":
        cfg["output"]["root"] = str(output / "smoke")
    return cfg


def enable_overfit(cfg, frames=32, epochs=100):
    """Enable an intentionally leaky tiny-set profile for learning diagnostics.

    This profile is never selected from a YAML file.  It must be requested
    explicitly on the command line and writes beneath a separate output root.
    """
    d, t, e = cfg["dataset"], cfg["training"], cfg["evaluation"]
    training_session = d["splits"]["train"][0]
    d["splits"]["train"] = [training_session]
    d["splits"]["val"] = [training_session]
    d["smoke"].update(
        {
            "enabled": True,
            "sessions_per_split": 1,
            "slices_per_session": 1,
            "frames_per_slice": frames,
        }
    )
    t.update(
        {
            "seeds": t["seeds"][:1],
            "epochs": epochs,
            "minimum_epochs": epochs,
            "early_stopping_patience": 0,
            "modes": ["single"],
            "node_counts": [1],
        }
    )
    t.setdefault("scheduler", {})["enabled"] = False
    e.update(
        {
            "strategies": ["early"],
            "node_counts": [1],
            "subsets_per_count": 1,
            "iou_thresholds": [0.5],
        }
    )
    cfg["diagnostic_overfit"] = True
    output = Path(cfg["output"]["root"])
    if output.name != "overfit":
        cfg["output"]["root"] = str(output / "overfit")
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
    b = d["bounds"]
    if (
        len(b) != 6
        or not all(math.isfinite(v) for v in b)
        or any(b[i] >= b[i + 3] for i in range(3))
    ):
        raise ValueError("Invalid dataset bounds")
    if not math.isfinite(d["resolution"]) or d["resolution"] <= 0:
        raise ValueError("resolution must be positive")
    d.setdefault("objects_per_cell", 1)
    if (
        not isinstance(d["objects_per_cell"], int)
        or isinstance(d["objects_per_cell"], bool)
        or d["objects_per_cell"] < 1
    ):
        raise ValueError("objects_per_cell must be a positive integer")
    if d["yaw_unit"] not in ("degrees", "radians"):
        raise ValueError("yaw_unit must be degrees or radians")
    if d["frame_stride"] < 1 or d.get("max_timestamp_delta_s", 0) < 0:
        raise ValueError("Invalid stride or synchronization tolerance")
    if any(
        smoke[k] < 1
        for k in (
            "sessions_per_split",
            "slices_per_session",
            "frames_per_slice",
            "epochs",
            "seeds",
        )
    ):
        raise ValueError("Smoke-mode limits must be positive")
    if not d["classes"] or len(d["classes"]) != len(set(d["classes"])):
        raise ValueError("Classes must be nonempty and unique")
    validate_splits(d["splits"])
    if not e["strategies"] or not set(e["strategies"]) <= {
        "early",
        "intermediate",
        "late",
        "adaptive",
    }:
        raise ValueError("Unknown fusion strategy")
    if not t["modes"] or not set(t["modes"]) <= {
        "early",
        "intermediate",
        "adaptive",
        "single",
    }:
        raise ValueError("Unknown training mode")
    if t["epochs"] < 1 or not t["seeds"] or t["learning_rate"] <= 0:
        raise ValueError("Invalid training configuration")
    t.setdefault("minimum_epochs", t["epochs"])
    t.setdefault("early_stopping_patience", 0)
    t.setdefault("early_stopping_min_delta", 0.0)
    scheduler = t.setdefault("scheduler", {})
    if not isinstance(scheduler, dict):
        raise ValueError("training.scheduler must be a mapping")
    scheduler.setdefault("enabled", False)
    scheduler.setdefault("factor", 0.5)
    scheduler.setdefault("patience", 5)
    scheduler.setdefault("minimum_learning_rate", 1e-5)
    if (
        not isinstance(t["minimum_epochs"], int)
        or isinstance(t["minimum_epochs"], bool)
        or not 1 <= t["minimum_epochs"] <= t["epochs"]
    ):
        raise ValueError(
            "training.minimum_epochs must be between 1 and training.epochs"
        )
    if (
        not isinstance(t["early_stopping_patience"], int)
        or isinstance(t["early_stopping_patience"], bool)
        or t["early_stopping_patience"] < 0
        or not isinstance(t["early_stopping_min_delta"], (int, float))
        or t["early_stopping_min_delta"] < 0
    ):
        raise ValueError("Invalid training early-stopping configuration")
    if (
        not isinstance(scheduler["enabled"], bool)
        or not 0 < scheduler["factor"] < 1
        or not isinstance(scheduler["patience"], int)
        or isinstance(scheduler["patience"], bool)
        or scheduler["patience"] < 1
        or not 0 < scheduler["minimum_learning_rate"] <= t["learning_rate"]
    ):
        raise ValueError("Invalid training scheduler configuration")
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
        raise ValueError("training.checkpointing must be a mapping")
    checkpointing.setdefault("keep_best", 3)
    checkpointing.setdefault(
        "weights", {"val_loss_quality": 0.34, "val_map": 0.33, "val_recall": 0.33}
    )
    if (
        not isinstance(checkpointing["keep_best"], int)
        or isinstance(checkpointing["keep_best"], bool)
        or checkpointing["keep_best"] < 1
    ):
        raise ValueError("training.checkpointing.keep_best must be positive")
    weights = checkpointing["weights"]
    expected = {"val_loss_quality", "val_map", "val_recall"}
    if (
        not isinstance(weights, dict)
        or set(weights) != expected
        or any(
            not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
            for v in weights.values()
        )
        or sum(weights.values()) <= 0
    ):
        raise ValueError(
            f"Checkpoint weights must be nonnegative and contain {sorted(expected)}"
        )
    recovery = t.setdefault("recovery", {})
    defaults = {
        "resume": True,
        "save_every_epochs": 1,
        "wait_for_resources": True,
        "poll_seconds": 60,
        "timeout_minutes": 0,
        "minimum_free_disk_gb": 5,
        "minimum_free_gpu_gb": 2,
        "cuda_oom_retries": 3,
        "on_incompatible": "fail",
    }
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
    if recovery["on_incompatible"] not in {"fail", "quarantine"}:
        raise ValueError("training.recovery.on_incompatible must be fail or quarantine")
    if any(n < 1 for n in e["node_counts"] + t["node_counts"]):
        raise ValueError("Node counts must be positive")
    if 1 not in e["node_counts"]:
        raise ValueError(
            "evaluation.node_counts must include 1 for cooperation baselines"
        )
    if e["subsets_per_count"] < 1 or e["max_detections"] < 1:
        raise ValueError("Invalid evaluation limits")
    e.setdefault("late_max_detections", e["max_detections"] * max(e["node_counts"]))
    if (
        not isinstance(e["late_max_detections"], int)
        or isinstance(e["late_max_detections"], bool)
        or e["late_max_detections"] < e["max_detections"]
    ):
        raise ValueError(
            "evaluation.late_max_detections must be an integer no smaller than max_detections"
        )
    e.setdefault("receiver_policy", "all")
    if e["receiver_policy"] not in {"all", "first"}:
        raise ValueError("evaluation.receiver_policy must be all or first")
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
    adaptive = cfg.setdefault("adaptive_fusion", {})
    adaptive.setdefault("training_budgets", [0.01, 0.05, 0.1])
    adaptive.setdefault("evaluation_budgets", [0.01, 0.05, 0.1, 0.25])
    adaptive.setdefault("ablations", ["random", "energy"])
    adaptive.setdefault("index_bytes", 4)
    adaptive.setdefault("value_bytes", 2)
    for key in ("training_budgets", "evaluation_budgets"):
        values = adaptive[key]
        if not values or any(
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 < value <= 1
            for value in values
        ):
            raise ValueError(f"adaptive_fusion.{key} must contain fractions in (0, 1]")
        if len(values) != len(set(values)):
            raise ValueError(f"adaptive_fusion.{key} contains duplicates")
    if adaptive["index_bytes"] not in (2, 4) or adaptive["value_bytes"] not in (2, 4):
        raise ValueError("adaptive fusion byte widths must be 2 or 4")
    height = math.ceil((d["bounds"][4] - d["bounds"][1]) / d["resolution"])
    width = math.ceil((d["bounds"][3] - d["bounds"][0]) / d["resolution"])
    if adaptive["index_bytes"] == 2 and height * width > 2**16:
        raise ValueError(
            "adaptive_fusion.index_bytes=2 cannot address every configured BEV cell"
        )
    if not isinstance(adaptive["ablations"], list) or not set(
        adaptive["ablations"]
    ) <= {"random", "energy"}:
        raise ValueError("adaptive_fusion.ablations supports random and energy")
    latency = cfg.setdefault("latency_experiment", {})
    latency.setdefault("frame_period_ms", 100.0)
    latency.setdefault("profile_batch_size", 1)
    latency.setdefault("deadlines_ms", [150.0, 250.0])
    latency.setdefault("burst_multipliers", [1, 2, 4])
    if latency["frame_period_ms"] <= 0:
        raise ValueError("latency_experiment.frame_period_ms must be positive")
    if (
        not isinstance(latency["profile_batch_size"], int)
        or isinstance(latency["profile_batch_size"], bool)
        or latency["profile_batch_size"] < 1
    ):
        raise ValueError("latency_experiment.profile_batch_size must be positive")
    if not latency["deadlines_ms"] or any(
        value <= 0 for value in latency["deadlines_ms"]
    ):
        raise ValueError("latency deadlines must be positive")
    if not latency["burst_multipliers"] or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in latency["burst_multipliers"]
    ):
        raise ValueError("latency burst multipliers must be positive integers")
    n = cfg["deployment_assumptions"]
    if not isinstance(n.get("source"), str) or not n["source"].strip():
        raise ValueError(
            "deployment_assumptions.source must identify the assumption source"
        )
    if any(not math.isfinite(x) or x <= 0 for x in n["bandwidths_mbps"]):
        raise ValueError("Bandwidth must be positive")
    minimum_deadline = n["sensing_ms"] + max(n["edge_rtt_ms"], n["cloud_rtt_ms"])
    if any(value <= minimum_deadline for value in latency["deadlines_ms"]):
        raise ValueError(
            "latency deadlines must exceed sensing_ms plus every configured RTT "
            f"({minimum_deadline:g} ms)"
        )
    for p in ("edge", "cloud"):
        if n[p + "_rtt_ms"] < 0 or any(
            n["processing_ms"][p][s] < 0 for s in e["strategies"]
        ):
            raise ValueError("Latency cannot be negative")
    return cfg
