"""Validated configuration with project-relative paths."""

import math
import os
from pathlib import Path
import yaml

PROJECT = Path(__file__).resolve().parents[1]


def validate_splits(splits, mode="sessions"):
    if mode == "random_frames":
        # CoInfra's published protocol: every listed session is pooled and its
        # frames are assigned to splits at random, so sessions are shared.
        if not splits["train"] or splits["val"] or splits["test"]:
            raise ValueError(
                "split_mode random_frames lists every pooled session under "
                "splits.train and leaves splits.val and splits.test empty"
            )
        return
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
    cfg["dataset"]["split_mode"] = "sessions"
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


def enable_overfit(cfg, frames=32, optimization_steps=2000):
    """Enable an intentionally leaky tiny-set profile for learning diagnostics.

    This profile is never selected from a YAML file.  It must be requested
    explicitly on the command line and writes beneath a separate output root.
    """
    d, t, e = cfg["dataset"], cfg["training"], cfg["evaluation"]
    overfit_node_count = max(t["node_counts"])
    batches_per_epoch = math.ceil(frames / t["batch_size"])
    epochs = math.ceil(optimization_steps / batches_per_epoch)
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
            "modes": ["early"],
            "node_counts": [overfit_node_count],
        }
    )
    t.setdefault("scheduler", {})["enabled"] = False
    # Memorization needs the identical frames every epoch.
    t.setdefault("augmentation", {})["enabled"] = False
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
    d, t, e = cfg["dataset"], cfg["training"], cfg["evaluation"]
    dataset_type = d.setdefault("type", "coinfra")
    if dataset_type not in ("coinfra", "advercity"):
        raise ValueError("dataset.type must be coinfra or advercity")
    root_environment = (
        "ADVERCITY_ROOT" if dataset_type == "advercity" else "COINFRA_ROOT"
    )
    d["root"] = str(resolve(os.environ.get(root_environment, d["root"])))
    if dataset_type == "advercity":
        acquisition = d.setdefault("acquisition", {})
        acquisition["work_dir"] = str(
            resolve(
                os.environ.get(
                    "ADVERCITY_WORK_DIR", acquisition.get("work_dir", "advercity_work")
                )
            )
        )
        acquisition.setdefault("minimum_free_gb", 35)
        acquisition.setdefault("cleanup_archives", True)
        acquisition.setdefault("retain_frame_stride", 1)
        acquisition.setdefault("allow_legacy_labels", False)
        if not isinstance(acquisition["allow_legacy_labels"], bool):
            raise ValueError("dataset.acquisition.allow_legacy_labels must be boolean")
        if acquisition["minimum_free_gb"] < 1 or acquisition["retain_frame_stride"] < 1:
            raise ValueError("Invalid Adver-City acquisition settings")
    cfg["output"]["root"] = str(resolve(cfg["output"]["root"]))
    cfg["output"].setdefault("plots", True)
    smoke = d.setdefault("smoke", {"enabled": False})
    smoke.setdefault("enabled", False)
    smoke.setdefault("splits", d["splits"])
    smoke.setdefault("sessions_per_split", 1)
    smoke.setdefault("slices_per_session", 1)
    smoke.setdefault("frames_per_slice", 1)
    smoke.setdefault("epochs", 2)
    smoke.setdefault("seeds", 1)
    if smoke["enabled"]:
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
    d.setdefault("intensity_max", None)
    if d["intensity_max"] is not None and (
        not isinstance(d["intensity_max"], (int, float))
        or isinstance(d["intensity_max"], bool)
        or not math.isfinite(d["intensity_max"])
        or d["intensity_max"] <= 0
    ):
        raise ValueError("dataset.intensity_max must be null or a positive number")
    d.setdefault("use_intensity", True)
    if not isinstance(d["use_intensity"], bool):
        raise ValueError("dataset.use_intensity must be boolean")
    d.setdefault("height_slices", 5)
    d.setdefault("height_slice_m", 0.5)
    if (
        not isinstance(d["height_slices"], int)
        or isinstance(d["height_slices"], bool)
        or d["height_slices"] < 0
        or not isinstance(d["height_slice_m"], (int, float))
        or not d["height_slice_m"] > 0
    ):
        raise ValueError(
            "dataset.height_slices must be a nonnegative integer and "
            "dataset.height_slice_m positive"
        )
    if d["yaw_unit"] not in ("degrees", "radians"):
        raise ValueError("yaw_unit must be degrees or radians")
    if d["frame_stride"] < 1 or d.get("max_timestamp_delta_s", 0) < 0:
        raise ValueError("Invalid stride or synchronization tolerance")
    d.setdefault("train_frame_stride", d["frame_stride"])
    if (
        not isinstance(d["train_frame_stride"], int)
        or isinstance(d["train_frame_stride"], bool)
        or d["train_frame_stride"] < 1
    ):
        raise ValueError("dataset.train_frame_stride must be a positive integer")
    d.setdefault("split_mode", "sessions")
    if d["split_mode"] not in ("sessions", "random_frames"):
        raise ValueError("dataset.split_mode must be sessions or random_frames")
    if (
        d["split_mode"] == "random_frames"
        and d["train_frame_stride"] != d["frame_stride"]
    ):
        raise ValueError(
            "split_mode random_frames splits one frame list, so "
            "dataset.train_frame_stride must equal dataset.frame_stride"
        )
    random_split = d.setdefault("random_split", {})
    random_split.setdefault("fractions", [0.8, 0.1, 0.1])
    random_split.setdefault("seed", 0)
    fractions = random_split["fractions"]
    if (
        not isinstance(fractions, list)
        or len(fractions) != 3
        or any(not isinstance(v, (int, float)) or not v > 0 for v in fractions)
        or not math.isclose(sum(fractions), 1.0)
    ):
        raise ValueError(
            "dataset.random_split.fractions must be three positive train/val/test "
            "fractions summing to 1"
        )
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
    validate_splits(d["splits"], d["split_mode"])
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
    if "hidden_channels" in t:
        raise ValueError(
            "training.hidden_channels was replaced by training.node_channels and "
            "training.backbone_channels"
        )
    t.setdefault("node_channels", 64)
    t.setdefault("backbone_channels", 128)
    t.setdefault("node_blocks", 1)
    t.setdefault("backbone_blocks", [2, 2, 2])
    t.setdefault("fusion", "attention")
    if (
        any(
            not isinstance(t[key], int) or isinstance(t[key], bool) or t[key] < 1
            for key in ("node_channels", "backbone_channels")
        )
        or not isinstance(t["node_blocks"], int)
        or isinstance(t["node_blocks"], bool)
        or t["node_blocks"] < 0
        or not isinstance(t["backbone_blocks"], list)
        or len(t["backbone_blocks"]) != 3
        or any(
            not isinstance(v, int) or isinstance(v, bool) or v < 1
            for v in t["backbone_blocks"]
        )
    ):
        raise ValueError(
            "training node/backbone channels must be positive, node_blocks "
            "nonnegative, and backbone_blocks three positive integers"
        )
    if t["fusion"] not in ("attention", "mean", "max"):
        raise ValueError("training.fusion must be attention, mean, or max")
    t.setdefault("amp_dtype", "auto")
    if t["amp_dtype"] not in ("auto", "bfloat16", "float16"):
        raise ValueError("training.amp_dtype must be auto, bfloat16, or float16")
    t.setdefault("repeat_factor_threshold", 0.0)
    if (
        not isinstance(t["repeat_factor_threshold"], (int, float))
        or isinstance(t["repeat_factor_threshold"], bool)
        or not 0 <= t["repeat_factor_threshold"] <= 1
    ):
        raise ValueError("training.repeat_factor_threshold must be in [0, 1]")
    stale = t.setdefault("stale", {})
    stale.setdefault("probability", 0.0)
    stale.setdefault("max_age_frames", 3)
    if (
        not isinstance(stale["probability"], (int, float))
        or not 0 <= stale["probability"] <= 1
        or not isinstance(stale["max_age_frames"], int)
        or isinstance(stale["max_age_frames"], bool)
        or stale["max_age_frames"] < 1
    ):
        raise ValueError(
            "training.stale needs probability in [0, 1] and a positive max_age_frames"
        )
    t.setdefault("output_stride", 2)
    if t["output_stride"] != 2:
        raise ValueError("training.output_stride must currently be 2")
    t.setdefault("minimum_epochs", t["epochs"])
    # Boxes with fewer points from the chosen nodes are not training targets.
    t.setdefault("min_target_points", 1)
    if (
        not isinstance(t["min_target_points"], int)
        or isinstance(t["min_target_points"], bool)
        or t["min_target_points"] < 0
    ):
        raise ValueError("training.min_target_points must be a nonnegative integer")
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
    augmentation = t.setdefault("augmentation", {})
    if not isinstance(augmentation, dict):
        raise ValueError("training.augmentation must be a mapping")
    augmentation.setdefault("enabled", False)
    augmentation.setdefault("flip_x", True)
    augmentation.setdefault("flip_y", True)
    augmentation.setdefault("rotation_degrees", 22.5)
    augmentation.setdefault("scale_range", [0.95, 1.05])
    scale_range = augmentation["scale_range"]
    if (
        any(
            not isinstance(augmentation[key], bool)
            for key in ("enabled", "flip_x", "flip_y")
        )
        or not isinstance(augmentation["rotation_degrees"], (int, float))
        or not 0 <= augmentation["rotation_degrees"] <= 180
        or not isinstance(scale_range, list)
        or len(scale_range) != 2
        or not all(
            isinstance(value, (int, float)) and math.isfinite(value) and value > 0
            for value in scale_range
        )
        or scale_range[0] > scale_range[1]
    ):
        raise ValueError("Invalid training.augmentation configuration")
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
    # 1 / (1 + validation loss) barely moves, so AP drives selection.
    checkpointing.setdefault(
        "weights", {"val_loss_quality": 0.0, "val_map": 0.8, "val_recall": 0.2}
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
    class_iou = e.setdefault("class_iou_thresholds", {})
    if not isinstance(class_iou, dict) or any(
        label not in d["classes"]
        or not isinstance(values, list)
        or len(values) != len(e["iou_thresholds"])
        or any(
            not isinstance(value, (int, float)) or not 0 < value <= 1
            for value in values
        )
        for label, values in class_iou.items()
    ):
        raise ValueError(
            "evaluation.class_iou_thresholds must map configured classes to one "
            "IoU in (0, 1] per evaluation.iou_thresholds level"
        )
    e.setdefault("ap_score_threshold", e["score_threshold"])
    e.setdefault("min_visible_points", 5)
    e.setdefault("iou_score_weight", 0.5)
    e.setdefault("tune_score_thresholds", True)
    e.setdefault("save_detections", False)
    if (
        not isinstance(e["min_visible_points"], int)
        or isinstance(e["min_visible_points"], bool)
        or e["min_visible_points"] < 1
    ):
        raise ValueError("evaluation.min_visible_points must be a positive integer")
    if (
        not isinstance(e["iou_score_weight"], (int, float))
        or isinstance(e["iou_score_weight"], bool)
        or not 0 <= e["iou_score_weight"] <= 1
    ):
        raise ValueError("evaluation.iou_score_weight must be in [0, 1]")
    for key in ("tune_score_thresholds", "save_detections"):
        if not isinstance(e[key], bool):
            raise ValueError(f"evaluation.{key} must be boolean")
    if (
        not 0 <= e["ap_score_threshold"] <= e["score_threshold"] <= 1
        or not 0 < e["nms_iou"] <= 1
    ):
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
    ) <= {"random", "energy", "confidence"}:
        raise ValueError(
            "adaptive_fusion.ablations supports random, energy, and confidence"
        )
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
    validate_comparison(e, d["classes"])
    validate_deadline_experiment(cfg)
    return cfg


def validate_comparison(e, classes):
    """Optional metric matching a published protocol (merged classes, BEV IoU)."""
    comparison = e.setdefault("comparison", None)
    if comparison is None:
        return
    comparison.setdefault("iou_type", "bev")
    comparison.setdefault("iou_threshold", 0.5)
    groups = comparison.get("groups")
    members = [label for values in (groups or {}).values() for label in values]
    if (
        comparison["iou_type"] not in ("bev", "3d")
        or not 0 < comparison["iou_threshold"] <= 1
        or not isinstance(groups, dict)
        or not groups
        or any(not isinstance(values, list) or not values for values in groups.values())
        or len(members) != len(set(members))
        or not set(members) <= set(classes)
    ):
        raise ValueError(
            "evaluation.comparison needs iou_type bev or 3d, an IoU in (0, 1], and "
            "groups mapping names to disjoint lists of configured classes"
        )


def validate_deadline_experiment(cfg):
    """Settings for deadline-constrained cooperative detection on 10 Hz sequences."""
    x = cfg.setdefault("deadline_experiment", {"enabled": False})
    x.setdefault("enabled", False)
    if not x["enabled"]:
        return
    defaults = {
        "frames_per_slice": 200,
        "slices_per_session": 1,
        "deadlines_ms": [200.0, 250.0],
        "target_miss_rate": 0.05,
        "tradeoffs": [0.5, 2.0, 10.0],
        "max_miss_probability": 0.5,
        "max_age_frames": 3,
        "estimator_alpha": 0.2,
        "controllers": ["fixed", "dense", "greedy", "drive", "oracle", "early", "late"],
        "traces": [
            {
                "name": "gilbert_elliott",
                "type": "gilbert_elliott",
                "good_mbps": 40.0,
                "bad_mbps": 5.0,
                "p_good_to_bad": 0.05,
                "p_bad_to_good": 0.25,
                "jitter": 0.2,
            }
        ],
    }
    for key, value in defaults.items():
        x.setdefault(key, value)
    known = {"fixed", "dense", "greedy", "drive", "oracle", "early", "late"}
    n = cfg["deployment_assumptions"]
    window = (
        n["sensing_ms"]
        + n["edge_rtt_ms"] / 2
        + n["processing_ms"]["edge"].get("adaptive", 0.0)
    )
    if (
        any(
            not isinstance(x[key], int) or isinstance(x[key], bool) or x[key] < 1
            for key in ("frames_per_slice", "slices_per_session", "max_age_frames")
        )
        or not x["deadlines_ms"]
        or any(value <= window for value in x["deadlines_ms"])
        or not 0 < x["target_miss_rate"] < 1
        or not 0 < x["max_miss_probability"] <= 1
        or not x["tradeoffs"]
        or any(not value > 0 for value in x["tradeoffs"])
        or not 0 < x["estimator_alpha"] <= 1
        or not x["controllers"]
        or not set(x["controllers"]) <= known
        or not x["traces"]
    ):
        raise ValueError(
            "Invalid deadline_experiment: positive frame and age limits, deadlines "
            f"above sensing + RTT/2 + receiver processing ({window:g} ms), "
            f"target_miss_rate in (0, 1), and controllers from {sorted(known)}"
        )
    names = set()
    for trace in x["traces"]:
        kind = trace.get("type")
        name = trace.get("name")
        if not name or name in names:
            raise ValueError("deadline_experiment traces need unique names")
        names.add(name)
        if kind == "constant":
            valid = trace.get("mbps", 0) > 0
        elif kind == "gilbert_elliott":
            valid = (
                trace.get("good_mbps", 0) > 0
                and trace.get("bad_mbps", 0) > 0
                and 0 < trace.get("p_good_to_bad", 0) < 1
                and 0 < trace.get("p_bad_to_good", 0) < 1
                and trace.setdefault("jitter", 0.0) >= 0
            )
        elif kind == "csv":
            trace["path"] = str(resolve(trace.get("path", "")))
            valid = Path(trace["path"]).suffix == ".csv"
        else:
            valid = False
        if not valid:
            raise ValueError(f"Invalid deadline_experiment trace: {trace}")
