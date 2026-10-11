import copy
import csv
import json
import math
import os
import subprocess
import sys
import weakref
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import yaml
import src.pipeline as pipeline_module
from src.config import PROJECT, enable_overfit, enable_smoke, load_config
from main import run_pipeline
from src.data.coinfra import (
    annotations,
    bev_channels,
    calibration,
    discover,
    load_cloud,
    rasterize,
)
from src.data.loader import BoundingBox3D
from src.data.visibility import box_point_counts
from src.detection.bev import REGRESSION_CHANNELS, BEVDetector, decode, loss, targets
from src.metrics.evaluation import (
    evaluate,
    f1_optimal_thresholds,
    iou3d,
    paired_iou3d,
)
from src.network.control import (
    PastOnlyLinkEstimator,
    past_frame,
    select_deadline_budget,
)
from src.pipeline import (
    audit_split_isolation,
    choose_groups,
    indexed,
    inspect,
    network_rows,
    prune_checkpoints,
    report,
    run_evaluation,
    simulate_fifo_latency,
    train,
    validation_metrics,
    verify_results,
)


def test_split_audit_rejects_shared_source_across_splits(tmp_path):
    common = str(tmp_path / "same.npy")
    base = {
        "session": "s",
        "slice": "slice0",
        "gt": str(tmp_path / "gt.yaml"),
        "nodes": {"1": {"cloud": common}},
    }
    samples = [
        base | {"id": "train/frame", "split": "train"},
        base | {"id": "test/frame", "split": "test"},
    ]
    with pytest.raises(ValueError, match="Data leakage"):
        audit_split_isolation(samples)


def test_write_csv_accepts_fields_added_to_later_rows(tmp_path):
    path = tmp_path / "metrics.csv"
    pipeline_module.write_csv(
        path,
        [
            {"epoch": 1, "loss": 0.5},
            {"epoch": 2, "loss": 0.25, "effective_batch_size": 4},
        ],
    )

    with path.open() as stream:
        rows = list(csv.DictReader(stream))

    assert rows[0] == {"epoch": "1", "loss": "0.5", "effective_batch_size": ""}
    assert rows[1]["effective_batch_size"] == "4"


def test_receiver_rotation_is_not_lexicographic():
    groups = choose_groups(
        {"10", "2"},
        {
            "subset_seed": 1,
            "node_counts": [2],
            "subsets_per_count": 1,
            "receiver_policy": "all",
        },
    )
    assert set(groups) == {("10", "2"), ("2", "10")}


def test_deadline_budget_uses_only_past_link_observations():
    estimator = PastOnlyLinkEstimator(10, alpha=1)
    estimator.observe(0, delivered_bytes=125_000, duration_ms=100)
    state = estimator.state(1, queue_ms=10)
    chosen = select_deadline_budget(
        [0.01, 0.05, 0.1, 0.25],
        cells=10_000,
        channels=32,
        collaborators=1,
        state=state,
        deadline_ms=150,
        fixed_ms=100,
    )
    assert chosen == 0.05
    with pytest.raises(ValueError, match="follow"):
        estimator.state(0)


def test_stale_lookup_is_strictly_past_and_sequence_local():
    history = [
        {"timestamp": str(value), "split": "test", "session": "s", "slice": "a"}
        for value in (1.0, 1.1, 1.2)
    ]
    assert past_frame(history, history[2], 1) is history[1]
    history[1] = history[1] | {"slice": "other"}
    assert past_frame(history, history[2], 1) is None


@pytest.fixture
def fixture_config(tmp_path):
    cfg = load_config(PROJECT / "configs/coinfra.yaml")
    d = cfg["dataset"]
    d.update(
        root=str(tmp_path / "data"),
        bounds=[-8, -8, -2, 8, 8, 4],
        resolution=1,
        frame_stride=1,
        classes=["car", "pedestrian"],
        # Fixture intensities are standard-normal draws.
        intensity_max=3.0,
        splits={
            "train": ["2025_01_01_sunny"],
            "val": ["2025_01_02_sunny"],
            "test": ["2025_01_03_rainy"],
        },
    )
    cfg["output"]["root"] = str(tmp_path / "output")
    cfg["training"].update(
        seeds=[1],
        epochs=1,
        minimum_epochs=1,
        early_stopping_patience=0,
        scheduler={
            "enabled": False,
            "factor": 0.5,
            "patience": 5,
            "minimum_learning_rate": 1e-5,
        },
        device="cpu",
        require_cuda=False,
        hidden_channels=4,
        node_counts=[1, 2],
    )
    cfg["evaluation"].update(
        node_counts=[1, 2],
        max_detections=5,
        score_threshold=0.01,
        class_iou_thresholds={"pedestrian": [0.25, 0.5]},
    )
    rng = np.random.default_rng(3)
    for sessions in d["splits"].values():
        for session in sessions:
            for slice_name in ("slice0", "slice1"):
                base = Path(d["root"]) / session / slice_name
                for node in ("4", "9"):
                    folder = base / "Calibration" / "lidar"
                    folder.mkdir(parents=True, exist_ok=True)
                    transform = np.eye(4)
                    transform[0, 3] = 1 if node == "4" else -1
                    (folder / f"{node}.yaml").write_text(
                        yaml.safe_dump(
                            {
                                "lidar_to_ground": np.eye(4).tolist(),
                                "ground_to_global": transform.tolist(),
                            }
                        )
                    )
                for timestamp in ("100.0", "100.1"):
                    folder = base / "PcImage" / timestamp / "lidar"
                    folder.mkdir(parents=True, exist_ok=True)
                    for node in ("4", "9"):
                        points = rng.normal(size=(40, 4)).astype(np.float32)
                        points[0, :3] = np.nan
                        np.save(folder / f"{node}_{timestamp}.npy", points)
                    gt = base / "GroundTruth" / timestamp
                    gt.mkdir(parents=True, exist_ok=True)
                    (gt / "global_roi.yaml").write_text(
                        yaml.safe_dump(
                            {
                                "objects": [
                                    {
                                        "class": "car",
                                        "position": [1, 0, 0],
                                        "dimensions": [4, 2, 2],
                                        "heading": 0.0,
                                    }
                                ]
                            }
                        )
                    )
    return cfg


def test_calibration_composition_and_nan_filter(fixture_config):
    sample = discover(fixture_config)[0]
    node = sample["nodes"]["4"]
    cloud = load_cloud(node, fixture_config["dataset"]["bounds"])
    raw = np.load(node["cloud"])
    xyz = raw[:, :3] + [1, 0, 0]
    mask = (
        np.isfinite(xyz).all(axis=1)
        & (xyz >= [-8, -8, -2]).all(axis=1)
        & (xyz < [8, 8, 4]).all(axis=1)
    )
    assert len(cloud) == int(mask.sum())
    assert np.isfinite(cloud).all()
    np.testing.assert_allclose(cloud[:, :3], xyz[mask], atol=1e-6)
    assert rasterize(cloud, fixture_config["dataset"]).shape == (4, 16, 16)
    assert calibration(node["calibration"])[0, 3] == 1


def test_default_coinfra_bounds_use_global_metric_frame(monkeypatch):
    monkeypatch.delenv("COINFRA_ROOT", raising=False)
    cfg = load_config(PROJECT / "configs/coinfra.yaml")
    bounds = cfg["dataset"]["bounds"]
    assert bounds[:2] == [-100.0, -100.0]
    assert bounds[3:5] == [100.0, 100.0]
    assert bounds[0] < 0 < bounds[3]
    assert bounds[1] < 0 < bounds[4]


# def test_coinfra_pilot_is_one_seed_early_fusion_profile():
#     cfg = load_config(PROJECT / "configs/coinfra_pilot.yaml")
#     assert cfg["training"]["seeds"] == [42]
#     assert cfg["training"]["epochs"] == 30
#     assert cfg["training"]["minimum_epochs"] == 20
#     assert cfg["training"]["hidden_channels"] == 64
#     assert cfg["training"]["modes"] == ["early"]
#     assert cfg["training"]["node_counts"] == [8]
#     assert cfg["training"]["checkpointing"]["weights"] == {
#         "val_loss_quality": 0.0,
#         "val_map": 0.8,
#         "val_recall": 0.2,
#     }
#     assert cfg["evaluation"]["strategies"] == ["early"]
#     assert cfg["evaluation"]["node_counts"] == [1, 8]
#     assert cfg["evaluation"]["subsets_per_count"] == 1
#     assert cfg["evaluation"]["ap_score_threshold"] == 0.01
#     assert cfg["evaluation"]["score_threshold"] == 0.15
#     assert Path(cfg["output"]["root"]).name == "coinfra_pilot"


def test_zero_yaw_and_empty_gt(fixture_config):
    sample = discover(fixture_config)[0]
    assert annotations(sample["gt"], fixture_config["dataset"])[0].yaw == 0
    Path(sample["gt"]).write_text("objects: []\n")
    assert annotations(sample["gt"], fixture_config["dataset"]) == []


def test_missing_gt_and_calibration_fail(fixture_config):
    sample = discover(fixture_config)[0]
    path = Path(sample["nodes"]["4"]["calibration"])
    path.unlink()
    with pytest.raises(FileNotFoundError):
        discover(fixture_config)


def test_sync_mismatch_fails(fixture_config):
    sample = discover(fixture_config)[0]
    p = Path(sample["nodes"]["4"]["cloud"])
    p.rename(p.with_name("4_900.0.npy"))
    with pytest.raises(ValueError, match="synchronization"):
        discover(fixture_config)


def test_per_slice_calibration(fixture_config):
    samples = discover(fixture_config)
    a = next(s for s in samples if s["slice"] == "slice0")
    b = next(s for s in samples if s["slice"] == "slice1")
    assert a["nodes"]["4"]["calibration"] != b["nodes"]["4"]["calibration"]


def test_rotated_3d_iou():
    a = BoundingBox3D(0, 0, 0, 4, 2, 2, 0, "car")
    b = BoundingBox3D(0, 0, 0, 4, 2, 2, math.pi / 2, "car")
    assert iou3d(a, a) == pytest.approx(1)
    assert iou3d(a, b) == pytest.approx(1 / 3)
    b.z = 10
    assert iou3d(a, b) == 0


def test_dataset_ap_false_positive_empty_frame():
    gt = BoundingBox3D(0, 0, 0, 4, 2, 2, 0, "car")
    pred = copy.copy(gt)
    pred.score = 0.5
    fp = copy.copy(gt)
    fp.score = 0.9
    rows = evaluate([([pred], [gt]), ([fp], [])], ["car", "pedestrian"], [0.5])
    assert rows[0]["ap_3d"] == pytest.approx(0.5)
    assert rows[0]["precision"] == pytest.approx(0.5)
    assert rows[0]["recall"] == pytest.approx(1)
    assert rows[0]["f1"] == pytest.approx(2 / 3)
    assert rows[0]["true_positives"] == 1
    assert rows[0]["false_positives"] == 1
    assert rows[0]["false_negatives"] == 0
    assert rows[0]["mean_matched_iou_3d"] == pytest.approx(1)
    assert rows[0]["mean_center_error_m"] == pytest.approx(0)
    assert rows[0]["mean_yaw_error_deg"] == pytest.approx(0)
    assert rows[0]["mean_size_iou"] == pytest.approx(1)
    assert rows[1]["ap_3d"] is None


def test_ap_threshold_is_separate_from_operating_point():
    gt = BoundingBox3D(0, 0, 0, 4, 2, 2, 0, "car")
    false_positive = BoundingBox3D(20, 20, 0, 4, 2, 2, 0, "car", 0.2)
    low_score_true_positive = copy.copy(gt)
    low_score_true_positive.score = 0.05
    row = evaluate(
        [([false_positive, low_score_true_positive], [gt])],
        ["car"],
        [0.5],
        operating_score_threshold=0.15,
    )[0]
    assert row["ap_3d"] == pytest.approx(0.5)
    assert row["precision"] == 0
    assert row["recall"] == 0
    assert row["predictions"] == 1
    assert row["ranked_predictions"] == 2


def test_group_selection_no_synthetic_nodes(fixture_config):
    groups = choose_groups(["4", "9"], fixture_config["evaluation"])
    assert groups == [("4",), ("9",), ("4", "9"), ("9", "4")]
    settings = fixture_config["evaluation"] | {"node_counts": [8]}
    with pytest.raises(ValueError):
        choose_groups(["4", "9"], settings)


def test_node_capacity_rejected_during_inspection(fixture_config):
    test_session = fixture_config["dataset"]["splits"]["test"][0]
    cloud = next(
        (Path(fixture_config["dataset"]["root"]) / test_session).glob(
            "slice*/PcImage/*/lidar/9_*.npy"
        )
    )
    cloud.unlink()
    with pytest.raises(ValueError, match="evaluation.node_counts requests 2"):
        inspect(fixture_config)


def test_inspection_parses_annotations_in_every_frame(fixture_config):
    samples = discover(fixture_config)
    later = next(
        sample
        for sample in samples
        if sample["split"] == "train" and sample["timestamp"] == "100.1"
    )
    Path(later["gt"]).write_text("objects:\n  - class: car\n    position: [bad]\n")
    with pytest.raises((TypeError, ValueError, KeyError)):
        inspect(fixture_config)


def test_split_overlap_rejected(fixture_config, tmp_path, monkeypatch):
    monkeypatch.delenv("COINFRA_ROOT", raising=False)
    fixture_config["dataset"]["splits"]["test"] = fixture_config["dataset"]["splits"][
        "train"
    ]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(fixture_config))
    with pytest.raises(ValueError, match="overlap"):
        load_config(path)


def test_partial_checkpointing_override_uses_defaults(
    fixture_config, tmp_path, monkeypatch
):
    monkeypatch.delenv("COINFRA_ROOT", raising=False)
    fixture_config["training"]["checkpointing"] = {"keep_best": 2}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(fixture_config))
    loaded = load_config(path)
    assert loaded["training"]["checkpointing"]["keep_best"] == 2
    assert set(loaded["training"]["checkpointing"]["weights"]) == {
        "val_loss_quality",
        "val_map",
        "val_recall",
    }


def test_coinfra_prepare_stage_is_explicit(fixture_config):
    with pytest.raises(ValueError, match="only supported for Adver-City"):
        run_pipeline(SimpleNamespace(stage="prepare", overwrite=False), fixture_config)


def test_network_units(fixture_config):
    rows = network_rows(fixture_config, {"strategy": "late"}, 1_000_000, 1000)
    assert rows[0]["transmission_ms_model"] == pytest.approx(0.8)
    assert rows[3]["transmission_ms_model"] == pytest.approx(800)
    assert (
        rows[0]["assumption_source"]
        == fixture_config["deployment_assumptions"]["source"]
    )


def test_fifo_latency_reports_queueing_and_deadline_misses():
    stable = simulate_fifo_latency([20, 20, 20], 100, 10, 100)
    assert stable["queue_wait_mean_ms"] == 0
    assert stable["deadline_miss_rate"] == 0
    overloaded = simulate_fifo_latency([80, 80, 80], 50, 10, 100)
    assert overloaded["queue_wait_mean_ms"] > 0
    assert overloaded["latency_p99_ms"] > overloaded["latency_p50_ms"]
    assert overloaded["deadline_miss_rate"] > 0


def test_smoke_profile_is_small_and_split_disjoint():
    cfg = load_config(PROJECT / "configs/coinfra.yaml")
    enable_smoke(cfg)
    splits = cfg["dataset"]["splits"]
    assert all(len(sessions) == 1 for sessions in splits.values())
    assert len({sessions[0] for sessions in splits.values()}) == 3
    assert cfg["training"]["epochs"] == 2
    assert cfg["training"]["seeds"] == [42]
    assert cfg["training"]["node_counts"] == [1, 2]
    assert cfg["evaluation"]["node_counts"] == [1, 2]
    assert Path(cfg["output"]["root"]).name == "smoke"


def test_overfit_profile_is_isolated_and_intentionally_reuses_training_data():
    cfg = load_config(PROJECT / "configs/coinfra.yaml")
    enable_overfit(cfg)
    assert cfg["diagnostic_overfit"] is True
    assert cfg["dataset"]["splits"]["train"] == cfg["dataset"]["splits"]["val"]
    assert cfg["dataset"]["smoke"]["frames_per_slice"] == 32
    assert cfg["training"]["seeds"] == [42]
    assert cfg["training"]["epochs"] == 500
    assert cfg["training"]["modes"] == ["early"]
    assert cfg["training"]["node_counts"] == [8]
    assert cfg["evaluation"]["strategies"] == ["early"]
    assert cfg["evaluation"]["node_counts"] == [1]
    assert Path(cfg["output"]["root"]).name == "overfit"


def test_checkpoint_pruning_keeps_highest_scores(tmp_path):
    for epoch, score in ((1, 0.2), (2, 0.8), (3, 0.5)):
        (tmp_path / f"epoch_{epoch:04d}.pt").write_bytes(b"checkpoint")
        (tmp_path / f"epoch_{epoch:04d}.json").write_text(
            json.dumps({"epoch": epoch, "selection_score": score})
        )
    retained = prune_checkpoints(tmp_path, 2)
    assert [item[1] for item in retained] == [2, 3]
    assert not (tmp_path / "epoch_0001.pt").exists()
    assert not (tmp_path / "epoch_0001.json").exists()


def test_checkpoint_pruning_ignores_truncated_metadata(tmp_path):
    (tmp_path / "epoch_0001.pt").write_bytes(b"checkpoint")
    (tmp_path / "epoch_0001.json").write_text("{")
    assert prune_checkpoints(tmp_path, 2) == []


def test_center_collision_uses_separate_slots(fixture_config):
    first = BoundingBox3D(1.0, 0.0, 0, 4, 2, 2, 0, "car")
    second = BoundingBox3D(1.1, 0.1, 0, 4, 2, 2, 0, "car")
    heat, regression, mask = targets(
        [first, second],
        fixture_config["dataset"],
        (16, 16),
        sample_id="session/slice/frame",
    )
    assert heat[0, 0, 8, 9] == 1
    assert heat[0, 1, 8, 9] == 1
    assert mask[0, :, 8, 9].tolist() == [True, True]
    assert regression[0, 0, 0, 8, 9] == pytest.approx(0)
    assert regression[0, 1, 0, 8, 9] == pytest.approx(0.1)


def test_center_collision_over_slot_capacity_is_explicit(fixture_config):
    box = BoundingBox3D(1, 0, 0, 4, 2, 2, 0, "car")
    with pytest.raises(ValueError, match="More than 2") as failure:
        targets(
            [box, box, box],
            fixture_config["dataset"],
            (16, 16),
            sample_id="session/slice/frame",
        )
    message = str(failure.value)
    assert "sample='session/slice/frame'" in message
    assert "class='car'" in message
    assert "cell=(x=9, y=8)" in message
    assert "resolution=1" in message
    assert "first_center=(1.000000, 0.000000)" in message
    assert "second_center=(1.000000, 0.000000)" in message
    assert "delta=(0.000000, 0.000000)" in message


def test_batched_detector_and_class_balanced_loss_are_finite(fixture_config):
    torch.manual_seed(7)
    model = BEVDetector(
        fixture_config["dataset"]["classes"],
        4,
        fixture_config["dataset"]["objects_per_cell"],
    )
    bev = torch.randn(2, 4, 16, 16)
    boxes = [
        [BoundingBox3D(1, 0, 0, 4, 2, 2, 0, "car")],
        [BoundingBox3D(-1, 1, 0, 1, 1, 2, 0, "pedestrian")],
    ]
    target = tuple(
        np.stack(values)
        for values in zip(
            *[targets(value, fixture_config["dataset"], (8, 8)) for value in boxes]
        )
    )
    batched_outputs = model(bev)
    batched_loss = loss(batched_outputs, target)
    assert torch.isfinite(batched_loss)
    batched_loss.backward()
    detector_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if not name.startswith("utility.")
    ]
    assert detector_gradients
    assert all(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in detector_gradients
    )

    per_node = torch.randn(2, 2, 4, 16, 16)
    batched = model(per_node, "intermediate")
    separate = [model(per_node[i], "intermediate") for i in range(2)]
    torch.testing.assert_close(batched[0], torch.cat([value[0] for value in separate]))
    torch.testing.assert_close(batched[1], torch.cat([value[1] for value in separate]))


def test_loss_returns_nonfinite_before_geometry_for_nan_box(
    fixture_config, monkeypatch
):
    classes = len(fixture_config["dataset"]["classes"])
    slots = fixture_config["dataset"]["objects_per_cell"]
    box = BoundingBox3D(1, 0, 0, 4, 2, 2, 0, "car")
    target = targets([box], fixture_config["dataset"], (8, 8))
    logits = torch.zeros((1, classes * slots, 8, 8))
    regression = torch.zeros((1, REGRESSION_CHANNELS * classes * slots, 8, 8))
    regression[:, 0] = float("nan")
    monkeypatch.setattr(
        "src.detection.bev.paired_iou3d",
        lambda *args: pytest.fail("nonfinite boxes must not reach geometry"),
    )

    value = loss((logits, regression), target)

    assert not torch.isfinite(value)


def test_encoder_receptive_field_spans_typical_car(fixture_config):
    model = BEVDetector(fixture_config["dataset"]["classes"], 4, 2)
    # Longer than a bus at the 0.4 m CoInfra resolution.
    assert model.receptive_field_cells * 0.4 >= 15.0
    assert (model.encoder(torch.randn(2, 4, 16, 16)) >= 0).all()
    logits, regression = model(torch.zeros(1, 4, 16, 16))
    assert logits.shape[-2:] == (8, 8)
    assert regression.shape[-2:] == (8, 8)


def test_detector_can_memorize_one_example(fixture_config):
    torch.manual_seed(3)
    model = BEVDetector(fixture_config["dataset"]["classes"], 8, 2)
    bev = torch.zeros(1, 4, 16, 16)
    bev[0, :, 7:10, 8:11] = torch.tensor([1.0, 0.6, 0.2, 0.4])[:, None, None]
    box = BoundingBox3D(1, 0, 0, 4, 2, 2, 0, "car")
    target = tuple(
        torch.from_numpy(value)[None]
        for value in targets([box], fixture_config["dataset"], (8, 8))
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0)
    initial = None
    for _ in range(150):
        optimizer.zero_grad(set_to_none=True)
        value = loss(model(bev), target)
        if initial is None:
            initial = float(value.detach())
        value.backward()
        optimizer.step()
    final = float(loss(model(bev), target).detach())
    assert final < initial * 0.1


def test_adaptive_fusion_obeys_spatial_budget(fixture_config):
    torch.manual_seed(11)
    model = BEVDetector(
        fixture_config["dataset"]["classes"],
        4,
        fixture_config["dataset"]["objects_per_cell"],
    )
    features = torch.rand(2, 3, 4, 10, 10)
    fused_small, selected_small = model.adaptive_features(features, 0.05)
    fused_large, selected_large = model.adaptive_features(features, 0.25)
    assert fused_small.shape == (2, 4, 10, 10)
    assert selected_small.shape == (2, 2, 10, 10)
    assert selected_small.sum(dim=(-1, -2)).unique().tolist() == [5]
    assert selected_large.sum(dim=(-1, -2)).unique().tolist() == [25]
    assert torch.all(fused_small >= features[:, 0])
    assert torch.all(fused_large >= features[:, 0])


def test_ablation_selectors_change_only_selected_cells(fixture_config):
    torch.manual_seed(21)
    model = BEVDetector(fixture_config["dataset"]["classes"], 4, 2)
    # Signed features: unsent cells must not act like a zero message.
    features = torch.randn(2, 3, 4, 10, 10)
    energy_fused, energy_mask = model.adaptive_features(features, 0.1, "energy")
    torch.manual_seed(22)
    random_fused, random_mask = model.adaptive_features(features, 0.1, "random")
    assert not torch.equal(random_mask[0, 0], random_mask[0, 1])
    assert not torch.equal(random_mask[0], random_mask[1])
    for fused, mask in ((energy_fused, energy_mask), (random_fused, random_mask)):
        sent = features[:, 1:].masked_fill(~mask[:, :, None], -float("inf"))
        expected = torch.maximum(features[:, 0], sent.amax(dim=1))
        torch.testing.assert_close(fused, expected)


def test_adaptive_fusion_leaves_unsent_receiver_cells_unchanged(fixture_config):
    torch.manual_seed(5)
    model = BEVDetector(fixture_config["dataset"]["classes"], 4, 2)
    features = torch.randn(1, 3, 4, 12, 12) - 1.0
    for selector in ("learned", "energy", "random"):
        fused, mask = model.adaptive_features(features, 0.05, selector)
        unsent = ~mask[0].any(dim=0)
        assert unsent.any()
        torch.testing.assert_close(fused[0][:, unsent], features[0, 0][:, unsent])


def test_full_budget_adaptive_equals_dense_max_fusion(fixture_config):
    model = BEVDetector(fixture_config["dataset"]["classes"], 4, 2)
    features = torch.randn(2, 3, 4, 6, 6)
    fused, _ = model.adaptive_features(features, 1.0, "energy")
    torch.testing.assert_close(fused, features.amax(dim=1))


def test_learned_selector_receives_gradient_through_sent_cells(fixture_config):
    torch.manual_seed(9)
    model = BEVDetector(fixture_config["dataset"]["classes"], 4, 2)
    features = torch.randn(1, 2, 4, 8, 8)
    fused, _ = model.adaptive_features(features, 0.25, "learned")
    fused.sum().backward()
    gradients = [p.grad for p in model.utility.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum() > 0 for g in gradients)


def test_adaptive_payload_grows_with_budget(fixture_config):
    torch.manual_seed(12)
    model = BEVDetector(
        fixture_config["dataset"]["classes"],
        4,
        fixture_config["dataset"]["objects_per_cell"],
    )
    features = torch.rand(1, 3, 4, 16, 16)
    _, small = model.adaptive_features(features, 0.01)
    _, large = model.adaptive_features(features, 0.25)
    small_bytes = pipeline_module.adaptive_payload(
        features[0], small[0], fixture_config
    )
    large_bytes = pipeline_module.adaptive_payload(
        features[0], large[0], fixture_config
    )
    assert 0 < small_bytes < large_bytes


def test_random_selector_is_independent_of_global_rng_history(fixture_config):
    model = BEVDetector(
        fixture_config["dataset"]["classes"],
        4,
        fixture_config["dataset"]["objects_per_cell"],
    )
    features = torch.rand(1, 3, 4, 8, 8)
    first_generator = torch.Generator().manual_seed(123)
    _, first = model.adaptive_features(
        features, 0.1, "random", generators=[first_generator]
    )
    torch.rand(1000)
    second_generator = torch.Generator().manual_seed(123)
    _, second = model.adaptive_features(
        features, 0.1, "random", generators=[second_generator]
    )
    assert torch.equal(first, second)


def test_dense_and_adaptive_payloads_exclude_receiver_and_use_same_value_width(
    fixture_config,
):
    features = np.ones((3, 4, 8, 8), dtype=np.float32)
    dense = sum(
        pipeline_module.feature_payload(item, fixture_config) for item in features[1:]
    )
    assert dense == 2 * pipeline_module.feature_payload(features[1], fixture_config)
    assert pipeline_module.feature_payload(
        features[0], fixture_config
    ) < pipeline_module.payload(features[0])


def test_duplicate_prediction_is_false_positive():
    truths = [
        BoundingBox3D(-0.5, 0, 0, 4, 2, 2, 0, "car"),
        BoundingBox3D(0.5, 0, 0, 4, 2, 2, 0, "car"),
    ]
    first = BoundingBox3D(-0.2, 0, 0, 4, 2, 2, 0, "car", score=0.9)
    duplicate = BoundingBox3D(-0.2, 0, 0, 4, 2, 2, 0, "car", score=0.8)
    row = evaluate([([first, duplicate], truths)], ["car"], [0.5])[0]
    assert row["true_positives"] == 1
    assert row["false_positives"] == 1
    assert row["precision"] == 0.5


def test_evaluation_fingerprint_covers_evaluation_only_settings(
    fixture_config, tmp_path
):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"weights")
    original = pipeline_module.evaluation_fingerprint(fixture_config, checkpoint)
    variants = []
    for section, key, value in (
        ("adaptive_fusion", "evaluation_budgets", [0.5]),
        ("deployment_assumptions", "edge_rtt_ms", 123.0),
        (
            "training",
            "mixed_precision",
            not fixture_config["training"]["mixed_precision"],
        ),
    ):
        changed = copy.deepcopy(fixture_config)
        changed[section][key] = value
        variants.append(pipeline_module.evaluation_fingerprint(changed, checkpoint))
    assert all(value != original for value in variants)


def test_finite_bursts_recover_between_groups():
    arrivals = pipeline_module.finite_burst_arrivals(8, 100.0, 4)
    assert arrivals[:4] == [0.0, 25.0, 50.0, 75.0]
    assert arrivals[4] == 400.0


def test_cooperation_gain_uses_only_participating_single_nodes(fixture_config):
    output = Path(fixture_config["output"]["root"])
    output.mkdir(parents=True)
    fixture_config["output"]["plots"] = False
    base = {
        "seed": 1,
        "session": "session",
        "weather": "sunny",
        "strategy": "early",
        "class": "car",
        "iou_level": 0,
        "iou": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "mean_matched_iou_3d": 0.6,
        "mean_center_error_m": 1.0,
        "mean_yaw_error_deg": 2.0,
        "mean_size_iou": 0.7,
        "true_positives": 1,
        "false_positives": 1,
        "false_negatives": 1,
        "ground_truth": 2,
        "predictions": 2,
        "frames": 1,
    }
    rows = [
        base | {"nodes": "A", "node_count": 1, "ap_3d": 0.9},
        base | {"nodes": "B", "node_count": 1, "ap_3d": 0.1},
        base | {"nodes": "C", "node_count": 1, "ap_3d": 0.8},
        base | {"nodes": "B,C", "node_count": 2, "ap_3d": 0.5},
    ]
    with (output / "accuracy.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "latency_communication.csv").write_text("")

    report(fixture_config)

    with (output / "cooperation_gains.csv").open() as stream:
        gain = next(csv.DictReader(stream))
    assert float(gain["single_node_ap_mean"]) == pytest.approx(0.45)
    assert float(gain["single_node_ap_best"]) == pytest.approx(0.8)
    assert gain["nodes"] == "B,C"


def test_validation_rejects_empty_split(fixture_config):
    model = BEVDetector(
        fixture_config["dataset"]["classes"],
        4,
        fixture_config["dataset"]["objects_per_cell"],
    )
    with pytest.raises(ValueError, match="Validation split contains no samples"):
        validation_metrics(model, [], fixture_config, torch.device("cpu"))


def test_validation_oom_splits_only_current_batch(monkeypatch, fixture_config):
    model = BEVDetector(
        fixture_config["dataset"]["classes"],
        4,
        fixture_config["dataset"]["objects_per_cell"],
    )
    gt = [BoundingBox3D(1, 0, 0, 4, 2, 2, 0, "car")]
    examples = []
    for _ in range(2):
        early = np.zeros((4, 16, 16), np.float32)
        per_node = np.zeros((2, 4, 16, 16), np.float32)
        target = targets(gt, fixture_config["dataset"], (8, 8))
        examples.append((early, per_node, (gt, []), target, 1, (gt, []), target))
    original = pipeline_module.device_tensor

    def oom_for_multi_item(array, device):
        if len(array) > 1:
            raise torch.OutOfMemoryError("out of memory")
        return original(array, torch.device("cpu"))

    monkeypatch.setattr(pipeline_module, "device_tensor", oom_for_multi_item)
    monkeypatch.setattr(pipeline_module, "release_cuda_memory", lambda device: None)
    fixture_config["training"]["mixed_precision"] = False
    losses, frames = pipeline_module.validate_examples(
        model, examples, fixture_config, SimpleNamespace(type="cuda"), retries=0
    )
    assert len(losses) == 2
    assert set(frames) == {*fixture_config["evaluation"]["strategies"], "single"}
    assert all(len(values) == 2 for values in frames.values())


def test_optimizer_step_oom_resets_scaler_before_retry(monkeypatch, fixture_config):
    model = BEVDetector(fixture_config["dataset"]["classes"], 4, 2)
    target = targets([], fixture_config["dataset"], (8, 8))
    examples = [(np.zeros((4, 16, 16), np.float32), target)]

    class Optimizer:
        def zero_grad(self, set_to_none=True):
            model.zero_grad(set_to_none=set_to_none)

    class Scaler:
        def __init__(self):
            self.state = "ready"
            self.calls = 0

        def scale(self, value):
            return value

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            if self.state == "stepped":
                raise RuntimeError("step() has already been called since update()")
            self.state = "stepped"
            self.calls += 1
            if self.calls == 1:
                raise torch.OutOfMemoryError("out of memory in optimizer step")

        def update(self):
            self.state = "ready"

    monkeypatch.setattr(
        pipeline_module,
        "device_tensor",
        lambda value, device: torch.as_tensor(value),
    )
    monkeypatch.setattr(pipeline_module, "release_cuda_memory", lambda device: None)
    monkeypatch.setattr(pipeline_module, "wait_for_resources", lambda *args: None)
    fixture_config["training"]["mixed_precision"] = False
    scaler = Scaler()
    losses = pipeline_module.train_examples(
        model,
        Optimizer(),
        scaler,
        examples,
        "early",
        fixture_config,
        SimpleNamespace(type="cuda"),
        "test",
        {"training": 1},
        retries=1,
    )
    assert len(losses) == 1
    assert scaler.calls == 2


def test_nonfinite_amp_loss_retries_batch_in_fp32(monkeypatch, fixture_config, capsys):
    autocast_states = []
    active = SimpleNamespace(enabled=False)

    class Autocast:
        def __init__(self, *, enabled, **kwargs):
            self.enabled = enabled

        def __enter__(self):
            active.enabled = self.enabled
            autocast_states.append(self.enabled)

        def __exit__(self, *args):
            active.enabled = False

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, bev, mode):
            value = self.weight * (float("inf") if active.enabled else 1.0)
            return value[None], value[None]

    class Scaler:
        def scale(self, value):
            return value

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass

    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    examples = {
        "bev": np.zeros((1, 1), np.float32),
        "target": tuple(np.zeros((1, 1), np.float32) for _ in range(3)),
    }
    fixture_config["training"]["mixed_precision"] = True
    monkeypatch.setattr(torch, "autocast", Autocast)
    monkeypatch.setattr(
        pipeline_module, "device_tensor", lambda value, device: torch.as_tensor(value)
    )
    monkeypatch.setattr(
        pipeline_module, "loss", lambda outputs, target, size: outputs[0].sum()
    )

    losses = pipeline_module.train_examples(
        model,
        optimizer,
        Scaler(),
        examples,
        "adaptive",
        fixture_config,
        SimpleNamespace(type="cuda"),
        "seed=43 epoch=10",
        {"training": 1},
        retries=0,
    )

    assert losses == [1.0]
    assert autocast_states == [True, False]
    assert "AMP nonfinite loss recovered in fp32" in capsys.readouterr().out


def test_fp32_retry_releases_amp_graph_first(monkeypatch, fixture_config):
    active = SimpleNamespace(enabled=False)
    amp_tensors = []

    class Autocast:
        def __init__(self, *, enabled, **kwargs):
            self.enabled = enabled

        def __enter__(self):
            active.enabled = self.enabled

        def __exit__(self, *args):
            active.enabled = False

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, bev, mode):
            if not active.enabled:
                # Holding the fp16 graph through the fp32 pass doubles peak memory.
                assert all(reference() is None for reference in amp_tensors)
            value = self.weight * (float("inf") if active.enabled else 1.0)
            output = value[None]
            if active.enabled:
                amp_tensors.append(weakref.ref(output))
            return output, output

    def tracked_loss(outputs, target, size):
        value = outputs[0].sum()
        if active.enabled:
            amp_tensors.append(weakref.ref(value))
        return value

    class Scaler:
        def scale(self, value):
            return value

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass

    model = Model()
    fixture_config["training"]["mixed_precision"] = True
    monkeypatch.setattr(torch, "autocast", Autocast)
    monkeypatch.setattr(
        pipeline_module, "device_tensor", lambda value, device: torch.as_tensor(value)
    )
    monkeypatch.setattr(pipeline_module, "loss", tracked_loss)
    losses = pipeline_module.train_examples(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        Scaler(),
        {
            "bev": np.zeros((1, 1), np.float32),
            "target": tuple(np.zeros((1, 1), np.float32) for _ in range(3)),
        },
        "intermediate",
        fixture_config,
        SimpleNamespace(type="cuda"),
        "test",
        {"training": 1},
        retries=0,
    )
    assert losses == [1.0]
    assert len(amp_tensors) == 2


def test_training_oom_split_releases_failed_batch(monkeypatch, fixture_config):
    failed = []

    def tracked_tensor(value, device):
        tensor = torch.as_tensor(value)
        if len(value) > 1:
            failed.append(weakref.ref(tensor))
        return tensor

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, bev, mode):
            if len(bev) > 1:
                raise torch.OutOfMemoryError("out of memory")
            # Parent frames must not keep the failed batch alive on the GPU.
            assert failed and all(reference() is None for reference in failed)
            value = self.weight * bev.sum()
            return value[None], value[None]

    class Scaler:
        def scale(self, value):
            return value

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass

    model = Model()
    fixture_config["training"]["mixed_precision"] = False
    monkeypatch.setattr(pipeline_module, "device_tensor", tracked_tensor)
    monkeypatch.setattr(pipeline_module, "release_cuda_memory", lambda device: None)
    monkeypatch.setattr(
        pipeline_module, "loss", lambda outputs, target, size: outputs[0].sum()
    )
    runtime_batches = {"training": 4}
    losses = pipeline_module.train_examples(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        Scaler(),
        {
            "bev": np.ones((4, 1), np.float32),
            "target": tuple(np.zeros((4, 1), np.float32) for _ in range(3)),
        },
        "early",
        fixture_config,
        SimpleNamespace(type="cuda"),
        "test",
        runtime_batches,
        retries=0,
    )
    assert len(losses) == 4
    assert runtime_batches["training"] == 1


def test_validation_oom_split_releases_failed_batch(monkeypatch, fixture_config):
    model = BEVDetector(
        fixture_config["dataset"]["classes"],
        4,
        fixture_config["dataset"]["objects_per_cell"],
    )
    gt = [BoundingBox3D(1, 0, 0, 4, 2, 2, 0, "car")]
    target = targets(gt, fixture_config["dataset"], (8, 8))
    early = np.zeros((4, 16, 16), np.float32)
    per_node = np.zeros((2, 4, 16, 16), np.float32)
    examples = [(early, per_node, (gt, []), target, 1, (gt, []), target)] * 2
    original = pipeline_module.device_tensor
    failed = []

    def oom_after_first_tensor(array, device):
        tensor = original(array, torch.device("cpu"))
        if len(array) > 1:
            failed.append(weakref.ref(tensor))
            if len(failed) == 2:
                raise torch.OutOfMemoryError("out of memory")
        else:
            assert all(reference() is None for reference in failed)
        return tensor

    monkeypatch.setattr(pipeline_module, "device_tensor", oom_after_first_tensor)
    monkeypatch.setattr(pipeline_module, "release_cuda_memory", lambda device: None)
    fixture_config["training"]["mixed_precision"] = False
    losses, _ = pipeline_module.validate_examples(
        model, examples, fixture_config, SimpleNamespace(type="cuda"), retries=0
    )
    assert len(losses) == 2
    assert len(failed) == 2


def test_nonfinite_tensor_summary_reports_counts():
    summary = pipeline_module.nonfinite_tensor_summary(
        [("finite", torch.zeros(2)), ("output[0]", torch.tensor([1.0, float("nan")]))]
    )
    assert summary == "output[0]=1/2"


def test_decode_accepts_mixed_precision_outputs(fixture_config):
    classes = len(fixture_config["dataset"]["classes"])
    slots = fixture_config["dataset"]["objects_per_cell"]
    logits = torch.full((1, classes * slots, 16, 16), -20, dtype=torch.float16)
    logits[0, 0, 8, 8] = 0
    regression = torch.zeros(
        (1, REGRESSION_CHANNELS * classes * slots, 16, 16), dtype=torch.float16
    )
    boxes = decode((logits, regression), fixture_config)
    assert boxes
    assert boxes[0].score == pytest.approx(0.5)


def test_decode_preserves_two_objects_from_one_class_cell(fixture_config):
    classes = len(fixture_config["dataset"]["classes"])
    slots = fixture_config["dataset"]["objects_per_cell"]
    logits = torch.full((1, classes * slots, 16, 16), -20.0)
    logits[0, 0, 8, 8] = 5
    logits[0, 1, 8, 8] = 4
    regression = torch.zeros((1, REGRESSION_CHANNELS * classes * slots, 16, 16))
    values = regression.reshape(1, classes, slots, REGRESSION_CHANNELS, 16, 16)
    values[0, 0, 0, 0, 8, 8] = 0.1
    values[0, 0, 1, 0, 8, 8] = 0.8
    values[0, 0, :, 3:6, 8, 8] = -4
    boxes = decode((logits, regression), fixture_config)
    cars = [box for box in boxes if box.label == "car"]
    assert len(cars) == 2
    assert sorted(box.x for box in cars) == pytest.approx([0.1, 0.8])


def test_end_to_end_train_all_fusions_report(fixture_config):
    torch.set_num_threads(1)
    fixture_config["evaluation"]["save_detections"] = True
    inspect(fixture_config)
    class_inventory = json.loads(
        (Path(fixture_config["output"]["root"]) / "class_inventory.json").read_text()
    )
    assert class_inventory["train"]["car"] == 4
    assert class_inventory["train"]["pedestrian"] == 0
    train(fixture_config)
    rows = run_evaluation(fixture_config)
    report(fixture_config)
    verification = verify_results(fixture_config)
    assert verification["status"] == "passed"
    expected_adaptive = {
        f"{name}@{budget}"
        for name in ("adaptive", "adaptive_random", "adaptive_energy")
        for budget in ("0.01", "0.05", "0.1", "0.25")
    }
    assert {r["strategy"] for r in rows} == {
        "early",
        "intermediate",
        "late",
    } | expected_adaptive
    assert {r["node_count"] for r in rows} == {1, 2}
    car_counts = {
        r["ground_truth"]
        for r in rows
        if r["class"] == "car" and r["gt_scope"] == "full"
    }
    assert car_counts == {4}
    assert {r["gt_scope"] for r in rows} == {"full", "visible"}
    visible = [r for r in rows if r["gt_scope"] == "visible"]
    assert all(
        r["ground_truth"] + r["ignored_ground_truth"] == 4
        for r in visible
        if r["class"] == "car"
    )
    output = Path(fixture_config["output"]["root"])
    assert (output / "seed_1/best.pt").exists()
    assert (output / "report.md").exists()
    with (output / "seed_1/training_metrics_seed_1.csv").open() as stream:
        training_row = next(csv.DictReader(stream))
    assert "val_ap_early_car" in training_row
    assert "val_precision_early_car" in training_row
    assert "val_recall_early_car" in training_row
    for filename in (
        "accuracy_summary.csv",
        "metrics_overview.csv",
        "cooperation_gains.csv",
        "weather_cooperation_summary.csv",
        "node_scaling.csv",
        "accuracy_latency_tradeoffs.csv",
        "adaptive_rate_accuracy.csv",
        "adaptive_rate_accuracy_auc.csv",
        "latency_trace.csv",
        "latency_tail_summary.csv",
        "verification.json",
    ):
        assert (output / filename).exists()
    with (output / "adaptive_rate_accuracy.csv").open() as stream:
        adaptive_rows = list(csv.DictReader(stream))
    assert adaptive_rows
    assert {float(row["budget_fraction"]) for row in adaptive_rows} == {
        0.01,
        0.05,
        0.1,
        0.25,
    }
    with (output / "cooperation_gains.csv").open() as stream:
        gain_rows = list(csv.DictReader(stream))
    assert gain_rows
    assert {"ap_gain_over_mean_single", "recall_gain_over_mean_single"} <= set(
        gain_rows[0]
    )
    with (output / "seed_1/training_metrics_seed_1.csv").open() as stream:
        training_rows = list(csv.DictReader(stream))
    assert {
        "val_map_early",
        "val_map_intermediate",
        "val_map_late",
        "val_map_adaptive",
        "val_precision",
        "val_f1",
    } <= set(training_rows[0])
    assert not (output / "seed_1/training.csv").exists()
    first = training_rows[0]
    assert float(first["val_map"]) == pytest.approx(
        0.5 * (float(first["val_map_single"]) + float(first["val_map_cooperative"]))
    )
    assert "val_map_early_iou0.7" in first
    # One node: every strategy label reports the same single-node detector.
    single = defaultdict(set)
    for row in rows:
        if row["node_count"] == 1:
            single[(row["nodes"], row["class"], row["iou_level"])].add(row["ap_3d"])
    assert single and all(len(values) == 1 for values in single.values())
    assert {r["iou"] for r in rows if r["class"] == "pedestrian"} == {0.25, 0.5}
    # Cloud placement uploads every participating node, the receiver included.
    with (output / "latency_communication.csv").open() as stream:
        cloud = [r for r in csv.DictReader(stream) if r["placement"] == "cloud"]
    assert all(float(r["payload_bytes"]) > 0 for r in cloud)
    with (output / "latency_trace.csv").open() as stream:
        traces = list(csv.DictReader(stream))
    single_bytes = {
        (t["sample"], t["nodes"]): float(t["cloud_raw_bytes"])
        for t in traces
        if t["node_count"] == "1"
    }
    pairs = [t for t in traces if t["node_count"] == "2"]
    assert pairs and all(
        float(t["cloud_raw_bytes"])
        == sum(single_bytes[(t["sample"], node)] for node in t["nodes"].split(","))
        for t in pairs
    )
    # The streamed trace is exactly the per-session parts, concatenated.
    part_rows = 0
    for part in output.glob("eval_parts/seed_1/*.traces.csv"):
        with part.open() as stream:
            part_rows += sum(1 for _ in csv.DictReader(stream))
    assert part_rows == len(traces) > 0
    train(fixture_config)
    thresholds = json.loads((output / "seed_1/operating_thresholds.json").read_text())
    assert set(thresholds["thresholds"]) == set(fixture_config["dataset"]["classes"])
    # Pedestrians have no validation truth and keep the configured fallback.
    assert thresholds["thresholds"]["pedestrian"] == pytest.approx(
        fixture_config["evaluation"]["score_threshold"]
    )
    assert list(output.glob("eval_parts/seed_1/*.detections.csv.gz"))
    assert (output / "range_breakdown.csv").exists()
    report_text = (output / "report.md").read_text()
    for heading in (
        "## Validation coverage",
        "## Operating thresholds",
        "## Sensor point counts",
        "## Cases where cooperation reduced AP",
        "## Latency feasibility",
        "## Accuracy by range",
    ):
        assert heading in report_text
    with (output / "latency_communication.csv").open() as stream:
        cost = next(csv.DictReader(stream))
    assert float(cost["end_to_end_ms_measured"]) > 0
    with (output / "latency_tail_summary.csv").open() as stream:
        assert "queue_stable" in next(csv.DictReader(stream))
    assert json.loads((output / "environment.json").read_text())["cpu_count"] >= 1
    fixture_config["dataset"]["frame_stride"] = 2
    with pytest.raises(ValueError, match="changed"):
        indexed(fixture_config)


def test_cli_from_another_directory(fixture_config, tmp_path):
    config = tmp_path / "run.yaml"
    config.write_text(yaml.safe_dump(fixture_config))
    env = os.environ.copy()
    env.pop("COINFRA_ROOT", None)
    env["COINFRA_PYTHON"] = sys.executable
    env["ECCO_LOG_DIR"] = str(tmp_path / "logs")
    result = subprocess.run(
        ["bash", str(PROJECT / "run.sh"), "--config", str(config), "--stage", "check"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Configuration valid" in result.stdout


def test_training_incomplete_checkpoint_resumes_from_latest(fixture_config):
    torch.set_num_threads(1)
    train(fixture_config)
    path = Path(fixture_config["output"]["root"]) / "seed_1/complete.json"
    path.unlink()
    fixture_config["training"]["preprocessing_workers"] = 2
    fixture_config["output"]["plots"] = False
    train(fixture_config)
    assert path.exists()
    assert (path.parent / "latest.pt").exists()


def test_completed_training_can_extend_epochs(fixture_config):
    torch.set_num_threads(1)
    train(fixture_config)
    fixture_config["training"]["epochs"] = 2
    fixture_config["training"]["batch_size"] = 1
    train(fixture_config)
    completion = Path(fixture_config["output"]["root"]) / "seed_1/complete.json"
    assert json.loads(completion.read_text())["epochs"] == 2


def test_incompatible_seed_is_quarantined_and_retrained(fixture_config):
    torch.set_num_threads(1)
    train(fixture_config)
    output = Path(fixture_config["output"]["root"])
    original = torch.load(
        output / "seed_1/best.pt", map_location="cpu", weights_only=True
    )

    fixture_config["training"]["learning_rate"] *= 2
    train(fixture_config)

    replacement = torch.load(
        output / "seed_1/best.pt", map_location="cpu", weights_only=True
    )
    quarantined = list((output / "stale_checkpoints").glob("seed_1_*"))
    assert len(quarantined) == 1
    assert (quarantined[0] / "best.pt").exists()
    assert original["signature"] != replacement["signature"]


def test_evaluation_reuses_completed_session_parts(monkeypatch, fixture_config):
    torch.set_num_threads(1)
    train(fixture_config)
    first = run_evaluation(fixture_config)
    monkeypatch.setattr(
        pipeline_module,
        "infer_batch",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("recomputed")),
    )
    second = run_evaluation(fixture_config)
    assert second == first


def test_inspection_checks_collisions_on_prediction_grid(fixture_config):
    d = fixture_config["dataset"]
    # One metre BEV cells hold at most two of these centers; the 2 m
    # prediction cells used by targets() hold all three.
    walkers = [
        BoundingBox3D(x, 0.1, 0.0, 0.6, 0.6, 1.7, 0.0, "pedestrian")
        for x in (0.1, 1.1, 1.9)
    ]
    grid = pipeline_module.output_shape(d, fixture_config["training"]["output_stride"])
    assert grid == (8, 8)
    count, description = pipeline_module.validate_target_centers(
        walkers, d, "frame", grid
    )
    assert count == 3
    assert "pedestrian" in description
    with pytest.raises(pipeline_module.TargetCollisionError):
        targets(walkers, d, grid)


def test_inspection_reports_required_slots(fixture_config):
    samples = discover(fixture_config)
    crowded = next(sample for sample in samples if sample["split"] == "train")
    Path(crowded["gt"]).write_text(
        yaml.safe_dump(
            {
                "objects": [
                    {
                        "class": "pedestrian",
                        "position": [x, 0.1, 0],
                        "dimensions": [0.6, 0.6, 1.7],
                        "heading": 0.0,
                    }
                    for x in (0.1, 1.1, 1.9)
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="objects_per_cell to at least 3"):
        inspect(fixture_config)


def test_focal_loss_keeps_gradient_for_confident_false_positive(fixture_config):
    target = tuple(
        torch.from_numpy(value)[None]
        for value in targets([], fixture_config["dataset"], (8, 8))
    )
    classes = len(fixture_config["dataset"]["classes"])
    logits = torch.full((1, classes * 2, 8, 8), -6.0, dtype=torch.float16)
    logits[0, 0, 3, 3] = 12.0
    logits.requires_grad_(True)
    regression = torch.zeros(
        (1, REGRESSION_CHANNELS * classes * 2, 8, 8), dtype=torch.float16
    )
    loss((logits, regression), target).backward()
    gradient = float(logits.grad[0, 0, 3, 3])
    assert math.isfinite(gradient) and gradient > 0


def test_late_payload_counts_only_operating_point_boxes(fixture_config):
    fixture_config["evaluation"]["score_threshold"] = 0.15
    kept = BoundingBox3D(1, 0, 0, 4, 2, 2, 0, "car", 0.9)
    candidate = BoundingBox3D(3, 0, 0, 4, 2, 2, 0, "car", 0.05)
    assert pipeline_module.late_payload(
        [kept, candidate], fixture_config
    ) == pipeline_module.late_payload([kept], fixture_config)


def test_patience_ignores_gains_below_min_delta():
    history = [{"selection_score": value} for value in (0.5, 0.5004, 0.5008, 0.5012)]
    reference, stale = pipeline_module.patience_progress(history, 0.0005)
    assert reference == pytest.approx(0.5008)
    assert stale == 1


def test_augmentation_moves_points_and_boxes_together(fixture_config):
    from src.data.augmentation import augment_scene

    d = fixture_config["dataset"]
    box = BoundingBox3D(2.0, 1.0, 0.5, 4.0, 2.0, 1.5, 0.3, "car")
    ahead = np.array([[2.0 + math.cos(0.3), 1.0 + math.sin(0.3), 0.5, 7.0]], np.float32)
    for flip_x in (False, True):
        for flip_y in (False, True):
            parameters = {
                "flip_x": flip_x,
                "flip_y": flip_y,
                "rotation": 0.4,
                "scale": 1.05,
            }
            (cloud,), (moved,) = augment_scene([ahead], [box], parameters, d)
            expected = (
                moved.x + 1.05 * math.cos(moved.yaw),
                moved.y + 1.05 * math.sin(moved.yaw),
            )
            np.testing.assert_allclose(cloud[0, :2], expected, atol=1e-5)
            assert cloud[0, 3] == 7.0
            assert moved.length == pytest.approx(4.2)


def test_augmented_collision_falls_back_to_original_labels(monkeypatch, fixture_config):
    sample = next(s for s in discover(fixture_config) if s["split"] == "train")
    crowd = [
        BoundingBox3D(0.1, 0.1, 0, 0.6, 0.6, 1.7, 0, "pedestrian") for _ in range(3)
    ]
    monkeypatch.setattr(
        pipeline_module, "augment_scene", lambda clouds, boxes, *args: (clouds, crowd)
    )
    plan = (
        sample,
        "early",
        ["4"],
        {"flip_x": False, "flip_y": False, "rotation": 0.0, "scale": 1.0},
    )
    _, (_, _, mask) = pipeline_module.prepare_training_example(plan, fixture_config)
    assert mask.sum() == 1


def test_class_iou_thresholds_apply_per_class():
    truth = [
        BoundingBox3D(0, 0, 0, 4, 2, 2, 0, "car"),
        BoundingBox3D(10, 0, 0, 0.6, 0.6, 1.7, 0, "pedestrian"),
    ]
    shifted = [
        BoundingBox3D(0.8, 0, 0, 4, 2, 2, 0, "car", 0.9),
        BoundingBox3D(10.2, 0, 0, 0.6, 0.6, 1.7, 0, "pedestrian", 0.9),
    ]
    rows = evaluate(
        [(shifted, truth)],
        ["car", "pedestrian"],
        [0.5, 0.7],
        class_thresholds={"pedestrian": [0.25, 0.5]},
    )
    applied = {(row["class"], row["iou_level"]): row["iou"] for row in rows}
    assert applied == {
        ("car", 0): 0.5,
        ("car", 1): 0.7,
        ("pedestrian", 0): 0.25,
        ("pedestrian", 1): 0.5,
    }
    pedestrian = {
        row["iou_level"]: row["ap_3d"] for row in rows if row["class"] == "pedestrian"
    }
    assert pedestrian[0] == pytest.approx(1)


def test_nms_matches_reference_greedy_algorithm():
    import random

    rng = random.Random(4)
    boxes = [
        BoundingBox3D(
            rng.uniform(-6, 6),
            rng.uniform(-6, 6),
            0,
            rng.uniform(1, 5),
            rng.uniform(0.5, 2),
            1.5,
            rng.uniform(-3, 3),
            rng.choice(["car", "pedestrian"]),
            rng.random(),
        )
        for _ in range(120)
    ]
    for threshold in (0.05, 0.2, 0.5):
        reference = []
        for box in sorted(boxes, key=lambda b: b.score, reverse=True):
            if all(
                box.label != b.label or iou3d(box, b) <= threshold for b in reference
            ):
                reference.append(box)
        assert pipeline_module.nms(boxes, threshold, 1000) == reference
        assert pipeline_module.nms(boxes, threshold, 7) == reference[:7]


def test_intensity_channel_and_scale_guard(fixture_config):
    d = fixture_config["dataset"] | {"intensity_max": 255.0, "use_intensity": True}
    points = np.array([[0.5, 0.5, 0.0, 510.0], [0.5, 0.5, 0.0, 51.0]], np.float32)
    assert rasterize(points, d)[4].max() == pytest.approx(0.6)
    pipeline_module.validate_intensity_scale([np.linspace(0, 250, 100)], d)
    with pytest.raises(ValueError, match="intensity_max"):
        pipeline_module.validate_intensity_scale([np.linspace(0, 1, 100)], d)
    with pytest.raises(ValueError, match="intensity_max"):
        pipeline_module.validate_intensity_scale([np.linspace(0, 4000, 100)], d)


def test_retry_does_not_repeat_overwrite(tmp_path):
    record = tmp_path / "calls.txt"
    fake = tmp_path / "python"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$*" >> "{record}"\n'
        f'if [[ $(wc -l < "{record}") -eq 1 ]]; then exit 75; fi\n'
        "exit 0\n"
    )
    fake.chmod(0o755)
    env = os.environ.copy() | {
        "ECCO_PYTHON": str(fake),
        "ECCO_LOG_DIR": str(tmp_path / "logs"),
        "ECCO_MAX_ATTEMPTS": "2",
        "ECCO_RETRY_DELAY_SECONDS": "0",
    }
    result = subprocess.run(
        ["bash", str(PROJECT / "run.sh"), "--config", "x.yaml", "--overwrite"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    first, second = record.read_text().splitlines()
    assert "--overwrite" in first
    assert "--overwrite" not in second
    assert "--config x.yaml" in second


def test_rasterize_height_channels_without_intensity(fixture_config):
    d = fixture_config["dataset"] | {"use_intensity": False}
    points = np.array(
        [[0.5, 0.5, -2.0, 9.0], [0.5, 0.5, 1.0, 9.0], [0.5, 0.5, 4.0 - 1e-3, 9.0]],
        np.float32,
    )
    bev = rasterize(points, d)
    assert bev.shape[0] == bev_channels(d) == 4
    cell = (8, 8)
    assert bev[0][cell] == pytest.approx(math.log1p(3))
    assert bev[1][cell] == pytest.approx(1.0, abs=1e-3)
    assert bev[2][cell] == pytest.approx(0.0)
    assert bev[3][cell] == pytest.approx((0 + 0.5 + 1.0) / 3, abs=1e-3)
    # Empty cells stay zero in every channel.
    assert not bev[:, 0, 0].any()


def test_constant_intensity_is_rejected_when_used(fixture_config):
    d = fixture_config["dataset"] | {"use_intensity": True, "intensity_max": 1.0}
    with pytest.raises(ValueError, match="use_intensity: false"):
        pipeline_module.validate_intensity_scale([np.ones(1000)], d)
    pipeline_module.validate_intensity_scale(
        [np.ones(1000)], d | {"use_intensity": False}
    )


def test_box_point_counts_match_brute_force():
    rng = np.random.default_rng(5)
    bounds = [-20, -20, -3, 20, 20, 5]
    points = rng.uniform([-20, -20, -3, 0], [20, 20, 5, 1], size=(5000, 4))
    boxes = [
        BoundingBox3D(
            *rng.uniform([-18, -18, -1], [18, 18, 2]),
            *rng.uniform([1, 0.5, 1], [6, 3, 3]),
            rng.uniform(-3, 3),
            "car",
        )
        for _ in range(25)
    ]
    expected = []
    for box in boxes:
        dx, dy = points[:, 0] - box.x, points[:, 1] - box.y
        along = dx * math.cos(box.yaw) + dy * math.sin(box.yaw)
        across = -dx * math.sin(box.yaw) + dy * math.cos(box.yaw)
        expected.append(
            int(
                np.count_nonzero(
                    (np.abs(along) <= box.length / 2)
                    & (np.abs(across) <= box.width / 2)
                    & (np.abs(points[:, 2] - box.z) <= box.height / 2)
                )
            )
        )
    np.testing.assert_array_equal(box_point_counts(points, boxes, bounds), expected)
    assert box_point_counts(points[:0], boxes, bounds).tolist() == [0] * len(boxes)


def test_training_targets_skip_boxes_without_points(fixture_config):
    sample = next(s for s in discover(fixture_config) if s["split"] == "train")
    Path(sample["gt"]).write_text(
        yaml.safe_dump(
            {
                "objects": [
                    {
                        "class": "car",
                        "position": [1, 0, 0],
                        "dimensions": [4, 2, 2],
                        "heading": 0.0,
                    },
                    # Far outside the fixture's unit-normal point clouds.
                    {
                        "class": "car",
                        "position": [6, 6, 0],
                        "dimensions": [1, 1, 1],
                        "heading": 0.0,
                    },
                ]
            }
        )
    )
    plan = (sample, "early", ["4", "9"], None)
    _, (_, _, mask) = pipeline_module.prepare_training_example(plan, fixture_config)
    assert mask.sum() == 1
    fixture_config["training"]["min_target_points"] = 0
    _, (_, _, mask) = pipeline_module.prepare_training_example(plan, fixture_config)
    assert mask.sum() == 2


def test_prediction_of_ignored_truth_is_neither_true_nor_false_positive():
    scored = BoundingBox3D(0, 0, 0, 4, 2, 2, 0, "car")
    ignored = BoundingBox3D(10, 0, 0, 4, 2, 2, 0, "car")
    predictions = [
        BoundingBox3D(0, 0, 0, 4, 2, 2, 0, "car", 0.9),
        BoundingBox3D(10, 0, 0, 4, 2, 2, 0, "car", 0.8),
        BoundingBox3D(-10, 0, 0, 4, 2, 2, 0, "car", 0.7),
    ]
    records = []
    row = evaluate(
        [(predictions, [scored], [ignored])], ["car"], [0.5], 0.5, records=records
    )[0]
    assert (row["true_positives"], row["false_positives"]) == (1, 1)
    assert row["predictions"] == 2
    assert row["ignored_predictions"] == 1
    assert (row["ground_truth"], row["ignored_ground_truth"]) == (1, 1)
    assert row["ap_3d"] == pytest.approx(1.0)
    statuses = sorted((r["record"], r["status"]) for r in records)
    assert statuses == [
        ("ground_truth", "ignored"),
        ("ground_truth", "matched"),
        ("prediction", "fp"),
        ("prediction", "ignored"),
        ("prediction", "tp"),
    ]
    # Without the ignored list the same prediction is a false positive.
    plain = evaluate([(predictions, [scored])], ["car"], [0.5], 0.5)[0]
    assert plain["false_positives"] == 2


def test_operating_threshold_may_differ_per_class():
    truth = [BoundingBox3D(0, 0, 0, 4, 2, 2, 0, "car")]
    prediction = [BoundingBox3D(0, 0, 0, 4, 2, 2, 0, "car", 0.3)]
    strict = evaluate([(prediction, truth)], ["car"], [0.5], {"car": 0.5})[0]
    loose = evaluate([(prediction, truth)], ["car"], [0.5], {"car": 0.2})[0]
    assert (strict["true_positives"], loose["true_positives"]) == (0, 1)
    assert strict["operating_score_threshold"] == 0.5


def test_f1_optimal_threshold_and_fallback():
    truths = [BoundingBox3D(10 * i, 0, 0, 4, 2, 2, 0, "car") for i in range(3)]
    predictions = [
        BoundingBox3D(0, 0, 0, 4, 2, 2, 0, "car", 0.9),
        BoundingBox3D(10, 0, 0, 4, 2, 2, 0, "car", 0.8),
        BoundingBox3D(50, 0, 0, 4, 2, 2, 0, "car", 0.7),
        BoundingBox3D(60, 0, 0, 4, 2, 2, 0, "car", 0.6),
        BoundingBox3D(20, 0, 0, 4, 2, 2, 0, "car", 0.2),
    ]
    chosen = f1_optimal_thresholds(
        [(predictions, truths)],
        ["car", "pedestrian"],
        [0.5],
        fallback=0.15,
        minimum=0.01,
    )
    # Top-2: F1 = 4/5; top-5: F1 = 6/8. The higher threshold wins.
    assert chosen["car"] == pytest.approx(0.8)
    assert chosen["pedestrian"] == 0.15


def test_iou_quality_target_and_rectified_score(fixture_config):
    from src.detection.bev import regression_boxes

    d = fixture_config["dataset"]
    box = BoundingBox3D(1.3, 0.4, 0.2, 4, 2, 1.5, 0.3, "car")
    heat, reg, mask = targets([box], d, (8, 8))
    expected = torch.from_numpy(reg).permute(0, 1, 3, 4, 2)[torch.from_numpy(mask)]
    same = paired_iou3d(
        regression_boxes(expected, (2.0, 2.0)), regression_boxes(expected, (2.0, 2.0))
    )
    assert same.tolist() == pytest.approx([1.0])
    # A perfect box with a confident IoU logit has a lower loss than with a
    # pessimistic one.
    classes = len(d["classes"])
    slots = d["objects_per_cell"]
    logits = torch.full((1, classes * slots, 8, 8), -8.0)
    regression = torch.zeros(1, classes, slots, REGRESSION_CHANNELS, 8, 8)
    regression[:, :, :, :8] = torch.from_numpy(reg)[None]
    target = tuple(torch.from_numpy(v)[None] for v in (heat, reg, mask))
    regression[:, :, :, 8] = 8.0
    confident = loss((logits, regression.reshape(1, -1, 8, 8)), target, (2.0, 2.0))
    regression[:, :, :, 8] = -8.0
    pessimistic = loss((logits, regression.reshape(1, -1, 8, 8)), target, (2.0, 2.0))
    assert float(confident) < float(pessimistic)
    # Decode multiplies the heatmap score by the predicted IoU.
    logits = torch.full((1, classes * slots, 16, 16), -20.0)
    logits[0, 0, 8, 8] = 0
    regression = torch.zeros((1, REGRESSION_CHANNELS * classes * slots, 16, 16))
    regression.reshape(1, classes, slots, REGRESSION_CHANNELS, 16, 16)[
        0, 0, 0, 8, 8, 8
    ] = -20.0
    fixture_config["evaluation"]["ap_score_threshold"] = 0.0
    fixture_config["evaluation"]["iou_score_weight"] = 0.5
    rectified = decode((logits, regression), fixture_config, score_threshold=0.0)
    assert rectified[0].score < 1e-3


def test_network_rows_report_required_rate_and_measured_latency(fixture_config):
    base = {
        "strategy": "early",
        "preprocessing_ms_mean": 5.0,
        "inference_decode_ms_mean": 20.0,
        "serialization_ms_mean": 1.0,
        "cloud_serialization_ms_mean": 3.0,
    }
    rows = network_rows(fixture_config, base, 2_500_000, 1_250_000)
    edge = next(r for r in rows if r["placement"] == "edge")
    net = fixture_config["deployment_assumptions"]
    # 1.25 MB per 100 ms frame needs 100 Mbps.
    assert edge["required_bandwidth_mbps"] == pytest.approx(100.0)
    assert edge["end_to_end_ms_measured"] == pytest.approx(
        net["sensing_ms"] + net["edge_rtt_ms"] + 26.0 + edge["transmission_ms_model"]
    )
    assert edge["link_keeps_up"] is False  # 10 Mbps link: 1000 ms per frame


def test_mixed_node_pilot_trains_longer_on_more_subsets_and_seeds():
    cfg = load_config(PROJECT / "configs/coinfra_mixed_nodes_pilot.yaml")
    assert cfg["training"]["epochs"] == 200
    assert cfg["training"]["minimum_epochs"] == 100
    assert cfg["training"]["seeds"] == [42, 43, 44]
    assert cfg["evaluation"]["subsets_per_count"] == 5
    assert cfg["dataset"]["use_intensity"] is False
    assert cfg["evaluation"]["save_detections"] is True
