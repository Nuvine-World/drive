import copy
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import yaml
import src.pipeline as pipeline_module
from src.config import PROJECT, enable_smoke, load_config
from main import run_pipeline
from src.data.coinfra import annotations, calibration, discover, load_cloud, rasterize
from src.data.loader import BoundingBox3D
from src.detection.bev import BEVDetector, decode, loss, targets
from src.metrics.evaluation import evaluate, iou3d
from src.pipeline import (
    choose_groups,
    indexed,
    inspect,
    network_rows,
    prune_checkpoints,
    report,
    run_evaluation,
    train,
    validation_metrics,
    verify_results,
)


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
        device="cpu",
        require_cuda=False,
        hidden_channels=4,
        node_counts=[1, 2],
    )
    cfg["evaluation"].update(node_counts=[1, 2], max_detections=5, score_threshold=0.01)
    rng = np.random.default_rng(3)
    for sessions in d['splits'].values():
        for session in sessions:
            for slice_name in ('slice0','slice1'):
                base = Path(d['root'])/session/slice_name
                for node in ('4','9'):
                    folder = base/'Calibration'/'lidar'
                    folder.mkdir(parents=True,exist_ok=True)
                    transform = np.eye(4)
                    transform[0,3] = 1 if node=='4' else -1
                    (folder/f'{node}.yaml').write_text(yaml.safe_dump({
                        'lidar_to_ground':np.eye(4).tolist(),'ground_to_global':transform.tolist()}))
                for timestamp in ('100.0','100.1'):
                    folder = base/'PcImage'/timestamp/'lidar'
                    folder.mkdir(parents=True,exist_ok=True)
                    for node in ('4','9'):
                        points = rng.normal(size=(40,4)).astype(np.float32)
                        points[0,:3] = np.nan
                        np.save(folder/f'{node}_{timestamp}.npy',points)
                    gt = base/'GroundTruth'/timestamp
                    gt.mkdir(parents=True,exist_ok=True)
                    (gt/'global_roi.yaml').write_text(yaml.safe_dump({'objects':[
                        {'class':'car','position':[1,0,0],'dimensions':[4,2,2],'heading':0.0}]}))
    return cfg


def test_calibration_composition_and_nan_filter(fixture_config):
    sample = discover(fixture_config)[0]
    node = sample['nodes']['4']
    cloud = load_cloud(node,fixture_config['dataset']['bounds'])
    raw = np.load(node['cloud'])
    xyz = raw[:,:3]+[1,0,0]
    mask = np.isfinite(xyz).all(axis=1) & (xyz >= [-8,-8,-2]).all(axis=1) & (xyz < [8,8,4]).all(axis=1)
    assert len(cloud)==int(mask.sum())
    assert np.isfinite(cloud).all()
    np.testing.assert_allclose(cloud[:,:3],xyz[mask],atol=1e-6)
    assert rasterize(cloud,fixture_config['dataset']).shape==(3,16,16)
    assert calibration(node['calibration'])[0,3]==1


def test_default_coinfra_bounds_use_global_metric_frame(monkeypatch):
    monkeypatch.delenv("COINFRA_ROOT", raising=False)
    cfg = load_config(PROJECT / "configs/coinfra.yaml")
    bounds = cfg["dataset"]["bounds"]
    assert bounds[:2] == [-100.0, -100.0]
    assert bounds[3:5] == [100.0, 100.0]
    assert bounds[0] < 0 < bounds[3]
    assert bounds[1] < 0 < bounds[4]


def test_zero_yaw_and_empty_gt(fixture_config):
    sample = discover(fixture_config)[0]
    assert annotations(sample['gt'],fixture_config['dataset'])[0].yaw==0
    Path(sample['gt']).write_text('objects: []\n')
    assert annotations(sample['gt'],fixture_config['dataset'])==[]


def test_missing_gt_and_calibration_fail(fixture_config):
    sample = discover(fixture_config)[0]
    path = Path(sample['nodes']['4']['calibration'])
    path.unlink()
    with pytest.raises(FileNotFoundError):
        discover(fixture_config)


def test_sync_mismatch_fails(fixture_config):
    sample = discover(fixture_config)[0]
    p = Path(sample['nodes']['4']['cloud'])
    p.rename(p.with_name('4_900.0.npy'))
    with pytest.raises(ValueError,match='synchronization'):
        discover(fixture_config)


def test_per_slice_calibration(fixture_config):
    samples = discover(fixture_config)
    a = next(s for s in samples if s['slice']=='slice0')
    b = next(s for s in samples if s['slice']=='slice1')
    assert a['nodes']['4']['calibration']!=b['nodes']['4']['calibration']


def test_rotated_3d_iou():
    a = BoundingBox3D(0,0,0,4,2,2,0,'car')
    b = BoundingBox3D(0,0,0,4,2,2,math.pi/2,'car')
    assert iou3d(a,a)==pytest.approx(1)
    assert iou3d(a,b)==pytest.approx(1/3)
    b.z=10
    assert iou3d(a,b)==0


def test_dataset_ap_false_positive_empty_frame():
    gt = BoundingBox3D(0,0,0,4,2,2,0,'car')
    pred = copy.copy(gt)
    pred.score=.5
    fp = copy.copy(gt)
    fp.score=.9
    rows = evaluate([([pred],[gt]),([fp],[])],['car','pedestrian'],[.5])
    assert rows[0]['ap_3d']==pytest.approx(.5)
    assert rows[0]['precision']==pytest.approx(.5)
    assert rows[0]['recall']==pytest.approx(1)
    assert rows[0]['f1']==pytest.approx(2/3)
    assert rows[0]['true_positives']==1
    assert rows[0]['false_positives']==1
    assert rows[0]['false_negatives']==0
    assert rows[0]['mean_matched_iou_3d']==pytest.approx(1)
    assert rows[0]['mean_center_error_m']==pytest.approx(0)
    assert rows[0]['mean_yaw_error_deg']==pytest.approx(0)
    assert rows[0]['mean_size_iou']==pytest.approx(1)
    assert rows[1]['ap_3d'] is None


def test_group_selection_no_synthetic_nodes(fixture_config):
    groups = choose_groups(['4','9'],fixture_config['evaluation'])
    assert groups==[('4',),('9',),('4','9')]
    settings = fixture_config['evaluation'] | {'node_counts':[8]}
    with pytest.raises(ValueError):
        choose_groups(['4','9'],settings)


def test_node_capacity_rejected_during_inspection(fixture_config):
    test_session = fixture_config['dataset']['splits']['test'][0]
    cloud = next((Path(fixture_config['dataset']['root'])/test_session).glob(
        'slice*/PcImage/*/lidar/9_*.npy'))
    cloud.unlink()
    with pytest.raises(ValueError,match='evaluation.node_counts requests 2'):
        inspect(fixture_config)


def test_inspection_parses_annotations_in_every_frame(fixture_config):
    samples = discover(fixture_config)
    later = next(sample for sample in samples if sample['split']=='train' and sample['timestamp']=='100.1')
    Path(later['gt']).write_text('objects:\n  - class: car\n    position: [bad]\n')
    with pytest.raises((TypeError,ValueError,KeyError)):
        inspect(fixture_config)


def test_split_overlap_rejected(fixture_config,tmp_path,monkeypatch):
    monkeypatch.delenv('COINFRA_ROOT',raising=False)
    fixture_config['dataset']['splits']['test']=fixture_config['dataset']['splits']['train']
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(fixture_config))
    with pytest.raises(ValueError,match='overlap'):
        load_config(path)


def test_partial_checkpointing_override_uses_defaults(fixture_config,tmp_path,monkeypatch):
    monkeypatch.delenv('COINFRA_ROOT',raising=False)
    fixture_config['training']['checkpointing'] = {'keep_best': 2}
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(fixture_config))
    loaded = load_config(path)
    assert loaded['training']['checkpointing']['keep_best']==2
    assert set(loaded['training']['checkpointing']['weights']) == {
        'val_loss_quality','val_map','val_recall'}


def test_coinfra_prepare_stage_is_explicit(fixture_config):
    with pytest.raises(ValueError,match='only supported for Adver-City'):
        run_pipeline(SimpleNamespace(stage='prepare',overwrite=False),fixture_config)


def test_network_units(fixture_config):
    rows = network_rows(fixture_config,{'strategy':'late'},1_000_000,1000)
    assert rows[0]['transmission_ms_model']==pytest.approx(.8)
    assert rows[3]['transmission_ms_model']==pytest.approx(800)
    assert rows[0]['assumption_source']==fixture_config['deployment_assumptions']['source']


def test_smoke_profile_is_small_and_split_disjoint():
    cfg = load_config(PROJECT/'configs/coinfra.yaml')
    enable_smoke(cfg)
    splits = cfg["dataset"]["splits"]
    assert all(len(sessions) == 1 for sessions in splits.values())
    assert len({sessions[0] for sessions in splits.values()}) == 3
    assert cfg["training"]["epochs"] == 2
    assert cfg["training"]["seeds"] == [42]
    assert cfg["training"]["node_counts"] == [1, 2]
    assert cfg["evaluation"]["node_counts"] == [1, 2]
    assert Path(cfg["output"]["root"]).name == "smoke"


def test_checkpoint_pruning_keeps_highest_scores(tmp_path):
    for epoch, score in ((1,.2),(2,.8),(3,.5)):
        (tmp_path/f'epoch_{epoch:04d}.pt').write_bytes(b'checkpoint')
        (tmp_path/f'epoch_{epoch:04d}.json').write_text(json.dumps(
            {'epoch':epoch,'selection_score':score}))
    retained = prune_checkpoints(tmp_path,2)
    assert [item[1] for item in retained]==[2,3]
    assert not (tmp_path/'epoch_0001.pt').exists()
    assert not (tmp_path/'epoch_0001.json').exists()


def test_checkpoint_pruning_ignores_truncated_metadata(tmp_path):
    (tmp_path / "epoch_0001.pt").write_bytes(b"checkpoint")
    (tmp_path / "epoch_0001.json").write_text("{")
    assert prune_checkpoints(tmp_path, 2) == []


def test_center_collision_explicit(fixture_config):
    box = BoundingBox3D(1, 0, 0, 4, 2, 2, 0, "car")
    with pytest.raises(ValueError, match="grid cell") as failure:
        targets(
            [box, box],
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


def test_batched_detector_and_loss_match_individual_samples(fixture_config):
    torch.manual_seed(7)
    model = BEVDetector(fixture_config["dataset"]["classes"], 4)
    bev = torch.randn(2, 3, 16, 16)
    boxes = [
        [BoundingBox3D(1, 0, 0, 4, 2, 2, 0, "car")],
        [BoundingBox3D(-1, 1, 0, 1, 1, 2, 0, "pedestrian")],
    ]
    target = tuple(
        np.stack(values)
        for values in zip(
            *[targets(value, fixture_config["dataset"], (16, 16)) for value in boxes]
        )
    )
    batched_outputs = model(bev)
    batched_loss = loss(batched_outputs, target)
    individual = [
        loss(
            (batched_outputs[0][i : i + 1], batched_outputs[1][i : i + 1]),
            tuple(value[i] for value in target),
        )
        for i in range(2)
    ]
    expected = float(torch.stack(individual).mean().detach())
    assert float(batched_loss.detach()) == pytest.approx(expected, rel=1e-6)

    per_node = torch.randn(2, 2, 3, 16, 16)
    batched = model(per_node, "intermediate")
    separate = [model(per_node[i], "intermediate") for i in range(2)]
    torch.testing.assert_close(batched[0], torch.cat([value[0] for value in separate]))
    torch.testing.assert_close(batched[1], torch.cat([value[1] for value in separate]))


def test_validation_rejects_empty_split(fixture_config):
    model = BEVDetector(fixture_config["dataset"]["classes"], 4)
    with pytest.raises(ValueError, match="Validation split contains no samples"):
        validation_metrics(model, [], fixture_config, torch.device("cpu"))


def test_validation_oom_splits_only_current_batch(monkeypatch, fixture_config):
    model = BEVDetector(fixture_config["dataset"]["classes"], 4)
    gt = [BoundingBox3D(1, 0, 0, 4, 2, 2, 0, "car")]
    examples = []
    for _ in range(2):
        early = np.zeros((3, 16, 16), np.float32)
        per_node = np.zeros((2, 3, 16, 16), np.float32)
        target = targets(gt, fixture_config["dataset"], (16, 16))
        examples.append((early, per_node, gt, target))
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
    assert all(len(values) == 2 for values in frames.values())


def test_decode_accepts_mixed_precision_outputs(fixture_config):
    classes = len(fixture_config["dataset"]["classes"])
    logits = torch.full((1, classes, 16, 16), -20, dtype=torch.float16)
    logits[0, 0, 8, 8] = 0
    regression = torch.zeros((1, 8 * classes, 16, 16), dtype=torch.float16)
    boxes = decode((logits, regression), fixture_config)
    assert boxes
    assert boxes[0].score == pytest.approx(0.5)


def test_end_to_end_train_all_fusions_report(fixture_config):
    torch.set_num_threads(1)
    inspect(fixture_config)
    train(fixture_config)
    rows = run_evaluation(fixture_config)
    report(fixture_config)
    verification = verify_results(fixture_config)
    assert verification['status']=='passed'
    assert {r['strategy'] for r in rows}=={'early','intermediate','late'}
    assert {r['node_count'] for r in rows}=={1,2}
    car_counts = {r['ground_truth'] for r in rows if r['class']=='car'}
    assert car_counts=={4}
    output = Path(fixture_config['output']['root'])
    assert (output/'seed_1/best.pt').exists()
    assert (output/'report.md').exists()
    for filename in ('accuracy_summary.csv','metrics_overview.csv','cooperation_gains.csv',
                     'weather_cooperation_summary.csv','node_scaling.csv',
                     'accuracy_latency_tradeoffs.csv','verification.json'):
        assert (output/filename).exists()
    with (output/'cooperation_gains.csv').open() as stream:
        gain_rows = list(csv.DictReader(stream))
    assert gain_rows
    assert {'ap_gain_over_mean_single','recall_gain_over_mean_single'} <= set(gain_rows[0])
    with (output/'seed_1/training_metrics_seed_1.csv').open() as stream:
        training_rows = list(csv.DictReader(stream))
    assert {'val_map_early','val_map_intermediate','val_map_late','val_precision','val_f1'} <= set(training_rows[0])
    assert not (output/'seed_1/training.csv').exists()
    train(fixture_config)
    fixture_config['dataset']['frame_stride']=2
    with pytest.raises(ValueError,match='changed'):
        indexed(fixture_config)


def test_cli_from_another_directory(fixture_config,tmp_path):
    config = tmp_path/'run.yaml'
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
    path = Path(fixture_config['output']['root'])/'seed_1/complete.json'
    path.unlink()
    fixture_config["training"]["preprocessing_workers"] = 2
    fixture_config["output"]["plots"] = False
    train(fixture_config)
    assert path.exists()
    assert (path.parent/'latest.pt').exists()


def test_completed_training_can_extend_epochs(fixture_config):
    torch.set_num_threads(1)
    train(fixture_config)
    fixture_config["training"]["epochs"] = 2
    fixture_config["training"]["batch_size"] = 1
    train(fixture_config)
    completion = Path(fixture_config["output"]["root"]) / "seed_1/complete.json"
    assert json.loads(completion.read_text())["epochs"] == 2


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
