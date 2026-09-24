# ECCO

Config-driven experiments for distributed roadside LiDAR perception in real
sunny, rainy, heavy-snow and freezing-rain recordings. This project works with
the extracted **CoInfra-I2I** layout, without an ego vehicle or synthetic RSUs.

See [Architecture and Mermaid diagrams](docs/architecture.md) for the data flow,
detector, fusion strategies, training, evaluation and edge/cloud timing model.

## Run

Python 3.10+ is recommended. On a GPU machine, install the appropriate PyTorch
build for your CUDA version before installing the remaining requirements.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
bash run_coinfra.sh --stage check
.venv/bin/python -m pytest -q
```

Edit `configs/coinfra.yaml`, especially `dataset.root`, `dataset.splits`, global
`bounds`, and `yaw_unit`. Alternatively, set `COINFRA_ROOT` to the extracted
dataset directory. Relative config paths resolve from this project directory.

Once the data is downloaded and extracted:

```bash
export COINFRA_ROOT=/path/to/extracted/CoInfra
bash run_coinfra.sh --stage inspect
bash run_coinfra.sh
```

For the smallest end-to-end check, use the new single entry point. Smoke mode
keeps train/validation/test recording sessions distinct, but uses only the first
session, slice and frame per split, two epochs, one seed and one node:

```bash
chmod +x run.sh
./run.sh --smoke
```

Smoke mode needs three distinct recording sessions to prevent train/validation/test
leakage. Download only their three archives with:

```bash
.venv/bin/python download_coinfra.py --smoke --download --output coinfra_downloads
```

Run the full configured experiment with `./run.sh`. Both commands write a complete
timestamped console log under `logs/` and a second pipeline log plus JSON completion
or failure status under the configured output directory. A traceback is preserved
when a run fails or is interrupted.

The last command runs inspect/index → train each seed → evaluate every configured
fusion/node subset → generate CSV summaries and a Markdown report. The script
does not download or extract data automatically. Run individual stages with
`--stage preprocess|train|evaluate|report`. Use `--config /path/to/config.yaml`
for another experiment, and `COINFRA_PYTHON` for another Python interpreter.
Existing compatible checkpoints are skipped; `--overwrite` explicitly retrains.
Interrupted training resumes from `latest.pt` at the next epoch when
`training.recovery.resume` is enabled. The recovery checkpoint includes model,
optimizer and random-number-generator state. Use `--overwrite` to restart training
(all configured seeds). Use a new output directory for a new
configuration. Completed seeds can be evaluated without training again.

## Download separately

```bash
.venv/bin/python download_coinfra.py --list
.venv/bin/python download_coinfra.py --download --output /path/to/archives
```

The downloader defaults to all archives, with a 400 GB compressed selection cap,
20 GB disk reserve, Hub resume, pinned revision and SHA-256 validation. The cap
does not include extraction, unrelated files or retries. Full archives are about
361 GB compressed; allow additional extraction and training space. `--subset`
selects the old eight-archive starter subset, which does NOT cover every session
in the full default experiment config. Change the split lists if using a subset.

Extract each `.tar.zst` with a zstd-capable tar into your data root. Avoid duplicate
scenario wrappers; the expected root immediately contains `2025_*` session folders.

## Data contract

```text
ROOT/2025_03_24_rainy/slice0/
  PcImage/1742850299.2/lidar/4_1742850299.199.npy
  Calibration/lidar/4.yaml
  GroundTruth/1742850299.2/global_roi.yaml
```

Both `slice0` and `slice_0` names are accepted. Calibration requires 4×4 matrices
`lidar_to_ground` and `ground_to_global`; their product maps each cloud to global
coordinates. Calibration is loaded per slice, never borrowed from another one.
Point arrays must be numeric Nx3 or Nx4+; column 4 is treated as intensity.
Nonfinite XYZ returns are discarded, nonfinite intensity is set to zero.

Annotations accept an object list or `objects`/`annotations`/`data` containing one.
Each object must provide class, position, dimensions (full L/W/H), and yaw.
Known field aliases follow the publisher baseline. Yaw units are explicit; zero
yaw is valid. Unknown labels, missing GT/calibration, duplicate nodes, excessive
timestamp offsets, and malformed arrays fail loudly. Global ROI labels are never
silently substituted with a different annotation set. GT and predictions are
scored in the same configured global region for every node subset.

The bounds come from the publisher baseline; `inspect` reports in-bounds point
and box counts on the first selected frame of each slice. Review `inspection.json`
before committing to training. Actual release schema and heading units still need
verification on your downloaded files. No real archive has been inspected here.

## Model and experiments

The included model is a small **trainable BEV center detector**, not PointPillars
or a reproduction of the CoInfra paper. It predicts class confidence, center,
dimensions and yaw from LiDAR density, maximum height and mean intensity rasters.
Batch size is one frame; node dropout samples real subsets during training.
Training mixes early, intermediate and single-node passes. Validation uses fixed
all-node early fusion loss, so checkpoint selection favors the early-fusion task.

- Early: merge global point clouds, then rasterize and detect.
- Intermediate: encode each global BEV with shared weights, max-fuse features,
  then decode detections.
- Late: detect independently at each node, then apply class-wise rotated 3D NMS.

No detector reads ground truth during inference. Global labels supervise training
and score evaluation only. Single-node models are scored against the same global
ROI, including objects they cannot see, making the task a coverage-sensitive test.
Dense centers colliding in a class/grid cell cause an explicit error; reduce the
grid resolution if this occurs. The model is intentionally a modest baseline;
its performance must be established experimentally before scientific claims.

The default splits are recording-session-disjoint: sunny March for training,
sunny November 18 for validation, sunny November 20 and adverse sessions for test.
This is **clear-to-adverse generalization**, not training on all weather. Rain and
freezing rain have one listed date each, so the config does not split adjacent
slices from those dates across training and test. Multiple weather sessions can
be assigned to training if you define a different valid experiment.

Evaluation uses all single nodes and deterministic sampled 2/4/8-node combinations
from nodes common to all selected frames within each session. It fails if requested
counts exceed availability. No extra RSUs are generated. Comparisons within each
session share timestamps and GT. Across sessions, node IDs and traffic can differ.

## Outputs and interpretation

Under `outputs/coinfra/`:

- `index.json`, `inventory.json`, `inspection.json`, `resolved_config.json`.
- `environment.json`: Python, platform, PyTorch and NumPy versions.
- `seed_N/best.pt` and `training.csv`.
- `seed_N/training_metrics_seed_N.csv` and `training_curves_seed_N.png`: timestamped
  epoch history with training/validation loss, validation mAP, recall, composite
  selection score, learning rate and elapsed time. The consolidated CSV is never
  pruned. Only the configured number of strongest `epoch_*.pt` checkpoints and
  their metric sidecars are retained; `best.pt` is the best composite checkpoint.
- `accuracy.csv`: per-session/subset/seed class AP and recall at rotated 3D IoU.
- `latency_communication.csv`: compressed payloads, measured inference/decode
  time, and modeled edge/cloud latency across configured bandwidths.
- `summary.csv`, `report.md`: descriptive aggregated results.
- `accuracy_by_nodes.png`: test AP trends across node counts and fusion strategies.
- `rsu_gains.csv`: multi-node AP minus mean/best single-node AP within each
  session, seed, strategy, class and IoU threshold. Best single is a retrospective
  comparison, not a deployment policy chosen without test labels.

AP ranks predictions across a complete session/subset, not an average of frame
AP. A class with no GT is undefined (blank), not zero. Summary averages over
sessions/subsets/seeds are descriptive and are not statistical confidence intervals.

Communication serializes actual raw point, encoded feature or detected-box arrays
as float32 and zlib level 1. It excludes protocol headers. Edge transmission sends
the strategy-specific representation; cloud transmission sends raw cropped points.
The analytical shared-link latency uses configured RTT and processing assumptions.
Measured inference/decode is serial on one machine and excludes rasterization,
I/O and serialization. It is not a measured distributed end-to-end deployment.
Cloud placement changes the transport model, not the accuracy predictions.

For the hypothesis, compare each multi-node result with the single-node results
under the same session and weather. Separate node-count gains from extra bandwidth;
these experiments report bytes but do not implement equal-byte feature selection.
Weather is confounded with recording date and traffic, so weather-stratified gains
are evidence of performance in those conditions, not a controlled causal weather
effect. Do not infer general deployment success from smoke-test results.

Sources: [CoInfra project](https://github.com/NingMingHao/CoInfra),
[dataset schema](https://github.com/NingMingHao/CoInfra/blob/main/docs/DATASET.md),
[publisher baseline](https://github.com/NingMingHao/CoInfra/blob/main/baselines/coop_lidar/dataset.py),
[downloads](https://huggingface.co/datasets/MinghaoNing/CoInfra).
