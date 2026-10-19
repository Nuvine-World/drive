# DRIVE

DRIVE evaluates cooperative LiDAR perception with Adver-City or CoInfra-I2I.
With no explicit configuration, `run.sh` executes CoInfra followed by Adver-City.
Passing either dataset configuration runs only that dataset.

The project compares early, intermediate, late, and budget-conditioned adaptive
sparse fusion for 1, 2, 4, and 8 RSUs. It measures rotated 3D AP, precision,
recall, F1, TP/FP/FN counts,
matched-box 3D IoU, center/yaw/size errors, representation size, local inference
time, and analytical edge/cloud latency. Edge/cloud deployment is not executed;
latency uses explicit configuration assumptions that must be cited in a paper.

CoInfra is infrastructure-to-infrastructure perception. Adver-City contributes
vehicle and roadside sensor viewpoints in simulation.

## Research questions

1. Does multi-RSU cooperation improve AP and recall over individual RSUs?
2. How does the benefit change with node count and fusion strategy?
3. What accuracy, communication, and assumed-latency trade-offs result?
4. Does the paired cooperation gain persist across recorded weather conditions?
5. Which object classes benefit most?
6. Can learned utility-per-byte selection preserve cooperative accuracy while
   transmitting only 1--25% of collaborator BEV cells?

Weather comparisons are observational. Recording date, traffic, lighting, and
weather are confounded, so results do not identify a causal weather effect.

## Setup

Python 3.10+ is recommended. Install a CUDA-compatible PyTorch build on the GPU
machine before installing the remaining dependencies.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

Optional local commit hook:

```bash
git config core.hooksPath githooks
```

## Adver-City server run

The default configuration selects 25 daytime scenarios covering clear weather,
soft rain, heavy rain, fog, and fog with heavy rain. The official archives total
310.85 GB over the complete sequential transfer, but only one archive is staged
at a time. Camera images are not extracted. LiDAR, annotations, poses, and
scenario metadata are converted to a compact ego-centered representation, then
the verified source archive and extraction directory are removed.

Set the persistent and temporary locations on the server and run:

```bash
export ADVERCITY_ROOT=/path/to/persistent/advercity
export ADVERCITY_WORK_DIR=/path/to/temporary/advercity_work
export ECCO_PYTHON=/path/to/venv/bin/python
./run.sh --config configs/advercity.yaml
```

The process resumes partial HTTP downloads and skips scenarios that already have
a validated converted result. Each frame's labels merge the vehicle and walker
lists of every agent (keyed by actor id), because an agent's own file need not
list itself or objects outside its range. Scenarios converted before this
change (`format: 1` in `scenario.yaml`) hold ego-only labels and stop the run;
delete them to re-prepare (this re-downloads their archives), or set
`dataset.acquisition.allow_legacy_labels: true` to use them as they are. A system `7z` or `7zz` executable is used when
available; otherwise `py7zr` from `requirements.txt` is used. To inspect the
official selection and transfer sizes without downloading archives:

```bash
.venv/bin/python prepare_advercity.py --config configs/advercity.yaml --preview
```

For an end-to-end one-frame test profile:

```bash
./run.sh --small --config configs/advercity.yaml
```

`--small` is the recommended whole-flow preflight when prepared data already
exists. It checks that every smoke-profile session is present, skips all download
and conversion work, replaces any earlier smoke checkpoints, then runs inspection,
indexing, two training epochs, evaluation, plots, reporting, and final verification.
Its split-disjoint metrics validate plumbing only and are not research results.

Do not interrupt the process during the final scenario rename. Failed downloads
remain as `.part` files and resume on the next invocation. An archive is deleted
only after checksum verification, conversion, and converted-data validation.

## CoInfra server run

Point `COINFRA_ROOT` at the extracted directory whose immediate children are
the `2025_*` recording sessions:

```bash
export COINFRA_ROOT="/path/to/CoInfra"
./run.sh --config configs/coinfra.yaml --stage check
./run.sh --config configs/coinfra.yaml --stage inspect
```

Edit [configs/coinfra.yaml](configs/coinfra.yaml) before a full experiment.
Verify the dataset bounds, heading units, split sessions, `use_intensity`/
`intensity_max`, and the source and values under `deployment_assumptions`.
Inspection records point intensity percentiles in `inspection.json`. When
`dataset.use_intensity` is true it stops if intensity is constant (CoInfra's is,
so its configs set `use_intensity: false`) or if the p99 intensity does not fit
`dataset.intensity_max` (above it, or below 1% of it). It also warns about
classes that have test but no validation ground truth; such classes are unseen
by checkpoint selection and threshold tuning (CoInfra validation has no
bicycles).

Training uses two sessions (2025_03_27_sunny and 2025_02_12_heavysnow);
validation uses 2025_11_18_sunny; every weather condition keeps at least one
held-out test session. Rainy and freezing-rain conditions are unseen in
training, which matters when comparing results across weather.

## Small end-to-end check

Smoke mode keeps train, validation, and test sessions separate but uses one
session, one slice, one frame, one seed, two epochs, and one RSU per split. It
validates the pipeline; its metrics are not research results.

```bash
.venv/bin/python download_coinfra.py --smoke --download --output coinfra_downloads
./run.sh --small --config configs/coinfra.yaml
```

The three smoke archives are approximately 32 GB compressed. Extract them into
`COINFRA_ROOT` before running the pipeline.

## Full experiment

```bash
./preflight.sh
nohup ./run.sh >> output.txt 2>&1 < /dev/null &
```

With no `--config`, `run.sh` runs CoInfra first and starts Adver-City only after
CoInfra exits successfully. If CoInfra fails, the sequence exits with the same
nonzero status and Adver-City is not started. `preflight.sh` runs the unit tests,
checks both configurations and runtime dependencies, verifies CUDA availability,
and requires the externally downloaded CoInfra data. Adver-City data is not
required by the preflight because its full run prepares and validates that data.

To run only one dataset, continue to pass its configuration explicitly:

```bash
./run.sh --config configs/coinfra.yaml
./run.sh --config configs/advercity.yaml
```

The runner performs inspection, indexing, training, held-out evaluation, and
report generation. Timestamped console logs and completion/failure status files
are written under `logs/`, including Python tracebacks from failed runs.
Runtime dependencies are imported before acquisition or training begins. Dataset
inspection parses annotations in every selected frame and verifies that each test
session has enough common nodes for the largest requested evaluation count.

Training resumes from `seed_N/latest.pt` after the last completed epoch. Recovery
checkpoints include model, optimizer, metric history, and random states. The best
checkpoint is selected using validation loss quality, mAP, and recall at the
primary IoU level. Validation scores every frame twice: with all of its nodes
under each configured fusion strategy, and with one deterministic single node.
The two views carry equal weight, so checkpoints are not chosen for multi-node
quality alone. Validation loss is averaged per sample over both views. Use
`--overwrite` to restart configured seeds. `run.sh` retries exit code 75
(transient failures) without `--overwrite`, so a retry resumes rather than
discarding checkpoints; `ECCO_RETRY_DELAY_SECONDS` sets the base retry delay.

Individual stages are available:

```bash
./run.sh --config configs/coinfra.yaml --stage preprocess
./run.sh --config configs/coinfra.yaml --stage train
./run.sh --config configs/coinfra.yaml --stage evaluate
./run.sh --config configs/coinfra.yaml --stage deadline
./run.sh --config configs/coinfra.yaml --stage report
```

`configs/coinfra.yaml` holds out whole recording sessions.
`configs/coinfra_random_split.yaml` reproduces the CoInfra paper's protocol,
which splits frames at random, so its numbers can be compared with the
published table. The `deadline` stage replays consecutive 10 Hz test frames
through per-frame communication controllers over modeled uplinks (see
`docs/decisions/2026-10-11-drive-detector.md`).

## Dataset download

```bash
.venv/bin/python download_coinfra.py --list
.venv/bin/python download_coinfra.py --download --output /path/to/archives
.venv/bin/python download_coinfra.py --subset --download --output /path/to/archives
.venv/bin/python download_coinfra.py \
  --file "2025_11_20_sunny_slice10-slice10.tar.zst" \
  --download --output /path/to/archives
```

The downloader pins the Hub revision, enforces size and free-space limits, and
verifies publisher SHA-256 hashes. It does not extract archives.

## Data contract

```text
COINFRA_ROOT/
  2025_03_24_rainy/
    slice0/
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
