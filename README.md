# ECCO on Adver-City and CoInfra

ECCO evaluates cooperative LiDAR perception with Adver-City or CoInfra-I2I.
The default server configuration uses an ego vehicle, connected vehicles, and
roadside units from selected Adver-City clear, rain, fog, and fog-plus-heavy-rain
scenarios. CoInfra remains available for real multi-RSU adverse-weather tests.

The project compares early, intermediate, and late fusion for 1, 2, 4, and 8
RSUs. It measures rotated 3D AP, precision, recall, F1, TP/FP/FN counts,
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
a validated converted result. A system `7z` or `7zz` executable is used when
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
Verify the dataset bounds, heading units, split sessions, and the source and
values under `deployment_assumptions`.

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
./run.sh --config configs/coinfra.yaml
```

The runner performs inspection, indexing, training, held-out evaluation, and
report generation. Timestamped console logs and completion/failure status files
are written under `logs/`, including Python tracebacks from failed runs.
Runtime dependencies are imported before acquisition or training begins. Dataset
inspection parses annotations in every selected frame and verifies that each test
session has enough common nodes for the largest requested evaluation count.

Training resumes from `seed_N/latest.pt` after the last completed epoch. Recovery
checkpoints include model, optimizer, metric history, and random states. The best
checkpoint is selected using validation loss quality, mAP, and recall across all
configured fusion strategies. Use `--overwrite` to restart configured seeds.

Individual stages are available:

```bash
./run.sh --config configs/coinfra.yaml --stage preprocess
./run.sh --config configs/coinfra.yaml --stage train
./run.sh --config configs/coinfra.yaml --stage evaluate
./run.sh --config configs/coinfra.yaml --stage report
```

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

Each cloud is transformed from its node frame to the shared global frame,
cropped to one fixed region, and rasterized as point density, maximum height,
and mean intensity. Ground-truth boxes use the same region. Missing calibration,
labels, frames, or excessive timestamp mismatch stops the run.

## Model and fusion

The included model is a compact trainable BEV center detector, not a reproduction
of the publisher's detector. It predicts car, truck, bus, bicycle, and pedestrian.

- **Early fusion:** combine globally aligned points before rasterization.
- **Intermediate fusion:** encode each RSU BEV and max-fuse its feature map.
- **Late fusion:** detect per RSU and merge boxes with class-wise rotated 3D NMS.

All strategies use the same weights and evaluation frames. Every single common
RSU is evaluated; larger node combinations are sampled deterministically. The
fixed global ground truth is unchanged between node subsets.

## Outputs

The configured output directory contains:

- `index.json`, `inventory.json`, `inspection.json`, and `resolved_config.json`;
- `environment.json` with software and platform versions;
- `seed_N/best.pt`, `latest.pt`, retained epoch checkpoints, and completion state;
- `seed_N/training_metrics_seed_N.csv` and training curves;
- `accuracy.csv` with per-session, subset, seed, class, and IoU metrics;
- `accuracy_summary.csv` with per-class accuracy and localization summaries;
- `metrics_overview.csv` with macro/micro AP, precision, recall, F1, counts, and localization quality;
- `cooperation_gains.csv` with paired multi-RSU gains over mean/best single RSUs;
- `weather_cooperation_summary.csv` with weather-stratified paired gains;
- `node_scaling.csv` with changes between successive configured node counts;
- `latency_communication.csv` with measured local timing and assumed latency;
- `accuracy_latency_tradeoffs.csv` joining accuracy and deployment costs;
- `accuracy_by_nodes.png`, `cooperation_gain_by_weather.png`,
  `latency_by_bandwidth.png`, and `report.md`;
- `verification.json`, written only after required artifacts and metric invariants are checked.

AP is confidence-ranked across a complete session/subset. Precision, recall, and
F1 use `evaluation.score_threshold` after class-wise NMS. Localization errors are
computed only for matched true positives. Undefined classes and match metrics are
left blank rather than treated as zero. Summary statistics are descriptive, not
confidence intervals. `run.sh` executes verification after report generation and
returns a nonzero exit code if required results are missing or inconsistent.

## Scientific scope

The experiment supports claims about multi-RSU cooperation on CoInfra-I2I. It
does not provide real edge/cloud deployment measurements, causal weather effects,
vehicle-to-infrastructure fusion, or guarantees about production deployment.
Smoke runs and synthetic tests validate software plumbing, not model quality.

Dataset resources: [project](https://github.com/NingMingHao/CoInfra),
[schema](https://github.com/NingMingHao/CoInfra/blob/main/docs/DATASET.md), and
[archives](https://huggingface.co/datasets/MinghaoNing/CoInfra).
