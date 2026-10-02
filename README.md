# ECCO on Adver-City and CoInfra

ECCO evaluates cooperative LiDAR perception with Adver-City or CoInfra-I2I.
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
- **Adaptive fusion:** retain one dense reference node and transmit only the
  highest-utility spatial cells from collaborator feature maps. A learned
  utility head is trained end-to-end with a straight-through top-k mask. One
  model is trained across several budgets and evaluated at 1%, 5%, 10%, and
  25% collaborator-cell budgets.

The same byte-budget sweep includes deterministic random-cell and feature-energy
selectors. These are ablations of the learned utility head, not separately
trained models, and appear as `adaptive_random@...` and `adaptive_energy@...`
in result files.

For adaptive fusion, the first node in the deterministic subset is the fusion
reference and therefore does not contribute to the transmitted collaborator
payload. Sparse messages contain flattened cell indices and feature vectors
serialized as configured integer and floating-point widths, then compressed
with zlib level 1. This reference-node assumption must be stated in papers and
should be tested with a reference-rotation ablation before a final submission.

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
- `latency_trace.csv` with per-frame loading, preprocessing, inference/decoding,
  serialization, and payload measurements;
- `latency_tail_summary.csv` with p50/p95/p99 and maximum latency, queue waiting,
  and deadline-miss rates under periodic and burst arrivals;
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

## Latency experiment

The latency trace separates measured local stages from network modeling. GPU
timing is synchronized around inference. The queue experiment replays the
measured per-frame service trace through a FIFO server at the configured frame
period and at 2x/4x burst arrival rates. For every placement, bandwidth, and
deadline it reports mean, p50, p95, p99, maximum, queue waiting, and deadline
miss rate. Transmission and RTT remain modeled unless an external network
testbed is used. Run the final latency experiment with evaluation batch size 1
so batch-amortized throughput is not reported as per-request response time.

## Scientific scope

The experiment supports claims about multi-RSU cooperation on CoInfra-I2I. It
does not provide real edge/cloud deployment measurements, causal weather effects,
vehicle-to-infrastructure fusion, or guarantees about production deployment.
Smoke runs and synthetic tests validate software plumbing, not model quality.

Dataset resources: [project](https://github.com/NingMingHao/CoInfra),
[schema](https://github.com/NingMingHao/CoInfra/blob/main/docs/DATASET.md), and
[archives](https://huggingface.co/datasets/MinghaoNing/CoInfra).
