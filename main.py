"""Command-line entry point for the ECCO experiment pipeline."""

import argparse
import fcntl
import os
from pathlib import Path
import sys
import traceback

from src.config import PROJECT, enable_overfit, enable_smoke, load_config

STAGES = (
    "check",
    "prepare",
    "inspect",
    "preprocess",
    "train",
    "evaluate",
    "deadline",
    "report",
    "verify",
    "all",
)


def check_runtime(cfg):
    """Import runtime dependencies before downloads or long-running work."""
    import numpy  # noqa: F401
    import shapely  # noqa: F401
    import torch  # noqa: F401

    if not hasattr(torch.amp, "GradScaler"):
        raise RuntimeError(
            f"Installed torch {torch.__version__} lacks torch.amp.GradScaler"
        )
    if not hasattr(torch, "OutOfMemoryError"):
        raise RuntimeError(
            f"Installed torch {torch.__version__} lacks torch.OutOfMemoryError"
        )
    if cfg["training"].get("require_cuda", False) and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required by training.require_cuda, but torch.cuda.is_available() is false. "
            f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda!r}"
        )
    if cfg["output"]["plots"]:
        import matplotlib  # noqa: F401
    if cfg['dataset'].get('type') == 'advercity':
        import py7zr  # noqa: F401


def require_existing_data(cfg):
    """Reject missing prepared data before a run that must not acquire anything."""
    root = Path(cfg['dataset']['root'])
    if not root.is_dir():
        raise FileNotFoundError(f'Prepared dataset directory does not exist: {root}')
    missing = []
    for sessions in cfg['dataset']['splits'].values():
        for session in sessions:
            directory = root/session
            if cfg['dataset'].get('type') == 'advercity':
                present = (directory/'scenario.yaml').is_file() and (directory/'frames').is_dir()
            else:
                present = directory.is_dir() and any(directory.glob('slice*'))
            if not present:
                missing.append(str(directory))
    if missing:
        preview = ', '.join(missing[:3])
        suffix = f' (and {len(missing)-3} more)' if len(missing) > 3 else ''
        raise FileNotFoundError(f'Prepared dataset is incomplete; missing: {preview}{suffix}')
    print(f'Prepared dataset check passed: {root}', flush=True)


def run_pipeline(args, cfg):
    skip_prepare = getattr(args, 'skip_prepare', False)
    if args.stage == 'check':
        print('Configuration valid.')
        print(f'Dataset: {cfg["dataset"]["root"]}')
        print(f'Output: {cfg["output"]["root"]}')
        if not Path(cfg['dataset']['root']).is_dir():
            print('Dataset not present yet; run --stage inspect after extraction.')
        return

    if args.stage == 'prepare' and skip_prepare:
        raise ValueError('--skip-prepare cannot be combined with --stage prepare')
    if args.stage == 'prepare' and cfg['dataset'].get('type') != 'advercity':
        raise ValueError('--stage prepare is only supported for Adver-City; '
                         'download and extract CoInfra before running --stage inspect')
    if (args.stage in ('prepare', 'all') and cfg['dataset'].get('type') == 'advercity'
            and not skip_prepare):
        from prepare_advercity import prepare
        prepare(cfg)
        if args.stage == 'prepare':
            return

    from src.pipeline import (
        inspect,
        preprocess,
        report,
        run_deadline_evaluation,
        run_evaluation,
        train,
        verify_results,
    )

    if args.stage in ('inspect', 'all'):
        inspect(cfg)
    elif args.stage == 'preprocess':
        preprocess(cfg)
    if args.stage in ('train', 'all'):
        train(cfg, args.overwrite)
    if args.stage in ('evaluate', 'all'):
        run_evaluation(cfg)
    if args.stage in ("deadline", "all"):
        run_deadline_evaluation(cfg)
    if args.stage in ('report', 'all'):
        report(cfg)
    if args.stage in ('verify', 'all'):
        verify_results(cfg)


def acquire_run_lock(cfg, stage):
    if stage == "check":
        return None
    output = Path(cfg["output"]["root"])
    output.mkdir(parents=True, exist_ok=True)
    stream = (output / ".lock").open("a+")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        raise ValueError(
            f"Another ECCO process is already using output directory {output}"
        )
    stream.seek(0)
    stream.truncate()
    stream.write(f"pid={os.getpid()}\n")
    stream.flush()
    return stream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT / "configs/coinfra.yaml"))
    parser.add_argument("--stage", choices=STAGES, default="check")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="restart training and replace checkpoints",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="use the smallest split-disjoint end-to-end profile",
    )
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="train on and validate against 32 identical frames as a learning diagnostic",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="never download or convert data; use an existing prepared dataset",
    )
    parser.add_argument(
        "--require-data",
        action="store_true",
        help="check all configured prepared sessions before starting",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.smoke and args.overfit:
        raise ValueError("--smoke and --overfit cannot be combined")
    if args.smoke and not cfg['dataset']['smoke']['enabled']:
        enable_smoke(cfg)
    if args.overfit:
        enable_overfit(cfg)
    check_runtime(cfg)
    if args.require_data:
        require_existing_data(cfg)
    lock = acquire_run_lock(cfg, args.stage)
    try:
        run_pipeline(args, cfg)
    finally:
        if lock is not None:
            lock.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        print(
            f"Fatal configuration or data error: {error}", file=sys.stderr, flush=True
        )
        raise SystemExit(2)
    except RuntimeError as error:
        traceback.print_exc()
        print(f"Fatal runtime error: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
    except (TimeoutError, OSError):
        traceback.print_exc()
        print(
            "Transient runtime failure; safe to retry without --overwrite.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(75)
