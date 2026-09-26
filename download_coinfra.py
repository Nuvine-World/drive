#!/usr/bin/env python3
"""Preview/download all CoInfra archives or a bounded subset."""

import argparse
import hashlib
import math
from pathlib import Path
import shutil
import sys

REPO = "MinghaoNing/CoInfra"
GB = 1_000_000_000
RECOMMENDED = [
    "2025_03_27_sunny_slice0-slice4.tar.zst",
    "2025_11_18_sunny_slice0-slice1.tar.zst",
    "2025_03_24_rainy_slice0-slice4.tar.zst",
    "2025_03_24_rainy_slice5-slice7.tar.zst",
    "2025_02_12_heavysnow_slice0-slice3.tar.zst",
    "2025_04_07_heavysnow_slice0-slice4.tar.zst",
    "2025_04_02_freezingrain_slice0-slice4.tar.zst",
    "2025_04_02_freezingrain_slice5-slice10.tar.zst",
]
SMOKE = [
    "2025_03_27_sunny_slice0-slice4.tar.zst",
    "2025_11_18_sunny_slice0-slice1.tar.zst",
    "2025_03_24_rainy_slice0-slice4.tar.zst",
]


def select_files(siblings, names, max_bytes):
    available = {f.rfilename: f for f in siblings}
    names = list(dict.fromkeys(names))
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError("Files not found in repository: " + ", ".join(missing))
    selected = [available[name] for name in names]
    for f in selected:
        p = Path(f.rfilename)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError("Unsafe repository filename")
        if f.size is None or f.size < 0:
            raise ValueError(
                f"Missing size for {f.rfilename}; refusing unbounded download"
            )
    total = sum(f.size for f in selected)
    if total > max_bytes:
        raise ValueError(
            f"Selection is {total / GB:.2f} GB, above the "
            f"{max_bytes / GB:.2f} GB cap. Select fewer files with --file."
        )
    return selected


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def expected_hash(f):
    lfs = getattr(f, "lfs", None)
    return lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)


def verified(path, f):
    if not path.is_file() or path.stat().st_size != f.size:
        return False
    checksum = expected_hash(f)
    if not checksum:
        raise ValueError(f"No publisher SHA-256 available for {f.rfilename}")
    return digest(path) == checksum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("coinfra_downloads"))
    parser.add_argument(
        "--max-gb",
        type=float,
        default=400,
        help="Maximum total selected compressed archive GB (decimal); default 400",
    )
    parser.add_argument(
        "--reserve-gb",
        type=float,
        default=20,
        help="Minimum free disk space after downloading; default 20 GB",
    )
    parser.add_argument(
        "--file",
        action="append",
        dest="files",
        help="Exact repository filename; repeat to replace default selection",
    )
    parser.add_argument(
        "--subset",
        action="store_true",
        help="Use the eight-archive starter subset instead of all archives",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Download only the three archives needed by --smoke pipeline mode",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="HF commit or branch; resolved to a pinned commit",
    )
    parser.add_argument(
        "--list", action="store_true", help="List all archive names and sizes only"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download after preview; otherwise metadata only",
    )
    args = parser.parse_args()
    if args.subset and args.smoke:
        parser.error("--subset and --smoke are mutually exclusive")
    if not math.isfinite(args.max_gb) or args.max_gb <= 0:
        parser.error("--max-gb must be a positive finite number")
    if not math.isfinite(args.reserve_gb) or args.reserve_gb < 0:
        parser.error("--reserve-gb must be a nonnegative finite number")
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        parser.exit(1, "Install dependency: python3 -m pip install huggingface_hub\n")

    info = HfApi().dataset_info(REPO, revision=args.revision, files_metadata=True)
    print(f"Source: https://huggingface.co/datasets/{REPO}")
    print(f"Pinned commit: {info.sha}")
    archives = [f for f in info.siblings if f.rfilename.endswith(".tar.zst")]
    if args.list:
        for f in sorted(archives, key=lambda f: f.rfilename):
            size = f"{f.size / GB:7.2f} GB" if f.size is not None else "unknown size"
            print(f"{size}  {f.rfilename}")
        return
    names = args.files or (
        SMOKE
        if args.smoke
        else RECOMMENDED if args.subset else [f.rfilename for f in archives]
    )
    if not names:
        raise ValueError("No dataset archives found")
    selected = select_files(archives, names, int(args.max_gb * GB))
    for f in selected:
        print(f"{f.size / GB:7.2f} GB  {f.rfilename}")
    total = sum(f.size for f in selected)
    print(f"Selected: {len(selected)} archives, {total / GB:.2f} GB compressed")
    print(
        "Cap covers selected archives, not extracted data, other files, or retry bandwidth."
    )
    if not args.download:
        print(
            "Preview only. Add --download to start. Archives are never auto-extracted."
        )
        return

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pending = []
    for f in selected:
        path = output / f.rfilename
        if not expected_hash(f):
            raise ValueError(
                f"No SHA-256 available for {f.rfilename}; cannot verify download"
            )
        if path.exists():
            print(f"Verifying existing {f.rfilename}", flush=True)
            if not verified(path, f):
                raise ValueError(
                    f"Existing archive is invalid: {path}. Move it aside and rerun."
                )
        else:
            pending.append(f)
    needed = sum(f.size for f in pending) + int(args.reserve_gb * GB)
    free = shutil.disk_usage(output).free
    if needed > free:
        raise ValueError(
            f"Need {needed / GB:.2f} GB free including reserve; "
            f"only {free / GB:.2f} GB available. Choose fewer archives or another disk."
        )
    for index, f in enumerate(pending, 1):
        print(f"Downloading {index}/{len(pending)}: {f.rfilename}", flush=True)
        path = Path(
            hf_hub_download(
                repo_id=REPO,
                repo_type="dataset",
                filename=f.rfilename,
                revision=info.sha,
                local_dir=str(output),
            )
        )
        print(f"Checking SHA-256: {f.rfilename}", flush=True)
        if not verified(path, f):
            raise ValueError(f"Checksum failed: {path}. Move it aside before retrying.")
    print(f"Done. Verified archives are in {output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Rerun the same command to resume.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        message = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        print(
            f"Error: {message}\nFor remote failures, check connectivity and dataset access; "
            "use `hf auth login` if authentication is required. Rerun to resume.",
            file=sys.stderr,
        )
        sys.exit(1)
