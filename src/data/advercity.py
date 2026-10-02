from pathlib import Path
import math
import warnings
import numpy as np
import yaml
from src.data.coinfra import first, in_bounds, vector
from src.data.loader import BoundingBox3D


WEATHER = {
    'cd': 'clear', 'cn': 'clear_night', 'srd': 'soft_rain',
    'srn': 'soft_rain_night', 'hrd': 'heavy_rain',
    'hrn': 'heavy_rain_night', 'fd': 'fog', 'fn': 'fog_night',
    'fhrd': 'fog_heavy_rain', 'fhrn': 'fog_heavy_rain_night',
    'gd': 'glare'
}


def scenario_weather(name):
    parts = name.split('_')
    if len(parts) != 3 or parts[1] not in WEATHER:
        raise ValueError(f'Invalid Adver-City scenario name: {name}')
    return WEATHER[parts[1]]


def annotations(path, cfg):
    """Read Adver-City labels while isolating corrupt publisher objects.

    A small number of source frames contain NaN/Inf poses or incomplete objects.
    Dropping only those objects keeps the frame usable and is preferable to aborting
    a multi-day run.  The strict CoInfra reader remains unchanged.
    """
    with open(path) as stream:
        loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
        raw = yaml.load(stream, Loader=loader)
    if isinstance(raw, dict):
        raw = first(raw, "objects", "annotations", "data")
    if not isinstance(raw, list):
        raise ValueError(f"Expected annotation list in {path}")
    boxes = []
    rejected = []
    for index, obj in enumerate(raw):
        try:
            label = str(first(obj, "label", "class", "type")).lower()
            label = cfg["class_aliases"].get(label, label)
            if label not in cfg["classes"]:
                raise ValueError(f"unknown label {label!r}")
            center = vector(first(obj, "center", "position", "translation"))
            dimensions = vector(first(obj, "size", "dimensions"), size=True)
            yaw = first(obj, "yaw", "heading", "rotation")
            if isinstance(yaw, dict):
                yaw = first(yaw, "z", "yaw", "heading", "0")
            yaw = float(yaw)
            if not math.isfinite(yaw) or (dimensions <= 0).any():
                raise ValueError("non-finite yaw or non-positive dimensions")
            if cfg["yaw_unit"] == "degrees":
                yaw = math.radians(yaw)
            box = BoundingBox3D(*center, *dimensions, yaw, label)
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            rejected.append((index, str(error)))
            continue
        if in_bounds(box.x, box.y, box.z, cfg["bounds"]):
            boxes.append(box)
    if rejected:
        preview = "; ".join(f"#{index}: {error}" for index, error in rejected[:3])
        warnings.warn(
            f"Skipped {len(rejected)} invalid Adver-City annotations in {path} "
            f"({preview})",
            RuntimeWarning,
        )
    return boxes


def discover(cfg):
    d = cfg['dataset']
    root = Path(d['root'])
    if not root.is_dir():
        raise FileNotFoundError(f'Prepared Adver-City directory not found: {root}')
    samples = []
    smoke = d.get('smoke', {'enabled': False})
    for split, scenarios in d['splits'].items():
        if smoke.get('enabled'):
            scenarios = scenarios[:smoke['sessions_per_split']]
        for scenario in scenarios:
            weather = scenario_weather(scenario)
            if weather not in d['weather']:
                continue
            scenario_dir = root / scenario
            metadata = scenario_dir / 'scenario.yaml'
            if not metadata.is_file():
                raise FileNotFoundError(f'Missing prepared scenario: {scenario_dir}')
            frames = sorted((scenario_dir / 'frames').glob('*'))[::d['frame_stride']]
            if d.get('max_frames_per_slice') is not None:
                frames = frames[:d['max_frames_per_slice']]
            if smoke.get('enabled'):
                frames = frames[:smoke['frames_per_slice']]
            count = 0
            for frame in frames:
                gt = frame / 'gt.yaml'
                node_files = sorted((frame / 'nodes').glob('*.npy'))
                nodes = {p.stem: {'cloud': str(p), 'dataset_type': 'advercity'} for p in node_files}
                selected = {str(n) for n in d.get('nodes', [])}
                if selected:
                    nodes = {k: v for k, v in nodes.items() if k in selected}
                if not gt.is_file() or not nodes:
                    raise ValueError(f'Incomplete prepared frame: {frame}')
                samples.append({
                    'id': f'{scenario}/{frame.name}', 'session': scenario,
                    'slice': scenario, 'timestamp': frame.name, 'split': split,
                    'weather': weather, 'gt': str(gt),
                    'nodes': nodes
                })
                count += 1
            if not count:
                raise ValueError(f'No frames found in {scenario_dir}')
    return samples


def load_cloud(node, bounds):
    cloud = np.load(node['cloud'], allow_pickle=False)
    if cloud.ndim != 2 or cloud.shape[1] != 4:
        raise ValueError(f'Expected Nx4 prepared cloud: {node["cloud"]}')
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    mask = ((cloud[:, :3] >= bounds[:3]) & (cloud[:, :3] < bounds[3:])).all(axis=1)
    return cloud[mask].astype(np.float32, copy=False)
