"""CoInfra I2I reader. Fail on missing calibration/labels instead of guessing."""
from pathlib import Path
from functools import lru_cache
import math
import numpy as np
import yaml
from src.data.loader import BoundingBox3D


def read_yaml(path):
    with open(path) as stream:
        loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
        return yaml.load(stream, Loader=loader)


def first(d, *keys):
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    raise ValueError(f'Missing one of {keys} in annotation')


def vector(value, size=False):
    if isinstance(value, dict):
        options = [('length', 'width', 'height'), ('l', 'w', 'h')] if size else []
        options += [('x', 'y', 'z'), ('0', '1', '2'), (0, 1, 2)]
        for keys in options:
            if all(k in value for k in keys):
                value = [value[k] for k in keys]
                break
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        raise ValueError('Expected finite 3-vector')
    return arr


def in_bounds(x, y, z, b):
    return b[0] <= x < b[3] and b[1] <= y < b[4] and b[2] <= z < b[5]


def annotations(path, cfg):
    raw = read_yaml(path)
    if isinstance(raw, dict):
        raw = first(raw, 'objects', 'annotations', 'data')
    if not isinstance(raw, list):
        raise ValueError(f'Expected annotation list in {path}')
    boxes = []
    for obj in raw:
        label = str(first(obj, 'label', 'class', 'type')).lower()
        label = cfg['class_aliases'].get(label, label)
        if label not in cfg['classes']:
            raise ValueError(f'Unknown label {label!r} in {path}; configure class_aliases')
        center = vector(first(obj, 'center', 'position', 'translation'))
        dimensions = vector(first(obj, 'size', 'dimensions'), size=True)
        yaw = first(obj, 'yaw', 'heading', 'rotation')
        if isinstance(yaw, dict):
            yaw = first(yaw, 'z', 'yaw', 'heading', '0')
        yaw = float(yaw)
        if not math.isfinite(yaw) or (dimensions <= 0).any():
            raise ValueError(f'Invalid box in {path}')
        if cfg['yaw_unit'] == 'degrees':
            yaw = math.radians(yaw)
        boxes.append(BoundingBox3D(*center, *dimensions, yaw, label))
    return [b for b in boxes if in_bounds(b.x, b.y, b.z, cfg['bounds'])]


def calibration(path):
    path = Path(path)
    stat = path.stat()
    return _calibration_cached(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=None)
def _calibration_cached(path, _mtime_ns, _size):
    raw = read_yaml(path)
    matrices = []
    for key in ('lidar_to_ground', 'ground_to_global'):
        m = np.asarray(raw[key], dtype=float)
        if m.shape != (4, 4) or not np.isfinite(m).all() or not np.allclose(m[3], [0, 0, 0, 1]):
            raise ValueError(f'Invalid {key} in {path}')
        if not np.allclose(m[:3, :3].T @ m[:3, :3], np.eye(3), atol=1e-3):
            raise ValueError(f'Non-rigid rotation in {path}')
        matrices.append(m)
    return matrices[1] @ matrices[0]


def discover(cfg):
    d, samples = cfg['dataset'], []
    smoke = d.get('smoke', {'enabled': False})
    root = Path(d['root'])
    if not root.is_dir():
        raise FileNotFoundError(f'Extract CoInfra into {root} or set COINFRA_ROOT')
    for split, sessions in d['splits'].items():
        if smoke.get('enabled'):
            sessions = sessions[:smoke['sessions_per_split']]
        for session in sessions:
            scenario = root / session
            if not scenario.is_dir():
                raise FileNotFoundError(f'Missing configured session: {scenario}')
            weather = session.split('_', 3)[-1]
            if weather not in d['weather']:
                continue
            session_count = 0
            slice_dirs = sorted(scenario.glob('slice*'))
            if smoke.get('enabled'):
                slice_dirs = slice_dirs[:smoke['slices_per_session']]
            for slice_dir in slice_dirs:
                frames = []
                for candidate in (slice_dir / "PcImage").iterdir():
                    try:
                        timestamp = float(candidate.name)
                    except ValueError:
                        continue
                    if candidate.is_dir():
                        frames.append((timestamp, candidate))
                frames = [candidate for _, candidate in sorted(frames)]
                frames = frames[::d['frame_stride']]
                if d.get('max_frames_per_slice') is not None:
                    frames = frames[:d['max_frames_per_slice']]
                if smoke.get('enabled'):
                    frames = frames[:smoke['frames_per_slice']]
                for frame in frames:
                    gt = slice_dir / 'GroundTruth' / frame.name / d['annotation_file']
                    if not gt.is_file():
                        raise FileNotFoundError(f'Missing fixed global ground truth: {gt}')
                    nodes = {}
                    for cloud in sorted((frame / 'lidar').glob('*.npy')):
                        node, timestamp = cloud.stem.split('_', 1)
                        if d['nodes'] and node not in {str(n) for n in d['nodes']}:
                            continue
                        if abs(float(timestamp) - float(frame.name)) > d['max_timestamp_delta_s']:
                            raise ValueError(f'Timestamp exceeds synchronization tolerance: {cloud}')
                        if node in nodes:
                            raise ValueError(f'Duplicate node {node} at {frame}')
                        calib = slice_dir / 'Calibration' / 'lidar' / f'{node}.yaml'
                        calibration(calib)
                        nodes[node] = {'cloud': str(cloud), 'calibration': str(calib)}
                    if not nodes:
                        raise ValueError(f'No selected LiDAR nodes at {frame}')
                    samples.append({'id': f'{session}/{slice_dir.name}/{frame.name}',
                                    'session': session, 'slice': slice_dir.name, 'timestamp': frame.name,
                                    'split': split, 'weather': weather, 'gt': str(gt), 'nodes': nodes})
                    session_count += 1
            if not session_count:
                raise ValueError(f'No frames found in {session}')
    if not samples:
        raise ValueError('No samples matched the configuration')
    return samples


def load_cloud(node, bounds):
    cloud = np.load(node['cloud'], allow_pickle=False)
    if cloud.ndim != 2 or cloud.shape[1] < 3:
        raise ValueError(f'Expected Nx3 or Nx4+ array: {node["cloud"]}')
    cloud = cloud[np.isfinite(cloud[:, :3]).all(axis=1)]
    transform = calibration(node['calibration'])
    xyz = cloud[:, :3] @ transform[:3, :3].T + transform[:3, 3]
    intensity = np.nan_to_num(cloud[:, 3], nan=0, posinf=0, neginf=0) if cloud.shape[1] > 3 else np.ones(len(cloud))
    mask = ((xyz >= bounds[:3]) & (xyz < bounds[3:])).all(axis=1)
    return np.column_stack((xyz[mask], intensity[mask])).astype(np.float32)


def rasterize(points, cfg):
    b, r = cfg['bounds'], cfg['resolution']
    w, h = int(np.ceil((b[3]-b[0])/r)), int(np.ceil((b[4]-b[1])/r))
    bev = np.zeros((3, h, w), dtype=np.float32)
    if not len(points):
        return bev
    x, y = ((points[:, 0]-b[0])/r).astype(int), ((points[:, 1]-b[1])/r).astype(int)
    valid = (x>=0) & (x<w) & (y>=0) & (y<h)
    x, y, points = x[valid], y[valid], points[valid]
    np.add.at(bev[0], (y, x), 1)
    np.maximum.at(bev[1], (y, x), (points[:, 2]-b[2])/(b[5]-b[2]))
    np.add.at(bev[2], (y, x), points[:, 3])
    bev[2] /= np.maximum(bev[0], 1)
    bev[0] = np.log1p(bev[0])
    return bev
