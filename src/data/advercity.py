from pathlib import Path
import numpy as np
from src.data.coinfra import annotations as normalized_annotations, in_bounds


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
    return normalized_annotations(path, cfg)


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
