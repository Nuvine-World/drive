import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import urllib.parse
import urllib.request
import warnings
from collections.abc import Mapping
from pathlib import Path
import numpy as np
import yaml


ITEM_ID = 1079
COLLECTION_ID = 2
ENDPOINT_ID = 'f163c1b3-9c88-42f6-a7bb-5839ed6c4063'
REMOTE_ROOT = '/1/published/publication_1079/submitted_data'
LIST_URL = 'https://www.frdr-dfdr.ca/repo/userfiles'
FILE_URL = 'https://www.frdr-dfdr.ca/repo/files/1/published/publication_1079/submitted_data'
CHECKSUM_URL = FILE_URL + '/frdr-dfdr-checksums.txt'


def request(url, headers=None):
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=120)


def remote_files():
    files = {}
    offset = 0
    while True:
        query = urllib.parse.urlencode({
            'item_id': ITEM_ID, 'collection_id': COLLECTION_ID,
            'endpoint_id': ENDPOINT_ID, 'path': REMOTE_ROOT,
            'limit': 100, 'offset': offset
        })
        with request(f'{LIST_URL}?{query}') as response:
            value = json.load(response)
        batch = value.get('files', [])
        for item in batch:
            name = Path(item['name']).name
            if name.endswith('.7z') and name != 'stats.7z':
                files[name[:-3]] = {'name': name, 'size': int(item['size'])}
        if len(batch) < 100:
            break
        offset += len(batch)
    return files


def remote_checksums():
    with request(CHECKSUM_URL) as response:
        text = response.read().decode()
    result = {}
    name = None
    for line in text.splitlines():
        if line.startswith('Filename/Fichier: ./'):
            name = line.split('./', 1)[1].strip()
        elif name and line.startswith('SHA256 Checksum/Sommes de contrôle:'):
            result[name] = line.rsplit(':', 1)[1].strip()
            name = None
    return result


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def ensure_space(path, required, reserve):
    free = shutil.disk_usage(path).free
    if free < required + reserve:
        raise OSError(f'Insufficient disk space: free={free/1e9:.1f}GB, '
                      f'required={required/1e9:.1f}GB, reserve={reserve/1e9:.1f}GB')


def download(url, destination, size):
    part = destination.with_suffix(destination.suffix + '.part')
    offset = part.stat().st_size if part.exists() else 0
    if offset > size:
        part.unlink()
        offset = 0
    headers = {'Range': f'bytes={offset}-'} if offset else {}
    with request(url, headers) as response:
        if offset and response.status != 206:
            part.unlink()
            offset = 0
            response.close()
            return download(url, destination, size)
        mode = 'ab' if offset else 'wb'
        with part.open(mode) as stream:
            received = offset
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                stream.write(block)
                received += len(block)
                print(f'\r{destination.name}: {received/1e9:.2f}/{size/1e9:.2f} GB', end='', flush=True)
    print()
    if part.stat().st_size != size:
        raise IOError(f'Incomplete download: {part.stat().st_size} != {size}')
    part.replace(destination)


def extract_archive(archive, destination):
    executable = shutil.which('7zz') or shutil.which('7z')
    if executable:
        subprocess.run([
            executable, 'x', str(archive), f'-o{destination}', '-y',
            '-xr!*.png', '-xr!*.jpg', '-xr!*.jpeg'
        ], check=True)
        return
    try:
        import py7zr
    except ImportError as error:
        raise RuntimeError('Install py7zr or the 7z/7zz executable') from error
    with py7zr.SevenZipFile(archive) as bundle:
        names = [n for n in bundle.getnames() if Path(n).suffix.lower() in ('.ply', '.yaml', '.yml')]
        bundle.extract(path=destination, targets=names)


PLY_TYPES = {
    'char': 'i1', 'uchar': 'u1', 'int8': 'i1', 'uint8': 'u1',
    'short': 'i2', 'ushort': 'u2', 'int16': 'i2', 'uint16': 'u2',
    'int': 'i4', 'uint': 'u4', 'int32': 'i4', 'uint32': 'u4',
    'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8'
}


def read_ply(path):
    with path.open('rb') as stream:
        if stream.readline().strip() != b'ply':
            raise ValueError(f'Invalid PLY file: {path}')
        fmt = None
        count = None
        properties = []
        vertex = False
        while True:
            line = stream.readline().decode('ascii').strip()
            if line.startswith('format '):
                fmt = line.split()[1]
            elif line.startswith('element '):
                _, element, number = line.split()
                vertex = element == 'vertex'
                if vertex:
                    count = int(number)
            elif line.startswith('property ') and vertex:
                parts = line.split()
                if parts[1] == 'list':
                    raise ValueError(f'List vertex properties are unsupported: {path}')
                properties.append((parts[2], parts[1]))
            elif line == 'end_header':
                break
        if count is None or fmt not in ('ascii', 'binary_little_endian'):
            raise ValueError(f'Unsupported PLY format: {path}')
        names = [name for name, _ in properties]
        for required in ('x', 'y', 'z'):
            if required not in names:
                raise ValueError(f'Missing {required} in {path}')
        if fmt == 'ascii':
            values = np.loadtxt(stream, dtype=np.float64, max_rows=count, ndmin=2)
            columns = {name: values[:, i] for i, name in enumerate(names)}
        else:
            dtype = np.dtype([(name, '<' + PLY_TYPES[kind]) for name, kind in properties])
            values = np.fromfile(stream, dtype=dtype, count=count)
            columns = {name: values[name] for name in names}
    intensity_name = next((n for n in ('intensity', 'reflectance', 'i') if n in columns), None)
    intensity = columns[intensity_name] if intensity_name else np.ones(count)
    return np.column_stack((columns['x'], columns['y'], columns['z'], intensity)).astype(np.float32)


def matrix(pose):
    if len(pose) != 6:
        raise ValueError(f'Expected six-element pose, got {pose}')
    x, y, z, roll, yaw, pitch = [float(v) for v in pose]
    roll, yaw, pitch = map(math.radians, (roll, yaw, pitch))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    value = np.eye(4)
    value[:3, :3] = [
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
        [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
        [sp, -cp * sr, cp * cr]
    ]
    value[:3, 3] = [x, y, z]
    return value


def transform_points(points, transform):
    xyz = points[:, :3] @ transform[:3, :3].T + transform[:3, 3]
    return np.column_stack((xyz, points[:, 3])).astype(np.float32)


def frame_yaml(agent, frame):
    path = agent / f'{frame}.yaml'
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open() as stream:
        return yaml.safe_load(stream)


def find_scenario(root, name):
    candidates = [p for p in root.rglob(name) if p.is_dir() and (p / 'data_protocol.yaml').is_file()]
    if len(candidates) != 1:
        raise ValueError(f'Expected one extracted {name} directory, found {len(candidates)}')
    return candidates[0]


def agent_key(path):
    try:
        value = int(path.name)
    except ValueError:
        return (2, path.name)
    return (0 if value >= 0 else 1, value)


def normalize_boxes(raw, world_to_ego, ego_yaw, aliases, classes, bounds):
    boxes = []
    groups, rejected = [], []
    for group_name in ("vehicles", "walkers"):
        group = raw.get(group_name) or {}
        if not isinstance(group, Mapping):
            rejected.append(f"{group_name}: expected mapping")
            continue
        groups.extend(group.values())
    for index, obj in enumerate(groups):
        try:
            if not isinstance(obj, Mapping):
                raise TypeError(f"expected mapping, got {type(obj).__name__}")
            label = aliases.get(
                str(obj.get("class", "")).lower(), str(obj.get("class", "")).lower()
            )
            if label not in classes:
                continue
            location = np.asarray(obj["location"], dtype=float)
            angle = np.asarray(obj.get("angle", [0, 0, 0]), dtype=float)
            center = np.asarray(obj.get("center", [0, 0, 0]), dtype=float)
            extent = np.asarray(obj["extent"], dtype=float)
            if (
                location.shape != (3,)
                or angle.shape != (3,)
                or center.shape != (3,)
                or extent.shape != (3,)
                or not np.isfinite(
                    np.concatenate((location, angle, center, extent))
                ).all()
                or (extent <= 0).any()
            ):
                raise ValueError("invalid vector shape, non-finite value, or extent")
            yaw = float(angle[1])
            radians = math.radians(yaw)
            rotation = np.array(
                [
                    [math.cos(radians), -math.sin(radians), 0],
                    [math.sin(radians), math.cos(radians), 0],
                    [0, 0, 1],
                ]
            )
            location = location + rotation @ center
            local = world_to_ego[:3, :3] @ location + world_to_ego[:3, 3]
        except (
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
        ) as error:
            rejected.append(f"#{index}: {error}")
            continue
        if ((local >= bounds[:3]) & (local < bounds[3:])).all():
            boxes.append({'class': label, 'position': local.tolist(),
                          'dimensions': (extent * 2).tolist(),
                          'heading': math.radians(yaw - ego_yaw)})
    if rejected:
        warnings.warn(
            f"Skipped {len(rejected)} invalid Adver-City source annotations "
            f'({"; ".join(rejected[:3])})',
            RuntimeWarning,
        )
    return {'objects': boxes}


def convert_scenario(extracted, destination, name, cfg):
    source = find_scenario(extracted, name)
    agents = sorted([p for p in source.iterdir() if p.is_dir() and list(p.glob('*_lidar.ply'))], key=agent_key)
    if not agents:
        raise ValueError(f'No LiDAR agents in {name}')
    frame_sets = [{p.stem.split('_', 1)[0] for p in agent.glob('*_lidar.ply')} for agent in agents]
    frames = sorted(set.intersection(*frame_sets))[::cfg['dataset']['acquisition']['retain_frame_stride']]
    if not frames:
        raise ValueError(f'No common LiDAR frames in {name}')
    first_frame = frames[0]
    vehicles = []
    for agent in agents:
        raw = frame_yaml(agent, first_frame)
        if 'RSU' in raw:
            is_rsu = bool(raw['RSU'])
        else:
            try:
                is_rsu = int(agent.name) < 0
            except ValueError:
                is_rsu = False
        if not is_rsu:
            vehicles.append(agent)
    if not vehicles:
        raise ValueError(f'No vehicle agents in {name}')
    ego = vehicles[0]
    stage = destination.parent / f'.{name}.converting'
    if stage.exists():
        shutil.rmtree(stage)
    (stage / 'frames').mkdir(parents=True)
    bounds = np.asarray(cfg['dataset']['bounds'], dtype=float)
    aliases = cfg['dataset']['class_aliases']
    classes = cfg['dataset']['classes']
    node_types = {}
    point_total = 0
    for frame in frames:
        ego_raw = frame_yaml(ego, frame)
        ego_pose = ego_raw['lidar_pose']
        ego_matrix = matrix(ego_pose)
        world_to_ego = np.linalg.inv(ego_matrix)
        target = stage / 'frames' / frame
        (target / 'nodes').mkdir(parents=True)
        valid_nodes = 0
        for agent in agents:
            raw = frame_yaml(agent, frame)
            node_types[agent.name] = 'rsu' if raw.get('RSU') else 'vehicle'
            points = read_ply(agent / f'{frame}_lidar.ply')
            relative = world_to_ego @ matrix(raw['lidar_pose'])
            points = transform_points(points, relative)
            keep = ((points[:, :3] >= bounds[:3]) & (points[:, :3] < bounds[3:])).all(axis=1)
            points = points[keep]
            if len(points):
                np.save(target / 'nodes' / f'{agent.name}.npy', points, allow_pickle=False)
                point_total += len(points)
                valid_nodes += 1
        if not valid_nodes:
            raise ValueError(f'All nodes empty after crop in {name}/{frame}')
        labels = normalize_boxes(ego_raw, world_to_ego, float(ego_pose[4]), aliases, classes, bounds)
        (target / 'gt.yaml').write_text(yaml.safe_dump(labels, sort_keys=False))
    metadata = {
        'scenario': name, 'frames': len(frames), 'nodes': node_types,
        'points': point_total, 'source_archive': f'{name}.7z',
        'format': 1, 'coordinate_frame': 'ego_lidar'
    }
    (stage / 'scenario.yaml').write_text(yaml.safe_dump(metadata, sort_keys=False))
    probe = sorted((stage / 'frames').glob('*/nodes/*.npy'))
    if not probe or np.load(probe[0], allow_pickle=False).shape[1] != 4:
        raise ValueError(f'Validation failed for {name}')
    if destination.exists():
        shutil.rmtree(destination)
    stage.replace(destination)
    return metadata


def selected_scenarios(cfg):
    values = []
    for split in ('train', 'val', 'test'):
        values.extend(cfg['dataset']['splits'][split])
    if len(values) != len(set(values)):
        raise ValueError('Adver-City scenario splits overlap')
    return values


def prepare(cfg, preview=False):
    d = cfg['dataset']
    acquisition = d['acquisition']
    root = Path(d['root'])
    work = Path(acquisition['work_dir'])
    archives = work / 'archives'
    extraction = work / 'extracting'
    root.mkdir(parents=True, exist_ok=True)
    archives.mkdir(parents=True, exist_ok=True)
    extraction.mkdir(parents=True, exist_ok=True)
    files = remote_files()
    checksums = remote_checksums()
    scenarios = selected_scenarios(cfg)
    unknown = sorted(set(scenarios) - set(files))
    if unknown:
        raise ValueError(f'Scenarios not in official release: {unknown}')
    total = sum(files[name]['size'] for name in scenarios)
    print(f'Selected {len(scenarios)} scenarios, {total/1e9:.2f} GB compressed in total')
    for name in scenarios:
        item = files[name]
        print(f'{name}: {item["size"]/1e9:.2f} GB')
    if preview:
        return
    reserve = acquisition['minimum_free_gb'] * 1_000_000_000
    for name in scenarios:
        destination = root / name
        if (destination / 'scenario.yaml').is_file():
            print(f'Skipping prepared scenario {name}')
            continue
        item = files[name]
        archive = archives / item['name']
        ensure_space(work, item['size'], reserve)
        expected = checksums.get(item['name'])
        if archive.exists() and (archive.stat().st_size != item['size'] or sha256(archive) != expected):
            archive.unlink()
        if not archive.exists():
            url = f'{FILE_URL}/{urllib.parse.quote(item["name"])}'
            download(url, archive, item['size'])
        if not expected or sha256(archive) != expected:
            raise ValueError(f'Publisher checksum mismatch: {archive}')
        scenario_extract = extraction / name
        if scenario_extract.exists():
            shutil.rmtree(scenario_extract)
        scenario_extract.mkdir()
        extract_archive(archive, scenario_extract)
        lidar_bytes = sum(path.stat().st_size for path in scenario_extract.rglob('*.ply'))
        ensure_space(root, int(lidar_bytes * 1.25), reserve)
        metadata = convert_scenario(scenario_extract, destination, name, cfg)
        print(json.dumps(metadata, indent=2))
        if acquisition['cleanup_archives']:
            archive.unlink()
        shutil.rmtree(scenario_extract)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/advercity.yaml')
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    from src.config import load_config
    cfg = load_config(args.config)
    if cfg['dataset'].get('type') != 'advercity':
        raise ValueError('prepare_advercity.py requires dataset.type=advercity')
    prepare(cfg, args.preview)


if __name__ == '__main__':
    main()
