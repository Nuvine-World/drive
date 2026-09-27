import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import yaml
from prepare_advercity import (
    convert_scenario,
    download,
    extract_archive,
    matrix,
    normalize_boxes,
    read_ply,
)
from src.config import PROJECT, load_config
from src.data.advercity import annotations, discover, load_cloud, scenario_weather


def write_ply(path, points):
    header = ('ply\nformat ascii 1.0\nelement vertex %d\n'
              'property float x\nproperty float y\nproperty float z\n'
              'property float intensity\nend_header\n') % len(points)
    path.write_text(header + '\n'.join(' '.join(map(str, row)) for row in points) + '\n')


def make_raw_scenario(root, name='ui_cd_d'):
    scenario = root / 'dump' / name
    scenario.mkdir(parents=True)
    (scenario / 'data_protocol.yaml').write_text('{}\n')
    for agent, pose, rsu in [('10', [0, 0, 0, 0, 0, 0], False),
                             ('11', [10, 0, 0, 0, 0, 0], False),
                             ('-1', [0, 10, 0, 0, -90, 0], True)]:
        folder = scenario / agent
        folder.mkdir()
        raw = {'RSU': rsu, 'lidar_pose': pose, 'vehicles': {}, 'walkers': {}}
        if agent == '10':
            raw['vehicles'] = {1: {'class': 'car', 'location': [5, 0, 0],
                'center': [0, 0, 0], 'extent': [2, 1, 1], 'angle': [0, 0, 0]}}
        (folder / '000060.yaml').write_text(yaml.safe_dump(raw))
        write_ply(folder / '000060_lidar.ply', [[1, 0, 0, .5], [200, 0, 0, .2]])
    return scenario


def test_convert_and_discover_advercity(tmp_path):
    raw = tmp_path / 'raw'
    make_raw_scenario(raw)
    cfg = load_config(PROJECT / 'configs/advercity.yaml')
    cfg['dataset'].update(root=str(tmp_path / 'prepared'), bounds=[-20, -20, -5, 20, 20, 5],
                          frame_stride=1, splits={'train': ['ui_cd_d'],
                          'val': ['ri_fd_d'], 'test': ['rcnj_fhrd_d']})
    cfg['dataset']['acquisition']['retain_frame_stride'] = 1
    destination = Path(cfg['dataset']['root']) / 'ui_cd_d'
    destination.parent.mkdir()
    result = convert_scenario(raw, destination, 'ui_cd_d', cfg)
    assert result['frames'] == 1
    cloud = np.load(destination / 'frames/000060/nodes/11.npy')
    np.testing.assert_allclose(cloud[0, :3], [11, 0, 0])
    assert annotations(destination / 'frames/000060/gt.yaml', cfg['dataset'])[0].length == 4
    for name, weather in [('ri_fd_d', 'fog'), ('rcnj_fhrd_d', 'fog_heavy_rain')]:
        target = Path(cfg['dataset']['root']) / name
        target.mkdir()
        (target / 'scenario.yaml').write_text('format: 1\n')
        frame = target / 'frames/000060'
        (frame / 'nodes').mkdir(parents=True)
        np.save(frame / 'nodes/10.npy', np.array([[1, 0, 0, 1]], np.float32))
        (frame / 'gt.yaml').write_text('objects: []\n')
        assert scenario_weather(name) == weather
    samples = discover(cfg)
    assert len(samples) == 3
    prepared = samples[0]['nodes']['10']
    assert load_cloud(prepared, cfg['dataset']['bounds']).shape == (1, 4)


def test_advercity_weather_filter_is_applied(tmp_path):
    root = tmp_path/'prepared'
    for name in ('ui_cd_d','ri_fd_d'):
        frame = root/name/'frames/000060'
        (frame/'nodes').mkdir(parents=True)
        (root/name/'scenario.yaml').write_text('format: 1\n')
        np.save(frame/'nodes/10.npy',np.array([[1,0,0,1]],np.float32))
        (frame/'gt.yaml').write_text('objects: []\n')
    cfg = load_config(PROJECT/'configs/advercity.yaml')
    cfg['dataset'].update(root=str(root),weather=['fog'],frame_stride=1,
                          splits={'train':['ui_cd_d','ri_fd_d'],'val':[],'test':[]})
    samples = discover(cfg)
    assert [sample['session'] for sample in samples]==['ri_fd_d']


def test_advercity_skips_only_invalid_annotations(tmp_path):
    cfg = load_config(PROJECT / "configs/advercity.yaml")["dataset"]
    path = tmp_path / "gt.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "objects": [
                    {
                        "class": "car",
                        "position": [1, 2, 0],
                        "dimensions": [4, 2, 2],
                        "heading": 0,
                    },
                    {
                        "class": "car",
                        "position": [float("nan"), 2, 0],
                        "dimensions": [4, 2, 2],
                        "heading": 0,
                    },
                    {
                        "class": "car",
                        "position": [1, 2, 0],
                        "dimensions": [0, 2, 2],
                        "heading": 0,
                    },
                ]
            }
        )
    )
    with pytest.warns(RuntimeWarning, match="Skipped 2 invalid"):
        boxes = annotations(path, cfg)
    assert len(boxes) == 1
    assert boxes[0].label == "car"


def test_normalize_boxes_skips_and_reports_malformed_source_objects():
    raw = {
        "vehicles": {
            1: {
                "class": "car",
                "location": [5, 0, 0],
                "center": [0, 0, 0],
                "extent": [2, 1, 1],
                "angle": [0, 0, 0],
            },
            2: None,
            3: {"class": "car", "location": [float("nan"), 0, 0], "extent": [2, 1, 1]},
            4: {"class": "car", "location": [1, 0, 0], "extent": [0, 1, 1]},
        },
        "walkers": ["not", "a", "mapping"],
    }
    with pytest.warns(RuntimeWarning, match="Skipped 4 invalid"):
        result = normalize_boxes(
            raw,
            np.eye(4),
            0,
            {"vehicle": "car"},
            ["car"],
            np.array([-20, -20, -5, 20, 20, 5]),
        )
    assert len(result["objects"]) == 1
    assert result["objects"][0]["dimensions"] == [4.0, 2.0, 2.0]


def test_convert_accepts_nonnumeric_vehicle_agent(tmp_path):
    raw = tmp_path/'raw'
    scenario = make_raw_scenario(raw)
    (scenario/'10').rename(scenario/'ego')
    cfg = load_config(PROJECT/'configs/advercity.yaml')
    cfg['dataset'].update(root=str(tmp_path/'prepared'),bounds=[-20,-20,-5,20,20,5])
    destination = Path(cfg['dataset']['root'])/'ui_cd_d'
    destination.parent.mkdir()
    result = convert_scenario(raw,destination,'ui_cd_d',cfg)
    assert result['frames']==1
    assert (destination/'frames/000060/nodes/ego.npy').is_file()


def test_binary_ply_and_pose(tmp_path):
    path = tmp_path / 'cloud.ply'
    header = ('ply\nformat binary_little_endian 1.0\nelement vertex 1\n'
              'property float x\nproperty float y\nproperty float z\n'
              'property float intensity\nend_header\n').encode()
    path.write_bytes(header + np.array([(1, 2, 3, .4)], dtype=[
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4')]).tobytes())
    np.testing.assert_allclose(read_ply(path), [[1, 2, 3, .4]])
    np.testing.assert_allclose(matrix([1, 2, 3, 0, 0, 0])[:3, 3], [1, 2, 3])


def test_resumable_download(tmp_path, monkeypatch):
    content = b'abcdefghij'
    destination = tmp_path / 'x.7z'
    part = destination.with_suffix('.7z.part')
    part.write_bytes(content[:4])

    class Response:
        status = 206
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def read(self, size=-1):
            value, self.data = self.data, b''
            return value
        data = content[4:]

    monkeypatch.setattr('prepare_advercity.request', lambda url, headers=None: Response())
    download('https://example.invalid/x', destination, len(content))
    assert destination.read_bytes() == content


def test_selective_py7zr_extraction(tmp_path, monkeypatch):
    import py7zr
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'frame.yaml').write_text('{}\n')
    (source / 'frame_lidar.ply').write_text('ply\n')
    (source / 'camera.png').write_bytes(b'image')
    archive = tmp_path / 'scenario.7z'
    with py7zr.SevenZipFile(archive, 'w') as bundle:
        bundle.writeall(source, arcname='scenario')
    monkeypatch.setattr('prepare_advercity.shutil.which', lambda name: None)
    output = tmp_path / 'output'
    extract_archive(archive, output)
    assert (output / 'scenario/frame.yaml').is_file()
    assert (output / 'scenario/frame_lidar.ply').is_file()
    assert not (output / 'scenario/camera.png').exists()
