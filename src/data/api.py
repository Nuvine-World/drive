from src.data.coinfra import bev_channels, rasterize


def backend(cfg):
    name = cfg['dataset'].get('type', 'coinfra')
    if name == 'coinfra':
        from src.data import coinfra
        return coinfra
    if name == 'advercity':
        from src.data import advercity
        return advercity
    raise ValueError(f'Unsupported dataset type: {name}')


def discover(cfg):
    return backend(cfg).discover(cfg)


def annotations(path, cfg):
    return backend({'dataset': cfg}).annotations(path, cfg)


def load_cloud(node, bounds):
    name = node.get('dataset_type', 'coinfra')
    return backend({'dataset': {'type': name}}).load_cloud(node, bounds)
