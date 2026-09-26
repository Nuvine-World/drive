"""Reproducible CoInfra preprocessing, training, inference and network sweeps."""
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import time
import zlib
import platform
import shutil
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np
import torch
from src.data.api import annotations, discover, load_cloud, rasterize
from src.detection.bev import BEVDetector, targets, loss, decode
from src.metrics.evaluation import evaluate, nms


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def plot_training(path, rows):
    if not rows:
        return
    try:
        import matplotlib
    except ImportError:
        print('Plot skipped: install matplotlib from requirements.txt', flush=True)
        return
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    epochs = [r['epoch'] for r in rows]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(epochs, [r['train_loss'] for r in rows], label='train loss')
    axes[0].plot(epochs, [r['val_loss'] for r in rows], label='validation loss')
    axes[0].set(xlabel='epoch', ylabel='loss', title='Loss')
    axes[0].legend()
    axes[1].plot(epochs, [r['val_map'] for r in rows], label='validation mAP')
    axes[1].plot(epochs, [r['val_precision'] for r in rows], label='validation precision')
    axes[1].plot(epochs, [r['val_recall'] for r in rows], label='validation recall')
    axes[1].plot(epochs, [r['val_f1'] for r in rows], label='validation F1')
    axes[1].plot(epochs, [r['selection_score'] for r in rows], label='selection score')
    axes[1].set(xlabel='epoch', ylabel='score', ylim=(0, 1), title='Validation metrics')
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def validation_metrics(model, samples, cfg, device):
    losses = []
    frames = {strategy: [] for strategy in cfg['evaluation']['strategies']}
    with torch.no_grad():
        for sample in samples:
            clouds = inputs(sample, sorted(sample['nodes']), cfg)
            bev = tensor_inputs(clouds, cfg, device, 'early')
            gt = annotations(sample['gt'], cfg['dataset'])
            target = targets(gt, cfg['dataset'], bev.shape[-2:])
            outputs = model(bev)
            losses.append(float(loss(outputs, target)))
            for strategy in frames:
                predictions, _, _ = infer(model, clouds, cfg, device, strategy)
                frames[strategy].append((predictions, gt))
    details, all_aps, all_precisions, all_recalls, all_f1s = {}, [], [], [], []
    for strategy, strategy_frames in frames.items():
        rows = evaluate(strategy_frames, cfg['dataset']['classes'], cfg['evaluation']['iou_thresholds'])
        aps = [r['ap_3d'] for r in rows if r['ap_3d'] is not None]
        precisions = [r['precision'] for r in rows if r['precision'] is not None]
        recalls = [r['recall'] for r in rows if r['recall'] is not None]
        f1s = [r['f1'] for r in rows if r['f1'] is not None]
        details[f'val_map_{strategy}'] = float(np.mean(aps)) if aps else 0.0
        details[f'val_precision_{strategy}'] = float(np.mean(precisions)) if precisions else 0.0
        details[f'val_recall_{strategy}'] = float(np.mean(recalls)) if recalls else 0.0
        details[f'val_f1_{strategy}'] = float(np.mean(f1s)) if f1s else 0.0
        all_aps.extend(aps)
        all_precisions.extend(precisions)
        all_recalls.extend(recalls)
        all_f1s.extend(f1s)
    val_loss = float(np.mean(losses))
    details['val_precision'] = float(np.mean(all_precisions)) if all_precisions else 0.0
    details['val_f1'] = float(np.mean(all_f1s)) if all_f1s else 0.0
    return (val_loss, float(np.mean(all_aps)) if all_aps else 0.0,
            float(np.mean(all_recalls)) if all_recalls else 0.0, details)


def selection_score(cfg, val_loss, val_map, val_recall):
    weights = cfg['training']['checkpointing']['weights']
    values = {'val_loss_quality': 1.0 / (1.0 + val_loss),
              'val_map': val_map, 'val_recall': val_recall}
    total = sum(weights.values())
    return sum(weights[key] * values[key] for key in values) / total


def prune_checkpoints(directory, keep):
    candidates = []
    for metadata in directory.glob('epoch_*.json'):
        value = json.loads(metadata.read_text())
        checkpoint = metadata.with_suffix('.pt')
        if checkpoint.exists():
            candidates.append((float(value['selection_score']), int(value['epoch']), checkpoint, metadata))
    candidates.sort(reverse=True)
    for _, _, checkpoint, metadata in candidates[keep:]:
        checkpoint.unlink()
        metadata.unlink()
    return candidates[:keep]


def wait_for_resources(cfg, device, reason='resource check'):
    settings = cfg['training']['recovery']
    if not settings['wait_for_resources']:
        return
    started = time.monotonic()
    disk_required = settings['minimum_free_disk_gb'] * 1_000_000_000
    gpu_required = settings['minimum_free_gpu_gb'] * 1_000_000_000
    output = Path(cfg['output']['root'])
    output.mkdir(parents=True, exist_ok=True)
    while True:
        disk_free = shutil.disk_usage(output).free
        gpu_free = None
        if device.type == 'cuda':
            gpu_free, _ = torch.cuda.mem_get_info(device)
        disk_ok = disk_free >= disk_required
        gpu_ok = gpu_free is None or gpu_free >= gpu_required
        if disk_ok and gpu_ok:
            return
        elapsed = time.monotonic() - started
        timeout = settings['timeout_minutes'] * 60
        details = f'disk_free={disk_free/1e9:.2f}GB'
        if gpu_free is not None:
            details += f' gpu_free={gpu_free/1e9:.2f}GB'
        if timeout and elapsed >= timeout:
            raise TimeoutError(f'Resource wait timed out ({reason}): {details}')
        print(f'[{utc_now()}] WAITING ({reason}): {details}; retrying in '
              f'{settings["poll_seconds"]}s', flush=True)
        time.sleep(settings['poll_seconds'])


def atomic_torch_save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def fingerprint(cfg):
    return hashlib.sha256(json.dumps(cfg['dataset'],sort_keys=True).encode()).hexdigest()


def signature(cfg):
    d = cfg['dataset']
    return {k:d[k] for k in ('classes','bounds','resolution')} | {'hidden':cfg['training']['hidden_channels']}


def device_for(cfg):
    name = cfg['training']['device']
    return torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if name == 'auto' else name)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


def preprocess(cfg):
    samples = discover(cfg)
    validate_node_capacity(samples, cfg)
    counts = defaultdict(int)
    for sample in samples:
        counts[f'{sample["split"]}/{sample["weather"]}'] += 1
    for split in ('train','val','test'):
        if not any(s['split']==split for s in samples):
            raise ValueError(f'No {split} samples after weather filters')
    output = Path(cfg['output']['root'])
    write_json(output/'index.json', {'dataset_fingerprint':fingerprint(cfg),'samples':samples})
    write_json(output/'resolved_config.json',cfg)
    write_json(output/'environment.json',{'python':platform.python_version(),
               'platform':platform.platform(),'torch':torch.__version__,'numpy':np.__version__})
    write_json(output/'inventory.json',dict(counts))
    print(json.dumps(dict(counts),indent=2))
    return samples


def indexed(cfg):
    path = Path(cfg['output']['root'])/'index.json'
    if not path.exists():
        return preprocess(cfg)
    value = json.loads(path.read_text())
    if value['dataset_fingerprint'] != fingerprint(cfg):
        raise ValueError('Dataset config changed. Rerun --stage preprocess to rebuild the index.')
    validate_node_capacity(value['samples'], cfg)
    return value['samples']


def validate_node_capacity(samples, cfg):
    """Ensure every test session can support requested node counts before training."""
    sessions = defaultdict(list)
    for sample in samples:
        if sample['split'] == 'test':
            sessions[sample['session']].append(sample)
    requested = max(cfg['evaluation']['node_counts'])
    for session, frames in sessions.items():
        common = set.intersection(*(set(frame['nodes']) for frame in frames))
        if len(common) < requested:
            raise ValueError(f'Test session {session} has {len(common)} common LiDAR nodes, but '
                             f'evaluation.node_counts requests {requested}; adjust the configuration')


def inspect(cfg):
    samples = preprocess(cfg)
    annotation_counts = {sample['id']:len(annotations(sample['gt'],cfg['dataset']))
                         for sample in samples}
    report = []
    seen = set()
    for sample in samples:
        key = (sample['session'],sample['slice'])
        if key in seen:
            continue
        seen.add(key)
        clouds = {node:load_cloud(value,cfg['dataset']['bounds']) for node,value in sample['nodes'].items()}
        report.append({'sample':sample['id'],'points_in_bounds':{n:len(p) for n,p in clouds.items()},
                       'boxes_in_bounds':annotation_counts[sample['id']]})
        if not any(len(p) for p in clouds.values()):
            raise ValueError(f'All transformed points outside configured bounds at {sample["id"]}')
    write_json(Path(cfg['output']['root'])/'inspection.json',report)
    print(f'Validated annotations in {len(samples)} frames and point transforms in '
          f'{len(report)} representative slices; see inspection.json')


def inputs(sample, nodes, cfg):
    return [load_cloud(sample['nodes'][n],cfg['dataset']['bounds']) for n in nodes]


def tensor_inputs(clouds, cfg, device, mode):
    if mode == 'early':
        bevs = rasterize(np.concatenate(clouds),cfg['dataset'])[None]
    else:
        bevs = np.stack([rasterize(p,cfg['dataset']) for p in clouds])
    return torch.from_numpy(bevs).to(device)


def make_model(cfg,device):
    return BEVDetector(cfg['dataset']['classes'],cfg['training']['hidden_channels']).to(device)


def train(cfg, overwrite=False):
    samples = indexed(cfg)
    training = [s for s in samples if s['split']=='train']
    validation = [s for s in samples if s['split']=='val']
    output, device = Path(cfg['output']['root']), device_for(cfg)
    wait_for_resources(cfg, device, 'before training')
    for seed in cfg['training']['seeds']:
        path = output/f'seed_{seed}'/'best.pt'
        latest = path.parent/'latest.pt'
        completion = path.parent/'complete.json'
        if path.exists() and completion.exists() and not overwrite:
            state = torch.load(path,map_location='cpu',weights_only=True)
            if state['signature'] != signature(cfg) or state['config'] != cfg:
                raise ValueError(f'{path} belongs to a different config; use another output directory')
            if json.loads(completion.read_text()).get('epochs') != cfg['training']['epochs']:
                raise ValueError(f'Incomplete training for seed {seed}; use --overwrite to restart that run')
            print(f'Skipping existing checkpoint {path}; --overwrite retrains it')
            continue
        if overwrite:
            for stale in list(path.parent.glob('epoch_*.pt')) + list(path.parent.glob('epoch_*.json')):
                stale.unlink()
            for stale in (completion, path, latest):
                if stale.exists():
                    stale.unlink()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        rng = random.Random(seed)
        model = make_model(cfg,device)
        optimizer = torch.optim.AdamW(model.parameters(),lr=cfg['training']['learning_rate'],
                                      weight_decay=cfg['training']['weight_decay'])
        best_score, history, start_epoch = -float('inf'), [], 0
        if not overwrite and not completion.exists() and latest.exists():
            if not cfg['training']['recovery']['resume']:
                raise ValueError(f'Incomplete training for seed {seed}; enable training.recovery.resume or use --overwrite')
            state = torch.load(latest, map_location=device, weights_only=True)
            if state['signature'] != signature(cfg) or state['config'] != cfg:
                raise ValueError(f'{latest} belongs to a different configuration')
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            start_epoch = int(state['epoch'])
            best_score = float(state['best_score'])
            history = state['history']
            rng.setstate(state['python_rng_state'])
            torch.set_rng_state(state['torch_rng_state'].cpu())
            if device.type == 'cuda' and state.get('cuda_rng_state') is not None:
                torch.cuda.set_rng_state(state['cuda_rng_state'].cpu(), device)
            print(f'[{utc_now()}] RESUMING seed={seed} after epoch={start_epoch} from {latest}', flush=True)
        elif not overwrite and path.exists() and not completion.exists():
            raise ValueError(f'Incomplete training for seed {seed} has no recovery checkpoint; use --overwrite')
        started = time.perf_counter()
        for epoch in range(start_epoch, cfg['training']['epochs']):
            wait_for_resources(cfg, device, f'before seed={seed} epoch={epoch+1}')
            model.train()
            order = training.copy()
            rng.shuffle(order)
            train_losses = []
            for sample in order:
                available = sorted(sample['nodes'])
                counts = [n for n in cfg['training']['node_counts'] if n<=len(available)]
                if not counts:
                    raise ValueError('No training node count fits available data')
                mode = rng.choice(cfg['training']['modes'])
                nodes = rng.sample(available,1 if mode=='single' else rng.choice(counts))
                cloud = inputs(sample,nodes,cfg)
                bev = tensor_inputs(cloud,cfg,device,'early' if mode=='single' else mode)
                attempts = 0
                while True:
                    try:
                        outputs = model(bev,'early' if mode=='single' else mode)
                        target = targets(annotations(sample['gt'],cfg['dataset']),cfg['dataset'],bev.shape[-2:])
                        value = loss(outputs,target)
                        if not torch.isfinite(value):
                            raise ValueError('Nonfinite training loss')
                        optimizer.zero_grad(set_to_none=True)
                        value.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),10)
                        optimizer.step()
                        break
                    except torch.OutOfMemoryError:
                        optimizer.zero_grad(set_to_none=True)
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        attempts += 1
                        if attempts > cfg['training']['recovery']['cuda_oom_retries']:
                            raise
                        print(f'[{utc_now()}] CUDA OOM seed={seed} epoch={epoch+1}; retry '
                              f'{attempts}/{cfg["training"]["recovery"]["cuda_oom_retries"]}', flush=True)
                        wait_for_resources(cfg, device, 'CUDA out of memory')
                train_losses.append(float(value.detach()))
            model.eval()
            val, val_map, val_recall, validation_details = validation_metrics(
                model, validation, cfg, device)
            if not all(np.isfinite(v) for v in (val, val_map, val_recall)):
                raise ValueError('Nonfinite validation metric')
            score = selection_score(cfg, val, val_map, val_recall)
            row = {'timestamp_utc':utc_now(), 'seed':seed, 'epoch':epoch+1,
                   'train_loss':float(np.mean(train_losses)), 'val_loss':val,
                   'val_map':val_map, 'val_recall':val_recall,
                   'val_loss_quality':1.0/(1.0+val), 'selection_score':score,
                   'learning_rate':optimizer.param_groups[0]['lr'],
                   'elapsed_seconds':time.perf_counter()-started} | validation_details
            history.append(row)
            path.parent.mkdir(parents=True, exist_ok=True)
            candidate = path.parent/f'epoch_{epoch+1:04d}.pt'
            temporary = candidate.with_suffix('.tmp')
            state = {'model':model.state_dict(),'signature':signature(cfg),'config':cfg,
                     'epoch':epoch+1,'seed':seed,'validation_loss':val,
                     'validation_map':val_map,'validation_recall':val_recall,
                     'selection_score':score}
            torch.save(state, temporary)
            temporary.replace(candidate)
            write_json(candidate.with_suffix('.json'), {k:state[k] for k in
                       ('epoch','seed','validation_loss','validation_map','validation_recall','selection_score')})
            retained = prune_checkpoints(path.parent, cfg['training']['checkpointing']['keep_best'])
            if score > best_score:
                best_score = score
                best_tmp = path.with_suffix('.tmp')
                shutil.copyfile(candidate, best_tmp)
                best_tmp.replace(path)
            metrics_path = path.parent/f'training_metrics_seed_{seed}.csv'
            write_csv(metrics_path, history)
            if cfg['output']['plots']:
                plot_training(path.parent/f'training_curves_seed_{seed}.png', history)
            kept_epochs = ','.join(str(item[1]) for item in retained)
            print(f'seed={seed} epoch={epoch+1} train={row["train_loss"]:.4f} '
                  f'val={val:.4f} mAP={val_map:.4f} precision={row["val_precision"]:.4f} '
                  f'recall={val_recall:.4f} F1={row["val_f1"]:.4f} '
                  f'score={score:.4f} kept=[{kept_epochs}]', flush=True)
            if (epoch+1) % cfg['training']['recovery']['save_every_epochs'] == 0:
                wait_for_resources(cfg, device, 'before recovery checkpoint')
                recovery_state = state | {'optimizer':optimizer.state_dict(),
                    'best_score':best_score,'history':history,
                    'python_rng_state':rng.getstate(),'torch_rng_state':torch.get_rng_state(),
                    'cuda_rng_state':torch.cuda.get_rng_state(device) if device.type=='cuda' else None}
                atomic_torch_save(recovery_state, latest)
        best_state = torch.load(path, map_location='cpu', weights_only=True)
        write_json(completion,{'timestamp_utc':utc_now(),'epochs':cfg['training']['epochs'],
                   'best_epoch':best_state['epoch'],'best_validation_loss':best_state['validation_loss'],
                   'best_validation_map':best_state['validation_map'],
                   'best_validation_recall':best_state['validation_recall'],
                   'best_selection_score':best_state['selection_score']})


def choose_groups(nodes, settings):
    nodes = sorted(nodes)
    rng = random.Random(settings['subset_seed'])
    groups = []
    for count in settings['node_counts']:
        if count > len(nodes):
            raise ValueError(f'Requested {count} nodes but only {len(nodes)} common nodes. Adjust evaluation.node_counts.')
        choices = list(itertools.combinations(nodes,count))
        if count != 1 and len(choices)>settings['subsets_per_count']:
            choices = sorted(rng.sample(choices,settings['subsets_per_count']))
        groups.extend(choices)
    return groups


def payload(array):
    return len(zlib.compress(np.asarray(array,dtype='<f4').tobytes(),level=1))


def infer(model, clouds, cfg, device, strategy):
    bev = tensor_inputs(clouds,cfg,device,strategy)
    sync(device)
    start = time.perf_counter()
    with torch.no_grad():
        features = model.encoder(bev)
        if strategy=='intermediate':
            output = model.heads(features.amax(dim=0,keepdim=True))
            boxes = decode(output,cfg)
        elif strategy=='late':
            local = [decode(model.heads(f[None]),cfg) for f in features]
            boxes = nms([b for group in local for b in group],cfg['evaluation']['nms_iou'],cfg['evaluation']['max_detections'])
        else:
            boxes = decode(model.heads(features),cfg)
    sync(device)
    elapsed_ms = (time.perf_counter()-start)*1000
    if strategy=='intermediate':
        representation = sum(payload(f) for f in features.detach().cpu().numpy())
    elif strategy=='late':
        representation = sum(payload(np.array([list(b.to_array())+[cfg['dataset']['classes'].index(b.label),b.score]
                                             for b in group],dtype=np.float32).reshape(-1,9)) for group in local)
    else:
        representation = sum(payload(p) for p in clouds)
    return boxes,elapsed_ms,representation


def network_rows(cfg,base,raw_bytes,representation_bytes):
    net = cfg['deployment_assumptions']
    rows = []
    for placement in ('edge','cloud'):
        transmitted = representation_bytes if placement=='edge' else raw_bytes
        for bandwidth in net['bandwidths_mbps']:
            transmit_ms = transmitted*8/(bandwidth*1e6)*1000
            total = net['sensing_ms']+net[placement+'_rtt_ms']+net['processing_ms'][placement][base['strategy']]+transmit_ms
            rows.append(base | {'placement':placement,'bandwidth_mbps':bandwidth,
                                'assumption_source':net['source'],
                                'payload_bytes':transmitted,'transmission_ms_model':transmit_ms,
                                'end_to_end_ms_model':total})
    return rows


def run_evaluation(cfg):
    samples = [s for s in indexed(cfg) if s['split']=='test']
    sessions = defaultdict(list)
    for sample in samples:
        sessions[sample['session']].append(sample)
    output, device = Path(cfg['output']['root']),device_for(cfg)
    all_metrics, all_costs = [], []
    for seed in cfg['training']['seeds']:
        checkpoint = output/f'seed_{seed}'/'best.pt'
        if not checkpoint.exists():
            raise FileNotFoundError(f'Train first; missing {checkpoint}')
        state = torch.load(checkpoint,map_location=device,weights_only=True)
        if state['signature'] != signature(cfg) or state['config']['dataset'] != cfg['dataset']:
            raise ValueError('Checkpoint data/model configuration mismatch')
        model = make_model(cfg,device)
        model.load_state_dict(state['model'])
        model.eval()
        warmed = set()
        for session,frames in sessions.items():
            common = set.intersection(*(set(f['nodes']) for f in frames))
            groups = choose_groups(common,cfg['evaluation'])
            buckets = defaultdict(list)
            costs = defaultdict(list)
            for sample in frames:
                gt = annotations(sample['gt'],cfg['dataset'])
                cloud_by_node = {n:load_cloud(sample['nodes'][n],cfg['dataset']['bounds']) for n in common}
                for nodes in groups:
                    clouds = [cloud_by_node[n] for n in nodes]
                    raw_bytes = sum(payload(p) for p in clouds)
                    for strategy in cfg['evaluation']['strategies']:
                        if strategy not in warmed:
                            infer(model,clouds,cfg,device,strategy)
                            warmed.add(strategy)
                        preds,elapsed,representation = infer(model,clouds,cfg,device,strategy)
                        key = (nodes,strategy)
                        buckets[key].append((preds,gt))
                        costs[key].append((elapsed,raw_bytes,representation))
            for (nodes,strategy),values in buckets.items():
                base = {'seed':seed,'session':session,'weather':frames[0]['weather'],
                        'nodes':','.join(nodes),'node_count':len(nodes),'strategy':strategy}
                all_metrics.extend(base | row for row in evaluate(values,cfg['dataset']['classes'],cfg['evaluation']['iou_thresholds']))
                measured = np.array(costs[(nodes,strategy)])
                cost_base = base | {'frames':len(values),'inference_decode_ms_mean':float(measured[:,0].mean()),
                                    'inference_decode_ms_p95':float(np.quantile(measured[:,0],.95))}
                all_costs.extend(network_rows(cfg,cost_base,float(measured[:,1].mean()),float(measured[:,2].mean())))
            print(f'Evaluated seed={seed} session={session}, frames={len(frames)}, subsets={len(groups)}',flush=True)
            write_csv(output/'accuracy.csv',all_metrics)
            write_csv(output/'latency_communication.csv',all_costs)
    return all_metrics


def report(cfg):
    output = Path(cfg['output']['root'])
    with (output/'accuracy.csv').open() as stream:
        accuracy = list(csv.DictReader(stream))
    with (output/'latency_communication.csv').open() as stream:
        costs = list(csv.DictReader(stream))

    def number(row, key):
        return float(row[key]) if row[key] not in ('', None) else None

    metric_names = ('ap_3d','precision','recall','f1','mean_matched_iou_3d',
                    'mean_center_error_m','mean_yaw_error_deg','mean_size_iou')
    dimensions = ('weather','node_count','strategy','class','iou')
    aggregates = defaultdict(list)
    for row in accuracy:
        if number(row, 'ap_3d') is not None:
            aggregates[tuple(row[k] for k in dimensions)].append(row)
    summary = []
    for key, rows in sorted(aggregates.items()):
        values = {name:[number(row,name) for row in rows if number(row,name) is not None]
                  for name in metric_names}
        result = dict(zip(dimensions,key))
        labels = {'ap_3d':'ap','mean_matched_iou_3d':'matched_iou_3d',
                  'mean_center_error_m':'center_error_m','mean_yaw_error_deg':'yaw_error_deg',
                  'mean_size_iou':'size_iou'}
        for name, data in values.items():
            label = labels.get(name,name)
            result[f'mean_{label}'] = float(np.mean(data)) if data else None
            result[f'std_{label}'] = float(np.std(data)) if data else None
        result['observations'] = len(rows)
        result['matched_observations'] = len(values['mean_matched_iou_3d'])
        summary.append(result)
    write_csv(output/'accuracy_summary.csv',summary)

    overview_groups = defaultdict(list)
    overview_dimensions = ('weather','node_count','strategy','iou')
    for row in accuracy:
        if number(row, 'ap_3d') is not None:
            overview_groups[tuple(row[k] for k in overview_dimensions)].append(row)
    overview = []
    for key, rows in sorted(overview_groups.items()):
        tp = sum(int(row['true_positives']) for row in rows)
        fp = sum(int(row['false_positives']) for row in rows)
        fn = sum(int(row['false_negatives']) for row in rows)
        macro = lambda name: float(np.mean([number(row,name) for row in rows]))
        micro_precision = tp/(tp+fp) if tp+fp else 0.0
        micro_recall = tp/(tp+fn) if tp+fn else 0.0
        micro_f1 = (2*micro_precision*micro_recall/(micro_precision+micro_recall)
                    if micro_precision+micro_recall else 0.0)
        matched = [row for row in rows if number(row,'mean_matched_iou_3d') is not None]
        matched_count = sum(int(row['true_positives']) for row in matched)
        weighted = lambda name: (sum(number(row,name)*int(row['true_positives']) for row in matched)/matched_count
                                 if matched_count else None)
        overview.append(dict(zip(overview_dimensions,key)) | {
            'map_3d_macro':macro('ap_3d'),'precision_macro':macro('precision'),
            'recall_macro':macro('recall'),'f1_macro':macro('f1'),
            'precision_micro':micro_precision,'recall_micro':micro_recall,'f1_micro':micro_f1,
            'true_positives':tp,'false_positives':fp,'false_negatives':fn,
            'mean_matched_iou_3d':weighted('mean_matched_iou_3d'),
            'mean_center_error_m':weighted('mean_center_error_m'),
            'mean_yaw_error_deg':weighted('mean_yaw_error_deg'),
            'mean_size_iou':weighted('mean_size_iou'),'class_observations':len(rows)})
    write_csv(output/'metrics_overview.csv',overview)

    paired = defaultdict(lambda: defaultdict(list))
    for row in accuracy:
        ap, recall = number(row, 'ap_3d'), number(row, 'recall')
        if ap is not None:
            key = tuple(row[k] for k in ('seed','session','weather','strategy','class','iou'))
            paired[key][int(row['node_count'])].append((ap, recall))
    gains = []
    for key, counts in sorted(paired.items()):
        if 1 not in counts:
            continue
        baseline_ap = [v[0] for v in counts[1]]
        baseline_recall = [v[1] for v in counts[1]]
        for count, values in sorted(counts.items()):
            if count==1:
                continue
            aps, recalls = zip(*values)
            mean_ap, mean_recall = float(np.mean(aps)), float(np.mean(recalls))
            gains.append(dict(zip(('seed','session','weather','strategy','class','iou'),key)) |
                         {'node_count':count,'multi_node_ap_mean':mean_ap,
                          'single_node_ap_mean':float(np.mean(baseline_ap)),
                          'single_node_ap_best':max(baseline_ap),
                          'ap_gain_over_mean_single':mean_ap-float(np.mean(baseline_ap)),
                          'ap_gain_over_best_single':mean_ap-max(baseline_ap),
                          'multi_node_recall_mean':mean_recall,
                          'single_node_recall_mean':float(np.mean(baseline_recall)),
                          'single_node_recall_best':max(baseline_recall),
                          'recall_gain_over_mean_single':mean_recall-float(np.mean(baseline_recall)),
                          'recall_gain_over_best_single':mean_recall-max(baseline_recall)})
    write_csv(output/'cooperation_gains.csv',gains)

    weather_groups = defaultdict(list)
    for row in gains:
        key = tuple(row[k] for k in ('weather','node_count','strategy','class','iou'))
        weather_groups[key].append(row)
    weather_summary = []
    for key, values in sorted(weather_groups.items()):
        ap = [v['ap_gain_over_mean_single'] for v in values]
        recall = [v['recall_gain_over_mean_single'] for v in values]
        weather_summary.append(dict(zip(dimensions,key)) | {
            'mean_ap_gain':float(np.mean(ap)), 'std_ap_gain':float(np.std(ap)),
            'ap_positive_rate':float(np.mean(np.asarray(ap)>0)),
            'mean_recall_gain':float(np.mean(recall)), 'std_recall_gain':float(np.std(recall)),
            'recall_positive_rate':float(np.mean(np.asarray(recall)>0)),
            'paired_observations':len(values)})
    write_csv(output/'weather_cooperation_summary.csv',weather_summary)

    scaling = []
    for key, counts in sorted(paired.items()):
        ordered = sorted(counts)
        for previous, current in zip(ordered, ordered[1:]):
            previous_ap = float(np.mean([v[0] for v in counts[previous]]))
            current_ap = float(np.mean([v[0] for v in counts[current]]))
            previous_recall = float(np.mean([v[1] for v in counts[previous]]))
            current_recall = float(np.mean([v[1] for v in counts[current]]))
            scaling.append(dict(zip(('seed','session','weather','strategy','class','iou'),key)) | {
                'from_node_count':previous, 'to_node_count':current,
                'ap_change':current_ap-previous_ap,
                'recall_change':current_recall-previous_recall})
    write_csv(output/'node_scaling.csv',scaling)

    costs_by_run = defaultdict(list)
    run_keys = ('seed','session','weather','nodes','node_count','strategy')
    for row in costs:
        costs_by_run[tuple(row[k] for k in run_keys)].append(row)
    tradeoffs = []
    for row in accuracy:
        for cost in costs_by_run[tuple(row[k] for k in run_keys)]:
            tradeoffs.append({k:row[k] for k in run_keys} | {
                'class':row['class'],'iou':row['iou'],'ap_3d':row['ap_3d'],
                'precision':row['precision'],'recall':row['recall'],'f1':row['f1'],
                'placement':cost['placement'],
                'bandwidth_mbps':cost['bandwidth_mbps'],'payload_bytes':cost['payload_bytes'],
                'inference_decode_ms_mean':cost['inference_decode_ms_mean'],
                'inference_decode_ms_p95':cost['inference_decode_ms_p95'],
                'transmission_ms_assumed':cost['transmission_ms_model'],
                'end_to_end_ms_assumed':cost['end_to_end_ms_model']})
    write_csv(output/'accuracy_latency_tradeoffs.csv',tradeoffs)

    if cfg['output']['plots'] and summary:
        try:
            import matplotlib
        except ImportError:
            print('Report plot skipped: install matplotlib from requirements.txt', flush=True)
            matplotlib = None
        if matplotlib is not None:
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            grouped = defaultdict(list)
            for row in summary:
                grouped[(row['strategy'], row['class'])].append(
                    (int(row['node_count']), float(row['mean_ap'])))
            figure, axis = plt.subplots(figsize=(10, 6))
            for (strategy, label), values in sorted(grouped.items()):
                by_count = defaultdict(list)
                for count, value in values:
                    by_count[count].append(value)
                x = sorted(by_count)
                y = [float(np.mean(by_count[count])) for count in x]
                axis.plot(x, y, marker='o', label=f'{strategy}/{label}')
            axis.set(xlabel='node count', ylabel='mean AP', title='Test AP by node count and fusion')
            axis.set_ylim(0, 1)
            axis.legend(fontsize=8, ncol=2)
            axis.grid(alpha=.25)
            figure.tight_layout()
            figure.savefig(output/'accuracy_by_nodes.png', dpi=160)
            plt.close(figure)
            if weather_summary:
                figure, axis = plt.subplots(figsize=(10, 6))
                grouped = defaultdict(list)
                for row in weather_summary:
                    grouped[(row['weather'], row['strategy'])].append(float(row['mean_ap_gain']))
                labels, values = [], []
                for key, group in sorted(grouped.items()):
                    labels.append('/'.join(key))
                    values.append(float(np.mean(group)))
                axis.bar(labels, values)
                axis.axhline(0, color='black', linewidth=.8)
                axis.set(ylabel='mean paired AP gain', title='Cooperation gain by weather and fusion')
                axis.tick_params(axis='x', rotation=45)
                figure.tight_layout()
                figure.savefig(output/'cooperation_gain_by_weather.png', dpi=160)
                plt.close(figure)

            figure, axis = plt.subplots(figsize=(9, 5))
            latency_groups = defaultdict(list)
            for row in costs:
                latency_groups[(row['placement'], float(row['bandwidth_mbps']))].append(
                    float(row['end_to_end_ms_model']))
            for placement in ('edge','cloud'):
                points = sorted((bandwidth, float(np.mean(values)))
                                for (name, bandwidth), values in latency_groups.items()
                                if name == placement)
                if points:
                    axis.plot([p[0] for p in points], [p[1] for p in points],
                              marker='o', label=placement)
            axis.set(xlabel='assumed bandwidth (Mbps)', ylabel='mean assumed end-to-end latency (ms)',
                     title='Analytical latency by placement and bandwidth', xscale='log')
            axis.legend()
            axis.grid(alpha=.25)
            figure.tight_layout()
            figure.savefig(output/'latency_by_bandwidth.png', dpi=160)
            plt.close(figure)
    advercity = cfg['dataset'].get('type') == 'advercity'
    primary = ('Primary question: how much do additional simulated vehicle and roadside LiDAR agents improve detection?'
               if advercity else
               'Primary question: how much do additional real roadside LiDAR nodes improve detection?')
    secondary = ('Secondary question: how does that cooperation gain vary across controlled simulated weather categories?'
                 if advercity else
                 'Secondary question: does that cooperation gain persist across recorded weather conditions?')
    weather_limit = ('Adver-City weather is simulated and categorical; no numerical meteorological visibility is inferred.'
                     if advercity else
                     'Weather sessions differ in traffic/date; weather comparisons alone do not establish a causal weather effect.')
    lines = ['# ECCO cooperation-first experiment report','', primary, secondary,
             '',
             'Accuracy: rotated 3D, globally confidence-ranked AP within each session/subset/seed.',
             'Precision, recall, and F1 use the configured score threshold after class-wise NMS.',
             'Matched-object diagnostics include 3D IoU, center error, yaw error, and dimension IoU.',
             'All aggregates are descriptive means, not confidence intervals or causal weather estimates.',
             'Single-node rows cover every common real node. The fixed global ROI/GT is unchanged across counts.',
             'Measured inference+decode includes serial node execution and NMS; excludes loading/rasterization/serialization.',
             'End-to-end latency is an analytical estimate using configured literature-derived assumptions.',
             'Logical payloads use float32 and zlib level 1; protocol headers and packet loss are not modeled.',
             'Cloud sends raw cropped points; edge sends raw points/features/local predictions according to strategy.',
             f'Assumption source: {cfg["deployment_assumptions"]["source"]}.',
             weather_limit,
             'cooperation_gains.csv uses paired comparisons within session/seed/class/strategy/IoU.',
             '', '## Cooperation gain by weather', '',
             '| Weather | Nodes | Fusion | Class | IoU | AP gain | Recall gain | AP positive rate |',
             '|---|---:|---|---|---:|---:|---:|---:|']
    for row in weather_summary:
        lines.append(f'| {row["weather"]} | {row["node_count"]} | {row["strategy"]} | '
                     f'{row["class"]} | {row["iou"]} | {row["mean_ap_gain"]:.4f} | '
                     f'{row["mean_recall_gain"]:.4f} | {row["ap_positive_rate"]:.2f} |')
    lines.extend(['', '## Absolute accuracy', '',
                  '| Weather | Nodes | Fusion | Class | IoU | Mean AP | Precision | Recall | F1 | Center error (m) |',
                  '|---|---:|---|---|---:|---:|---:|---:|---:|---:|'])
    for row in summary:
        center = ('n/a' if row['mean_center_error_m'] is None else
                  f'{row["mean_center_error_m"]:.3f}')
        lines.append(f'| {row["weather"]} | {row["node_count"]} | {row["strategy"]} | '
                     f'{row["class"]} | {row["iou"]} | {row["mean_ap"]:.4f} | '
                     f'{row["mean_precision"]:.4f} | {row["mean_recall"]:.4f} | '
                     f'{row["mean_f1"]:.4f} | {center} |')
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print(f'Report: {output / "report.md"}')


def verify_results(cfg):
    """Fail the unattended run if required artifacts or metric invariants are invalid."""
    output = Path(cfg['output']['root'])
    required = ['index.json','inventory.json','inspection.json','resolved_config.json',
                'environment.json','accuracy.csv','accuracy_summary.csv','metrics_overview.csv',
                'latency_communication.csv','accuracy_latency_tradeoffs.csv','report.md']
    allowed_empty = ['cooperation_gains.csv','weather_cooperation_summary.csv','node_scaling.csv']
    checks, errors = {}, []
    for name in required:
        path = output/name
        checks[f'artifact:{name}'] = path.is_file() and path.stat().st_size > 0
        if not checks[f'artifact:{name}']:
            errors.append(f'missing or empty artifact: {name}')
    for name in allowed_empty:
        path = output/name
        checks[f'artifact:{name}'] = path.is_file()
        if not checks[f'artifact:{name}']:
            errors.append(f'missing artifact: {name}')
    for seed in cfg['training']['seeds']:
        for name in ('best.pt','latest.pt','complete.json',f'training_metrics_seed_{seed}.csv'):
            path = output/f'seed_{seed}'/name
            checks[f'seed_{seed}:{name}'] = path.is_file() and path.stat().st_size > 0
            if not checks[f'seed_{seed}:{name}']:
                errors.append(f'missing or empty seed artifact: seed_{seed}/{name}')
    metric_rows = []
    if (output/'accuracy.csv').is_file():
        with (output/'accuracy.csv').open() as stream:
            metric_rows = list(csv.DictReader(stream))
    checks['accuracy_rows_present'] = bool(metric_rows)
    if not metric_rows:
        errors.append('accuracy.csv contains no metric rows')
    bounded = ('ap_3d','precision','recall','f1','mean_matched_iou_3d','mean_size_iou')
    for line, row in enumerate(metric_rows, start=2):
        for name in bounded:
            value = row.get(name)
            if value not in ('',None) and (not math.isfinite(float(value)) or not 0 <= float(value) <= 1):
                errors.append(f'accuracy.csv line {line}: invalid {name}={value}')
        tp, fp, fn = (int(row[name]) for name in ('true_positives','false_positives','false_negatives'))
        if tp+fp != int(row['predictions']) or tp+fn != int(row['ground_truth']):
            errors.append(f'accuracy.csv line {line}: TP/FP/FN totals are inconsistent')
        for name in ('mean_center_error_m','mean_yaw_error_deg'):
            value = row.get(name)
            if value not in ('',None) and (not math.isfinite(float(value)) or float(value) < 0):
                errors.append(f'accuracy.csv line {line}: invalid {name}={value}')
    checks['metric_invariants'] = not errors
    result = {'timestamp_utc':utc_now(),'status':'passed' if not errors else 'failed',
              'accuracy_rows':len(metric_rows),'checks':checks,'errors':errors}
    write_json(output/'verification.json',result)
    if errors:
        raise RuntimeError('Result verification failed; see verification.json: ' + '; '.join(errors[:5]))
    print(f'Result verification passed: {output / "verification.json"}', flush=True)
    return result
