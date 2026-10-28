"""Portable run paths and integrity checks for the archived two-site recipe."""
import os, sys, json, hashlib, csv, datetime
from pathlib import Path
from contextlib import nullcontext

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '4')

import numpy as np
import pandas as pd
from .lightgbm_features import DEFAULT_TIME_FEATURE_NAMES
from .gfs_adapter import FEATURE_NAMES as NWP_NAMES

R = Path(os.environ.get('PV3_RUN_DIR', 'runs')).resolve()
SOURCE = Path(os.environ.get('PV3_SOURCE_DIR', str(R / 'source'))).resolve()
DEVICE = os.environ.get('PV3_DEVICE', 'cuda')
TIME_NAMES = ('solar_hour_angle_sin', 'solar_hour_angle_cos',
              'solar_declination_sin', 'solar_declination_cos',
              'solar_ecliptic_longitude_sin', 'solar_ecliptic_longitude_cos') + DEFAULT_TIME_FEATURE_NAMES[6:]
CAP = {'9069': 38687.0, '9068': 4738.0}
ROLES = ('train', 'early_stop', 'inner', 'outer', 'evaluation')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)

def register(stage, message, artifact):
    """Write only to this user-specified run; no private project registry."""
    R.mkdir(parents=True, exist_ok=True)
    with (R / 'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps({'stage': stage, 'message': message, 'artifact': str(artifact)}, ensure_ascii=False) + '\n')

def precision_context():
    import torch
    return torch.autocast('cuda', dtype=torch.bfloat16) if DEVICE == 'cuda' else nullcontext()

def freeze_check():
    manifest = json.loads((R / 'freeze.json').read_text(encoding='utf-8'))
    for relative, digest in manifest['hashes'].items():
        path = R / relative
        if sha(path) != digest:
            raise ValueError(f'Frozen input changed: {relative}')
    return manifest

def freeze(sites):
    """Hash prepared inputs before any fit; preserve known data conventions."""
    required_shapes = {
        'past_power': (96, 1), 'observed': (96, 1), 'past_A': (96, 13),
        'future_A': (16, 13), 'past_nwp': (96, 27), 'future_nwp': (16, 27),
        'target': (16,), 'p48': (16,), 'daylight': (16,), 'coordinates': (2,),
        'capacity': (), 'origins': (),
    }
    paths = [R / 'architecture_configs.json', R / 'tree_parameters.json']
    counts = {}
    for site in sites:
        counts[site] = {}
        for role in ROLES:
            folder = R / site / 'data' / role
            target = np.load(folder / 'target.npy', mmap_mode='r', allow_pickle=False)
            n = len(target)
            if n == 0:
                raise ValueError(f'Empty split: {site}/{role}')
            for name, tail in required_shapes.items():
                path = folder / (name + '.npy')
                x = np.load(path, mmap_mode='r', allow_pickle=False)
                if x.shape != (n, *tail):
                    raise ValueError(f'{site}/{role}/{name}: unexpected shape {x.shape}')
                if name != 'p48' and not np.isfinite(x).all():
                    raise ValueError(f'{site}/{role}/{name}: incomplete inputs')
            origins = np.load(folder / 'origins.npy', allow_pickle=False)
            if not np.all(np.diff(origins) > 0):
                raise ValueError(f'Issue times must be unique and increasing: {site}/{role}')
            counts[site][role] = n
            paths.extend(sorted(folder.glob('*.npy')))
            paths.extend(sorted(folder.glob('*.csv')))
            paths.extend(sorted((R / site / 'tree_features' / role).glob('*.npy')))
        paths.extend([R / site / 'normalizer.json', R / site / 'tree_features.json'])
    manifest = {
        'status': 'PREPARED_INPUTS_FROZEN', 'sites': list(sites), 'counts': counts,
        'hashes': {p.relative_to(R).as_posix(): sha(p) for p in paths},
        'seeds': [2021, 2022, 2023], 'tree_seeds': '2027+h for h=0..15',
        'evaluation_exposed': True, 'weather_availability': 'initialization+6h research assumption',
        'source_validation_boundary': 'Array checks do not replace upstream timestamp/channel/NWP provenance checks',
    }
    save(R / 'freeze.json', manifest)
    return manifest
