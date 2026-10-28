"""Replay frozen PVDAQ forecasts. No fitting, inference, or interval computation."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def metric(y, prediction, daylight):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    m = np.asarray(daylight, dtype=bool)
    assert y.shape == p.shape == m.shape and y.shape[1] == 16
    assert np.isfinite(y).all() and np.isfinite(p).all()
    count = m.sum(axis=0)
    assert (count > 0).all()
    # y and p are P/K: normalize exactly once at source, then express in percent.
    horizon = 100*np.sum(np.abs(p-y)*m, axis=0)/count
    return {'all16_pct':float(horizon.mean()),'horizon_pct':horizon.tolist(),
            'daylight_cells':int(count.sum()),'issues':int(y.shape[0])}

def replay(root):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    for name, digest in manifest['bundle_sha256'].items():
        assert sha(root/name) == digest, f'Hash mismatch: {name}'
    expected = json.loads((root/'expected_scores.json').read_text(encoding='utf-8'))
    output = {}; maxdiff = 0.0
    for site in ('2107','9069','9068'):
        a = np.load(root/(site+'.npz'), allow_pickle=False)
        assert len(np.unique(a['issue_utc_seconds'])) == len(a['issue_utc_seconds'])
        assert np.all(np.diff(a['issue_utc_seconds']) > 0)
        scores = {name:metric(a['target'],a[name],a['daylight'])
                  for name in a.files if name not in manifest['metadata_members']}
        for name,v in scores.items():
            target = expected[site][name]
            maxdiff = max(maxdiff, abs(v['all16_pct']-target))
            # The archived additional-site scores use some float32 reductions;
            # this replay uses float64. 1e-5 pp is below the printed 1e-4 pp precision.
            assert abs(v['all16_pct']-target) < 1e-5, (site,name,v['all16_pct'],target)
        summary = {}
        for method in ('nhitsx','timemixer','tree_timemixer','static3'):
            names = [f'{seed}_{method}' for seed in (2021,2022,2023)]
            if all(name in scores for name in names):
                values = np.array([scores[name]['all16_pct'] for name in names])
                summary[method] = {'mean_pct':float(values.mean()),'sample_sd_pct':float(values.std(ddof=1)),
                                   'per_seed_pct':values.tolist()}
        output[site] = {'scores':scores,'seed_summaries':summary}
    return {'status':'PASS','max_replay_difference_pp':maxdiff,'archive_float_precision_tolerance_pp':1e-5,'sites':output,
            'new_fit_or_bootstrap':False,'units':'targets and predictions already divided once by fixed DC capacity'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    result = replay(args.bundle)
    if args.output:
        args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'status':result['status'],'max_replay_difference_pp':result['max_replay_difference_pp'],
                     'three_expert_seed_summaries':{s:q['seed_summaries']['static3'] for s,q in result['sites'].items()}},indent=2))
