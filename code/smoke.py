"""Small synthetic CPU checks. This does not reproduce a paper experiment."""
from pathlib import Path
import importlib.util
import json
import numpy as np
import torch

def run():
    from pv3.pv_nhitsx import PVNHITSx
    from pv3.pv_timemixer import PVTimeMixer
    from pv3.lightgbm_features import build_lightgbm_features, DEFAULT_NWP_METADATA_FEATURE_NAMES
    from pv3.solar4 import build as solar4
    from pv3.common import TIME_NAMES, NWP_NAMES
    from pv3.score import fit

    torch.set_num_threads(2)
    torch.manual_seed(2021)
    rng = np.random.default_rng(2021)
    b, l, h = 2, 96, 16
    arrays = {
        'past_power': rng.random((b,l,1),dtype=np.float32),
        'past_features': rng.random((b,l,13),dtype=np.float32),
        'future_features': rng.random((b,h,13),dtype=np.float32),
        'past_observed_mask': np.ones((b,l,1),bool),
        'past_complete_mask': np.ones((b,l,1),bool),
        'past_source_span_mask': np.ones((b,l,1),bool),
        'past_nwp': rng.random((b,l,27),dtype=np.float32),
        'future_nwp': rng.random((b,h,27),dtype=np.float32),
        'past_nwp_mask': np.ones((b,l,27),bool),
        'future_nwp_mask': np.ones((b,h,27),bool),
        'station_coordinates': np.tile(np.array([33.6762,-83.6760],np.float32),(b,1)),
        'station_index': np.zeros(b,np.int64),
    }
    batch = {k:torch.as_tensor(v) for k,v in arrays.items()}
    target = torch.as_tensor(rng.random((b,h,1),dtype=np.float32))
    config = json.loads((Path(__file__).parent/'configs/architecture_configs.json').read_text())
    expected = {'nhitsx':2270692,'timemixer':3509419}
    results = {}
    for name,cls in [('nhitsx',PVNHITSx),('timemixer',PVTimeMixer)]:
        net=cls(**config[name]).cpu()
        count=sum(p.numel() for p in net.parameters())
        assert count==expected[name],(name,count)
        opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-4)
        out=net(batch);assert out.shape==(b,h,1) and torch.isfinite(out).all()
        loss=(out-target).abs().mean();loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1,error_if_nonfinite=True)
        opt.step();opt.zero_grad(set_to_none=True)
        net.eval()
        with torch.no_grad(): assert torch.isfinite(net(batch)).all()
        results[name]={'parameters':count,'output_shape':list(out.shape),'synthetic_loss':float(loss.detach())}
    features = dict(arrays)
    features['power_scale_w']=np.full(b,38687000,np.float32)
    features['recent_5min_power']=np.zeros((b,24,1),np.float32)
    for k in ('recent_5min_complete_mask','recent_5min_observed_mask','recent_5min_source_span_mask'):
        features[k]=np.zeros((b,24,1),bool)
    for k,n in [('past_nwp_metadata',96),('future_nwp_metadata',16)]:
        features[k]=np.full((b,n,len(DEFAULT_NWP_METADATA_FEATURE_NAMES)),np.nan,np.float32)
    f=build_lightgbm_features(features,causal_48h_power=np.full((b,16),np.nan,np.float32),time_feature_names=TIME_NAMES,nwp_feature_names=NWP_NAMES)
    extra=solar4(f.shared,f.horizon,list(f.shared_feature_names),list(f.horizon_feature_names))
    assert f.shared.shape[1]+f.horizon.shape[2]+extra.shape[2]==311
    a=rng.random((24,16));z=rng.random((24,16));m=np.ones((24,16),bool)
    weights=fit(.7*a+.3*z,a,z,m)
    assert np.max(abs(weights-.7))<1e-6
    assert np.all(fit(a,a,z,m)==1.)
    results['features']={'direct_tree_columns':311}
    results['fusion']={'max_weight_error':float(np.max(abs(weights-.7))),'endpoint_selection':'PASS'}
    if importlib.util.find_spec('lightgbm'):
        import lightgbm as lgb
        x=np.column_stack([f.shared,f.horizon[:,0],extra[:,0]])
        x=np.tile(x,(16,1));y=rng.random(len(x))
        params={'objective':'regression_l1','metric':'l1','num_threads':2,'num_leaves':3,'min_data_in_leaf':2,'verbosity':-1}
        tree=lgb.train(params,lgb.Dataset(x,label=y),num_boost_round=2)
        assert np.isfinite(tree.predict(x,num_threads=2)).all()
        results['tree']={'tiny_synthetic_fit':'PASS','rounds':2,'paper_budget_used':False}
    else:
        results['tree']={'tiny_synthetic_fit':'NOT_RUN','reason':'LightGBM not installed'}
    return {'status':'PASS','scope':'synthetic CPU API checks only; no real data read, no full training or paper-result reproduction','checks':results}

if __name__=='__main__':
    print(json.dumps(run(),indent=2))
