from .common import *
import torch,gc,time,lightgbm as lgb
from .pv_nhitsx import PVNHITSx
from .pv_timemixer import PVTimeMixer
from .lightgbm_features import build_lightgbm_features,DEFAULT_NWP_METADATA_FEATURE_NAMES
from .solar4 import build as solar4,NAMES
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
def model(name):
    conf=json.loads((R/'architecture_configs.json').read_text())[name]
    return {'nhitsx':PVNHITSx,'timemixer':PVTimeMixer}[name](**conf),conf
class Data:
    def __init__(self,site,role):
        self.root=R/site;self.a={p.stem:np.load(p,mmap_mode='r') for p in (self.root/'data'/role).glob('*.npy')};self.n=len(self.a['origins'])
        z=json.loads((self.root/'normalizer.json').read_text());self.mean=np.array(z['mean']);self.scale=np.array(z['scale'])
    def batch(self,ix,device=DEVICE):
        a=self.a;b={'past_power':np.array(a['past_power'][ix]),'past_features':np.array(a['past_A'][ix]),'future_features':np.array(a['future_A'][ix]),'past_observed_mask':np.array(a['observed'][ix]),'past_complete_mask':np.array(a['observed'][ix]),'past_source_span_mask':np.ones_like(a['observed'][ix]),'station_index':np.zeros(len(ix),np.int64),'station_coordinates':np.array(a['coordinates'][ix]),'target':np.array(a['target'][ix])}
        for k in ('past_nwp','future_nwp'):
            x=np.array(a[k][ix]);b[k+'_mask']=np.isfinite(x);b[k]=np.where(b[k+'_mask'],(x-self.mean)/self.scale,0).astype(np.float32)
        return {k:torch.as_tensor(v,device=device) for k,v in b.items()}
def features(d,ix):
    b={k:v.numpy() for k,v in d.batch(ix,'cpu').items()};b['power_scale_w']=np.array(d.a['capacity'][ix]);n=len(ix)
    b['recent_5min_power']=np.zeros((n,24,1),np.float32)
    for k in ('recent_5min_complete_mask','recent_5min_observed_mask','recent_5min_source_span_mask'):b[k]=np.zeros((n,24,1),bool)
    for k,length in (('past_nwp_metadata',96),('future_nwp_metadata',16)):b[k]=np.full((n,length,len(DEFAULT_NWP_METADATA_FEATURE_NAMES)),np.nan,np.float32)
    f=build_lightgbm_features(b,causal_48h_power=d.a['p48'][ix],time_feature_names=TIME_NAMES,nwp_feature_names=NWP_NAMES)
    return f,solar4(f.shared,f.horizon,list(f.shared_feature_names),list(f.horizon_feature_names))
def prepare_tree(site):
    out=R/site;manifest={}
    for role in ROLES:
        d=Data(site,role);folder=out/'tree_features'/role;folder.mkdir(parents=True,exist_ok=True);f,e=features(d,np.arange(2))
        schema={'shared_names':list(f.shared_feature_names),'horizon_names':list(f.horizon_feature_names),'extra_names':NAMES}
        assert len(schema['shared_names'])+len(schema['horizon_names'])+4==311
        buffers={k:np.lib.format.open_memmap(folder/(k+'.npy'),mode='w+',shape=shape,dtype='float32') for k,shape in {'shared':(d.n,f.shared.shape[1]),'horizon':(d.n,16,f.horizon.shape[2]),'solar4':(d.n,16,4)}.items()}
        for start in range(0,d.n,512):
            ix=np.arange(start,min(start+512,d.n));f,e=features(d,ix);buffers['shared'][ix]=f.shared;buffers['horizon'][ix]=f.horizon;buffers['solar4'][ix]=e
        for v in buffers.values():v.flush()
        del buffers;gc.collect()
        for p in folder.glob('*.npy'):manifest[str(p)]=sha(p)
        print('TREE_FEATURES',site,role,d.n,flush=True)
    save(out/'tree_features.json',{'schema':schema,'hashes':manifest})
def tree(site):
    freeze_check();out=R/site;ds={r:Data(site,r) for r in ROLES};xx={r:{k:np.load(out/'tree_features'/r/(k+'.npy'),mmap_mode='r') for k in ('shared','horizon','solar4')} for r in ROLES};schema=json.loads((out/'tree_features.json').read_text())['schema'];names=schema['shared_names']+schema['horizon_names']+NAMES
    ledger=[];(out/'trees').mkdir(exist_ok=True)
    for h in range(16):
        path=out/'trees'/f'h{h+1:02}.txt';assert not path.exists(),'Existing fit requires explicit recovery review'
        mat=lambda r:np.column_stack([xx[r]['shared'],xx[r]['horizon'][:,h],xx[r]['solar4'][:,h]])
        seed=2027+h;params={**json.loads((R/'tree_parameters.json').read_text()),'num_threads':8,'seed':seed,'feature_fraction_seed':seed,'bagging_seed':seed,'data_random_seed':seed}
        tr=lgb.Dataset(mat('train'),label=ds['train'].a['target'][:,h],feature_name=names,categorical_feature=['station_index']);va=lgb.Dataset(mat('early_stop'),label=ds['early_stop'].a['target'][:,h],reference=tr,feature_name=names,categorical_feature=['station_index']);start=time.monotonic()
        m=lgb.train(params,tr,num_boost_round=1800,valid_sets=[va],callbacks=[lgb.early_stopping(120,verbose=False)]);m.save_model(str(path));ledger.append({'horizon':h+1,'best_iteration':m.best_iteration,'seconds':time.monotonic()-start,'sha256':sha(path)});save(out/'tree_progress.json',ledger);print('TREE',site,h+1,ledger[-1],flush=True);del m,tr,va;gc.collect()
def neural(site,name,seed):
    freeze_check();out=R/site/'models';out.mkdir(exist_ok=True);path=out/f'{name}_{seed}.pt';assert not path.exists()
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed);net,conf=model(name);net=net.to(DEVICE);opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-4,fused=(DEVICE == "cuda"));d=Data(site,'train');ledger=[];start=time.monotonic()
    for epoch in range(12):
        net.train();order=rng.permutation(d.n);total=0
        for i in range(0,d.n,256):
            ix=order[i:i+256];b=d.batch(ix);opt.zero_grad(set_to_none=True)
            with precision_context():loss=(net(b).squeeze(-1).float()-b['target']).abs().mean()
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1,error_if_nonfinite=True);opt.step();total+=float(loss.detach())*len(ix)
        ledger.append({'epoch':epoch+1,'loss':total/d.n,'seconds':time.monotonic()-start});save(out/f'{name}_{seed}_progress.json',ledger);print(site,name,seed,ledger[-1],flush=True)
    if name=='timemixer':
        opt=torch.optim.AdamW(net.parameters(),lr=3e-6,weight_decay=1e-4,fused=(DEVICE == "cuda"));torch.manual_seed(20260909+seed-2021);torch.cuda.manual_seed_all(20260909+seed-2021);rng=np.random.default_rng(20260909+seed-2021);order=np.empty(0,np.int64);pos=0
        for step in range(2400):
            while len(order)-pos<256:order=np.r_[order[pos:],rng.permutation(d.n)];pos=0
            ix=order[pos:pos+256];pos+=256;b=d.batch(ix);opt.zero_grad(set_to_none=True)
            with precision_context():loss=(net(b).squeeze(-1).float()-b['target']).abs().mean()
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1,error_if_nonfinite=True);opt.step()
            if (step+1)%200==0:print(site,name,seed,'LOWLR',step+1,flush=True)
    torch.save({'model_state_dict':{k:v.detach().cpu() for k,v in net.state_dict().items()},'config':conf,'seed':seed,'freeze_sha':sha(R/'freeze.json'),'training_seconds':time.monotonic()-start},path)
def predict(site):
    freeze_check();out=R/site;folder=out/'predictions';folder.mkdir(exist_ok=True);manifest={}
    for role in ('early_stop','inner','outer','evaluation'):
        d=Data(site,role);x={k:np.load(out/'tree_features'/role/(k+'.npy'),mmap_mode='r') for k in ('shared','horizon','solar4')};pred=np.empty((d.n,16),np.float32)
        for h in range(16):
            m=lgb.Booster(model_file=str(out/'trees'/f'h{h+1:02}.txt'));pred[:,h]=np.maximum(m.predict(np.column_stack([x['shared'],x['horizon'][:,h],x['solar4'][:,h]]),num_threads=8),0)
        np.save(folder/f'{role}_tree.npy',pred)
        for seed in (2021,2022,2023):
            for name in ('nhitsx','timemixer'):
                ck=torch.load(out/'models'/f'{name}_{seed}.pt',map_location='cpu',weights_only=False);net,_=model(name);net.load_state_dict(ck['model_state_dict']);net=net.to(DEVICE).eval()
                with torch.inference_mode():
                    for start in range(0,d.n,256):
                        ix=np.arange(start,min(start+256,d.n))
                        with precision_context():y=net(d.batch(ix)).squeeze(-1)
                        pred[ix]=np.maximum(y.float().cpu().numpy(),0)
                np.save(folder/f'{role}_{name}_{seed}.npy',pred);del net;torch.cuda.empty_cache()
        print('PREDICTED',site,role,flush=True)
    for p in folder.glob('*.npy'):manifest[str(p)]=sha(p)
    save(out/'predictions_frozen.json',{'hashes':manifest,'model_hashes':{str(p):sha(p) for p in list((out/'models').glob('*.pt'))+list((out/'trees').glob('*.txt'))}})
