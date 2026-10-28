from .common import *
from . import gfs_adapter as ga
import pvlib,gc
def geometry(tt,lat,lon):
    idx=pd.to_datetime(tt-450,unit='s',utc=True)
    pos=pvlib.solarposition.get_solarposition(idx,lat,lon,altitude=10)
    el=np.deg2rad(pos.elevation.to_numpy());az=np.deg2rad(pos.azimuth.to_numpy());phi=np.deg2rad(lat)
    dec=np.arcsin(np.clip(np.sin(el)*np.sin(phi)+np.cos(el)*np.cos(az)*np.cos(phi),-1,1))
    hour=np.arctan2(-np.cos(el)*np.sin(az),np.sin(el)*np.cos(phi)-np.cos(el)*np.cos(az)*np.sin(phi))
    dt=pvlib.spa.calculate_deltat(idx.year.to_numpy(),idx.month.to_numpy());jme=((tt-450)/86400+2440587.5+dt/86400-2451545)/365250
    lam=np.deg2rad(pvlib.spa.geocentric_longitude(pvlib.spa.heliocentric_longitude(jme)));day=el>0;prev=np.r_[False,day[:-1]]
    ghi=pvlib.clearsky.haurwitz(pos.apparent_zenith).ghi.to_numpy()/1000
    g=np.column_stack([np.sin(hour),np.cos(hour),np.sin(dec),np.cos(dec),np.sin(lam),np.cos(lam),np.sin(el),np.sin(az),np.cos(az),ghi,day,day&~prev,~day&prev]).astype(np.float32)
    return g,pos.elevation.to_numpy()
def run(site):
    out=R/site;out.mkdir(exist_ok=True)
    if (out/'prepared.json').exists():
        q=json.loads((out/'prepared.json').read_text());assert all(sha(p)==h for p,h in q['hashes'].items());return
    power=pd.read_csv(SOURCE/f'{site}_strict_power.csv');queue=pd.read_csv(SOURCE/f'{site}_power_only_queue.csv')
    # DST NaT rows never enter frozen power queue; preserve their missing grid slots.
    utc=pd.to_datetime(power.utc_interval_end_assumed,utc=True,errors='coerce')
    power=power.loc[utc.notna()].copy();utc=utc[utc.notna()]
    tt=np.arange(int(utc.iloc[0].timestamp()),int(utc.iloc[-1].timestamp())+900,900,dtype=np.int64)
    series=pd.Series(power.power_kw.to_numpy(),index=utc.astype('int64')//10**9)
    assert not series.index.duplicated().any()
    p=series.reindex(tt).to_numpy(np.float32)/CAP[site];valid=np.isfinite(p)
    positions=np.searchsorted(tt,queue.issue_utc_seconds.to_numpy());assert np.array_equal(tt[positions],queue.issue_utc_seconds.to_numpy())
    needed=np.unique(np.concatenate((positions[:,None]+np.arange(-96,0),positions[:,None]),axis=1))
    snap=ga.snapshot_sources(tt[needed],delays_hours=(6,),root=SOURCE/f'gfs_{site}_v3',verify_raw=True,verify_decode=True)
    # site_id=2107 in raw receipts is a legacy serializer label, not location.
    # The frozen config, native NetCDF grid and exact URL/key establish identity.
    assert all(e['status'] in ('VERIFIED','DOCUMENTED_ENDPOINT_UNAVAILABLE') for e in snap['entries']),[(e['item_id'],e['status'],e.get('error')) for e in snap['entries'] if e['status'] not in ('VERIFIED','DOCUMENTED_ENDPOINT_UNAVAILABLE')][:10]
    save(out/'source_snapshot.json',snap)
    built=ga.build_features(tt[needed],delays_hours=(6,),snapshot=snap)
    assert list(built['feature_names'])==json.loads((SOURCE/'nwp_feature_schema.json').read_text())['names']
    weather=np.full((len(tt),16,27),np.nan,np.float32);weather[needed]=built['scenarios'][6]['features']
    del built;gc.collect()
    config=snap['config'];lat=config['site']['latitude'];lon=config['site']['longitude'];geom,elev=geometry(tt,lat,lon)
    stats={};hashes={};excluded=[]
    for role in ROLES:
        mask=queue.split.eq(role).to_numpy();ix=positions[mask];q=queue.loc[mask].copy()
        keep=np.ones(len(ix),bool)
        for start in range(0,len(ix),512):
            a=ix[start:start+512];past=a[:,None]+np.arange(-96,0);future=a[:,None]+np.arange(1,17)
            assert valid[past].all() and valid[future].all()
            keep[start:start+len(a)]=np.isfinite(weather[past,0,:]).all(axis=(1,2))&np.isfinite(weather[a]).all(axis=(1,2))
        bad=q.loc[~keep].copy();bad['reason']='required_historical_or_future_nwp_incomplete';excluded.append(bad)
        q=q.loc[keep].copy();ix=ix[keep];folder=out/'data'/role;folder.mkdir(parents=True,exist_ok=True);q.to_csv(folder/'keys.csv',index=False);n=len(ix);assert n>0
        shapes={'past_power':(n,96,1),'observed':(n,96,1),'past_A':(n,96,13),'future_A':(n,16,13),'past_nwp':(n,96,27),'future_nwp':(n,16,27),'target':(n,16),'p48':(n,16),'daylight':(n,16)}
        buf={k:np.lib.format.open_memmap(folder/(k+'.npy'),mode='w+',shape=v,dtype='bool' if k in ('observed','daylight') else 'float32') for k,v in shapes.items()}
        for start in range(0,n,512):
            a=ix[start:start+512];past=a[:,None]+np.arange(-96,0);future=a[:,None]+np.arange(1,17);old=future-192;safe=np.maximum(old,0);sl=slice(start,start+len(a))
            buf['past_power'][sl,:,0]=p[past];buf['observed'][sl]=True;buf['past_A'][sl]=geom[past];buf['future_A'][sl]=geom[future];buf['past_nwp'][sl]=weather[past,0];buf['future_nwp'][sl]=weather[a];buf['target'][sl]=p[future];buf['p48'][sl]=np.where((old>=0)&valid[safe],p[safe],np.nan);buf['daylight'][sl]=elev[future]>5
        for v in buf.values():v.flush()
        daylight=np.array(buf['daylight']);stats[role]={'power_candidates':len(keep),'retained':n,'nwp_excluded':int((~keep).sum()),'daylight_cells':int(daylight.sum()),'per_horizon_daylight':daylight.sum(axis=0).tolist(),'all_cells':n*16}
        for k,v in {'origins':tt[ix],'positions':ix,'capacity':np.full(n,CAP[site]*1000,np.float32),'coordinates':np.tile(np.array([lat,lon],np.float32),(n,1)),'target_qc':np.ones((n,16),bool)}.items():np.save(folder/(k+'.npy'),v)
        del buf;gc.collect()
        for f in folder.iterdir():hashes[str(f)]=sha(f)
        print('MATERIALIZED',site,role,stats[role],flush=True)
    pd.concat(excluded).to_csv(out/'exclusions.csv',index=False)
    x=np.load(out/'data/train/future_nwp.npy',mmap_mode='r');mean=x.mean(axis=(0,1),dtype=np.float64);scale=x.std(axis=(0,1),dtype=np.float64);scale=np.where(scale>1e-6,scale,1)
    save(out/'normalizer.json',{'mean':mean.tolist(),'scale':scale.tolist(),'names':list(NWP_NAMES),'fit_split':'train'})
    hashes[str(out/'normalizer.json')]=sha(out/'normalizer.json');hashes[str(out/'exclusions.csv')]=sha(out/'exclusions.csv');hashes[str(out/'source_snapshot.json')]=sha(out/'source_snapshot.json')
    save(out/'prepared.json',{'status':'PASS','hashes':hashes,'counts':stats,'capacity_kw':CAP[site],'capacity_provenance':'Inherited fixed public benchmark DC nameplate;not new asset certification','source_power_sha':sha(SOURCE/f'{site}_strict_power.csv'),'source_queue_sha':sha(SOURCE/f'{site}_power_only_queue.csv'),'timezone_assumption':'America/New_York' if site=='9069' else 'America/Denver','receipt_site_id_label':'Legacy2107 retained;native coordinates and exact site acquisition config validated','solar_midpoint':'target ending timestamp minus450s','history_delay_seconds':900,'historical_weather_delay_hours':6,'source_gate':ga.source_gate(snap),'strict_complete_weather_intersection':True,'evaluation_exposed':True})
    register('MATERIALIZED_'+site,'KEEP：'+site+'严格天气功率共同样本完成；原生源码重放和哈希通过；'+json.dumps(stats,ensure_ascii=False)+'；0模型fit/score，评价已暴露',out/'prepared.json')
if __name__=='__main__':run(sys.argv[1])
