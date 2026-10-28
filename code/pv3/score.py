from .common import *
from scipy.optimize import minimize_scalar
def metric(y,p,m):
    count=m.sum(0);assert (count>0).all();e=p-y
    return {'nmae_pct':float(np.mean(np.sum(np.abs(e)*m,0)/count)*100),'nrmse_pct':float(np.sqrt(np.mean(np.sum(e*e*m,0)/count))*100),'bias_pct':float(np.mean(np.sum(e*m,0)/count)*100),'per_horizon_nmae_pct':(np.sum(np.abs(e)*m,0)/count*100).tolist(),'points':int(count.sum())}
def fit(y,a,b,m,global_weight=False):
    def one(h):
        use=m if h is None else m[:,h];aa=a if h is None else a[:,h];bb=b if h is None else b[:,h];yy=y if h is None else y[:,h]
        objective=lambda w:metric(yy,w*aa+(1-w)*bb,use)['nmae_pct'] if h is None else float(np.abs((w*aa+(1-w)*bb-yy)[use]).mean())
        opt=minimize_scalar(objective,bounds=(0,1),method='bounded',options={'xatol':1e-10})
        return min((0.,float(opt.x),1.),key=objective)
    return np.full(16,one(None)) if global_weight else np.array([one(h) for h in range(16)])
def main(sites=('9069','9068')):
    freeze_check();results={};rows=[];contrasts={};cost={};weight_manifest={}
    for site in sites:
        out=R/site;seal=json.loads((out/'predictions_frozen.json').read_text());assert all(sha(p)==h for p,h in {**seal['hashes'],**seal['model_hashes']}.items())
        y={r:np.load(out/'data'/r/'target.npy') for r in ('inner','outer','evaluation')};mask={r:np.load(out/'data'/r/'daylight.npy') for r in y}
        load=lambda r,name:np.load(out/'predictions'/f'{r}_{name}.npy')
        issue=pd.read_csv(out/'data/evaluation/keys.csv');dates=pd.to_datetime(issue.issue_local_assumed);days=dates.dt.normalize().astype('int64').to_numpy()//86400000000000;group=(days-days.min())//7;nb=int(group.max())+1
        boot=np.random.default_rng(20261008).integers(nb,size=(1000,nb));evaluation={};bestpairs={}
        for seed in (2021,2022,2023):
            raw={r:{'tree':load(r,'tree'),'nhitsx':load(r,f'nhitsx_{seed}'),'timemixer':load(r,f'timemixer_{seed}')} for r in y}
            alpha=fit(y['inner'],raw['inner']['nhitsx'],raw['inner']['timemixer'],mask['inner']);alpha_g=fit(y['inner'],raw['inner']['nhitsx'],raw['inner']['timemixer'],mask['inner'],True)
            neural={r:alpha*raw[r]['nhitsx']+(1-alpha)*raw[r]['timemixer'] for r in y};ng={r:alpha_g*raw[r]['nhitsx']+(1-alpha_g)*raw[r]['timemixer'] for r in y}
            beta=fit(y['outer'],raw['outer']['tree'],neural['outer'],mask['outer']);beta_g=fit(y['outer'],raw['outer']['tree'],ng['outer'],mask['outer'],True)
            pair_weights={};selection={};pred=raw['evaluation'].copy()
            for a,b in (('tree','nhitsx'),('tree','timemixer'),('nhitsx','timemixer')):
                name=a+'_'+b;w=fit(y['outer'],raw['outer'][a],raw['outer'][b],mask['outer']);pair_weights[name]=w.tolist();pred[name]=w*raw['evaluation'][a]+(1-w)*raw['evaluation'][b];selection[name]=metric(y['outer'],w*raw['outer'][a]+(1-w)*raw['outer'][b],mask['outer'])['nmae_pct']
            bestpairs[seed]=min(selection,key=selection.get)
            pred['static3']=beta*raw['evaluation']['tree']+(1-beta)*neural['evaluation'];pred['global3']=beta_g*raw['evaluation']['tree']+(1-beta_g)*ng['evaluation'];pred['equal3']=sum(raw['evaluation'].values())/3
            weight_manifest[f'{site}_{seed}']={'alpha':alpha.tolist(),'beta':beta.tolist(),'global_alpha':float(alpha_g[0]),'global_beta':float(beta_g[0]),'pairs':pair_weights,'outer_selected_pair':bestpairs[seed],'outer_pair_scores':selection}
            evaluation[seed]=pred;results[f'{site}_{seed}']={}
            for name,p in pred.items():
                assert np.isfinite(p).all()
                for mode,m in [('daylight',mask['evaluation']),('all',np.ones_like(mask['evaluation']))]:
                    v=metric(y['evaluation'],p,m);results[f'{site}_{seed}'][name+'_'+mode]=v;rows.append({'site':site,'seed':seed,'model':name,'mode':mode,'group':'all','horizon':'all16',**{k:v[k] for k in ('nmae_pct','nrmse_pct','bias_pct','points')}})
                    for h,mae in enumerate(v['per_horizon_nmae_pct']):rows.append({'site':site,'seed':seed,'model':name,'mode':mode,'group':'all','horizon':(h+1)*15,'nmae_pct':mae,'nrmse_pct':np.nan,'bias_pct':np.nan,'points':int(m[:,h].sum())})
                for month in sorted(dates.dt.strftime('%Y-%m').unique()):
                    use=dates.dt.strftime('%Y-%m').eq(month).to_numpy();v=metric(y['evaluation'][use],p[use],mask['evaluation'][use]);rows.append({'site':site,'seed':seed,'model':name,'mode':'daylight','group':month,'horizon':'all16',**{k:v[k] for k in ('nmae_pct','nrmse_pct','bias_pct','points')}})
        for control in ('tree','best_pair'):
            deltas=[]
            for seed in (2021,2022,2023):
                p=evaluation[seed];c='tree' if control=='tree' else bestpairs[seed];delta=(abs(p['static3']-y['evaluation'])-abs(p[c]-y['evaluation']))*mask['evaluation'];deltas.append(delta)
                numer=np.stack([delta[group==g].sum(0) for g in range(nb)]);denom=np.stack([mask['evaluation'][group==g].sum(0) for g in range(nb)]);s=numer[boot].sum(1);n=denom[boot].sum(1);good=(n>0).all(1);samples=(s[good]/n[good]).mean(1)*100
                contrasts[f'{site}_{seed}_static3_minus_{control}']={'delta_pp':float((delta.sum(0)/mask['evaluation'].sum(0)).mean()*100),'ci95_pp':np.quantile(samples,[.025,.975]).tolist(),'valid_draws':int(good.sum()),'calendar_blocks':nb,'control':c}
            delta=np.mean(deltas,axis=0);numer=np.stack([delta[group==g].sum(0) for g in range(nb)]);s=numer[boot].sum(1);samples=(s[good]/n[good]).mean(1)*100
            contrasts[f'{site}_seedmean_static3_minus_{control}']={'delta_pp':float((delta.sum(0)/mask['evaluation'].sum(0)).mean()*100),'ci95_pp':np.quantile(samples,[.025,.975]).tolist(),'valid_draws':int(good.sum())}
        cost[site]={'tree_fit_seconds':sum(q['seconds'] for q in json.loads((out/'tree_progress.json').read_text())),'neural_fit_seconds':{f'{name}_{seed}':__import__('torch').load(out/'models'/f'{name}_{seed}.pt',map_location='cpu',weights_only=False)['training_seconds'] for name in ('nhitsx','timemixer') for seed in (2021,2022,2023)}}
    save(R/'fusion_weights.json',weight_manifest);save(R/'results.json',{'scores':results,'contrasts':contrasts,'cost':cost,'evaluation_exposed':True,'metric':'macro-horizon capacity-normalized error;normalized target divided by capacity exactly once','nrmse':'sqrt(mean per-horizon normalized MSE)','seed_dispersion':'sample SD of per-seed scores,not prediction averaging'})
    pd.DataFrame(rows).to_csv(R/'metrics.csv',index=False)
    register('SCORED','KEEP/INVESTIGATE：两站3seed同键全部单体/双体/等权/全局/逐时距融合完成评分；保留全部方向；见结果，非独立确认',R/'results.json')
