import numpy as np
NAMES=['power_clear_sky_index','power_clear_sky_index_valid','solar_scaled_persistence','solar_scaled_persistence_valid']
def build(shared,horizon,shared_names,horizon_names):
    p=shared[:,shared_names.index('past_power_t_minus_15m')].astype(float)
    g=shared[:,shared_names.index('latest_past_clear_sky_ghi_haurwitz_per_1000')].astype(float)
    valid=(shared[:,shared_names.index('past_power_t_minus_15m_valid')]>0)&np.isfinite(p)&np.isfinite(g)&(g>=.05)
    ratio=np.full(len(p),np.nan);np.divide(p,g,out=ratio,where=valid)
    future=horizon[:,:,horizon_names.index('future_clear_sky_ghi_haurwitz_per_1000')].astype(float)
    good=valid[:,None]&np.isfinite(future)&(future>=0)
    scaled=np.where(good,ratio[:,None]*future,np.nan)
    out=np.stack([np.broadcast_to(ratio[:,None],future.shape),np.broadcast_to(valid[:,None],future.shape),scaled,good],axis=-1).astype(np.float32)
    assert not np.isinf(out).any()
    return out

def test(shared_names,horizon_names):
    s=np.zeros((4,len(shared_names)),dtype=np.float32);h=np.zeros((4,16,len(horizon_names)),dtype=np.float32)
    s[:,shared_names.index('past_power_t_minus_15m')]=.2
    s[:,shared_names.index('past_power_t_minus_15m_valid')]=[1,1,1,0]
    s[:,shared_names.index('latest_past_clear_sky_ghi_haurwitz_per_1000')]=[0,.049,.5,.5]
    h[:,:,horizon_names.index('future_clear_sky_ghi_haurwitz_per_1000')]=.8
    o=build(s,h,shared_names,horizon_names)
    assert np.isnan(o[[0,1,3],:,0]).all() and (o[[0,1,3],:,1]==0).all()
    np.testing.assert_allclose(o[2,:,0],.4);np.testing.assert_allclose(o[2,:,2],.32)
    return True
