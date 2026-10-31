#!/usr/bin/env python3
"""Estimates against European markets where the display duty remained in force."""
import os, sys, json
import numpy as np, pandas as pd
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.join(BASE,'scripts'))
from s1_09_estimate import build_panel, wild_cluster_boot, randomization_inference
RJ=os.path.join(BASE,'out','results.json'); R=json.load(open(RJ))
REG={'EU_aligned':'fr','EU_reversal':'es','EU_pretreated':'steady','EU_noscheme':'noscheme',
     'nonEU':'noneu','EEA_nonEU':'eea','EU_unconfirmed':'unver'}
panel=build_panel(); panel['regime']=panel.grp.map(REG)
tr=pd.read_csv(os.path.join(BASE,'out','transitions_v2.csv'))
r1=lambda v: None if v is None else round(float(v),1)

def fit(d,periods,nperm=2000):
    d=d[d.t_rel.isin(periods)].copy()
    ok=d.groupby('unit')['t_rel'].nunique(); d=d[d.unit.isin(ok[ok==len(periods)].index)]
    d['post']=(d.t_rel>0).astype(int)
    d=d.dropna(subset=['pct_number'])
    if d.empty or d.treat.nunique()<2: return None
    y=d.pct_number.to_numpy(float)
    U=pd.get_dummies(d['unit'],drop_first=True,dtype=float).to_numpy()
    T=pd.get_dummies(d['t_rel'],drop_first=True,dtype=float).to_numpy()
    inter=(d['post']*d['treat']).to_numpy(float).reshape(-1,1)
    X=np.hstack([np.ones((len(d),1)),inter,U,T])
    o=wild_cluster_boot(y,X,d['unit'].to_numpy(),j=1)
    o['randomization']=randomization_inference(d[['unit','t_rel','post','pct_number']].dropna(),
        'pct_number',d[d.treat==1].unit.unique().tolist(),nperm=nperm)
    o['n_treat_units']=int(d[d.treat==1].unit.nunique()); o['n_ctrl_units']=int(d[d.treat==0].unit.nunique())
    o['n_obs']=int(len(d))
    return o

def emit(pfx,o):
    if not o:
        for k in ('b','se','ci','ci_lo','ci_hi','wcb_p','ri_p'): R[f'{pfx}_{k}']=None
        return
    b,se=o['coef'],o['se_cluster']
    R[f'{pfx}_b']=r1(b); R[f'{pfx}_se']=round(se,3)
    R[f'{pfx}_ci_lo']=r1(b-1.96*se); R[f'{pfx}_ci_hi']=r1(b+1.96*se)
    R[f'{pfx}_ci']=f'{b-1.96*se:.1f} to {b+1.96*se:.1f}'
    R[f'{pfx}_wcb_p']=round(o['p_wildboot'],3)
    _ri=(o.get('randomization') or {}).get('p_randomization')
    R[f'{pfx}_ri_p']=None if _ri is None else round(_ri,3)
    R[f'{pfx}_n_treat_units']=o.get('n_treat_units'); R[f'{pfx}_n_ctrl_units']=o.get('n_ctrl_units')
    R[f'{pfx}_n_obs']=o.get('n_obs')

CTRL={'vsteady':['steady','fr'], 'vsteady_nofr':['steady']}
for tag,ctrl in CTRL.items():
    d0=panel[panel.regime.isin(['es']+ctrl)].copy(); d0['treat']=(d0.regime=='es').astype(int)
    emit(f'did_disp_es_{tag}_t1', fit(d0,[-2,1]))
    emit(f'did_disp_es_{tag}_t2', fit(d0,[-2,2]))
    emit(f'did_disp_es_{tag}_pooled', fit(d0,[-2,1,2]))
    print(tag,'t1',R.get(f'did_disp_es_{tag}_t1_b'),R.get(f'did_disp_es_{tag}_t1_wcb_p'),
          '| t2',R.get(f'did_disp_es_{tag}_t2_b'),'| pooled',R.get(f'did_disp_es_{tag}_pooled_b'),flush=True)

# 프랑스 사후구간 플라시보를 EU 내부 대조로 재검
d=panel[panel.regime.isin(['fr','steady'])].copy(); d['treat']=(d.regime=='fr').astype(int)
s=d[d.t_rel.isin([1,2])].copy()
ok=s.groupby('unit')['t_rel'].nunique(); s=s[s.unit.isin(ok[ok==2].index)]
s=s.dropna(subset=['pct_number'])
if not s.empty and s.treat.nunique()>1:
    y=s.pct_number.to_numpy(float)
    U=pd.get_dummies(s['unit'],drop_first=True,dtype=float).to_numpy()
    p=(s.t_rel==2).astype(float).to_numpy().reshape(-1,1)
    inter=(p.ravel()*s.treat.to_numpy()).reshape(-1,1)
    X=np.hstack([np.ones((len(s),1)),inter,p,U])
    emit('placebo_post_fr_vsteady', wild_cluster_boot(y,X,s['unit'].to_numpy(),j=1))
    print('placebo_post_fr_vsteady',R.get('placebo_post_fr_vsteady_b'),R.get('placebo_post_fr_vsteady_wcb_p'),flush=True)

# 전이·면제 전환을 EU 내부 대조로 (도시 수준 수준차)
def trfit(ycol,origin,ctrl_regimes,seg='all',pair='main'):
    s=tr[(tr.pair==pair)&(tr.segment==seg)&(tr.origin==origin)&
         (tr.regime.isin(['es']+ctrl_regimes))].copy()
    if s.empty or s.regime.nunique()<2: return None
    s['treat']=(s.regime=='es').astype(float)
    if ycol=='lose':
        s['y']=100*(s.n_exempt+s.n_blank+s.n_other)/(s.n0-s.n_gone).replace(0,np.nan)
        s=s.dropna(subset=['y'])
    else:
        s['y']=s[ycol]
    y=s['y'].to_numpy(float)
    X=np.hstack([np.ones((len(s),1)),s.treat.to_numpy().reshape(-1,1)])
    o=wild_cluster_boot(y,X,s['unit'].to_numpy(),j=1)
    o['n_treat_units']=int((s.treat==1).sum()); o['n_ctrl_units']=int((s.treat==0).sum()); o['n_obs']=len(s)
    return o
for tag,ctrl in CTRL.items():
    emit(f'did_exempt_es_{tag}',  trfit('pct_exempt','number',ctrl))
    emit(f'did_lose_es_{tag}',    trfit('lose','number',ctrl))
    emit(f'did_acquire_es_{tag}', trfit('pct_number','blank',ctrl))
    emit(f'did_exit_es_{tag}',    trfit('pct_gone','blank',ctrl))
    print(tag,'exempt',R.get(f'did_exempt_es_{tag}_b'),R.get(f'did_exempt_es_{tag}_wcb_p'),
          '| lose',R.get(f'did_lose_es_{tag}_b'),'| acq',R.get(f'did_acquire_es_{tag}_b'),
          '| exit',R.get(f'did_exit_es_{tag}_b'),R.get(f'did_exit_es_{tag}_wcb_p'),flush=True)

R['within_eu_control_note']=('대조군 = 의무가 그대로 유지된 EU 시장. vsteady = IT GR PT BE NL + 프랑스, '
 'vsteady_nofr = IT GR PT BE NL. EU-비유럽 잔여추세를 제거하기 위한 강건성이며, 같은 Regulation 아래 '
 '같은 대륙의 시장끼리 비교한다.')
json.dump(R,open(RJ,'w'),indent=1,ensure_ascii=False,default=str)
print('WROTE',RJ,'keys=',len(R))
