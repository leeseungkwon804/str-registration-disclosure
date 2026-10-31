#!/usr/bin/env python3
"""Add the named quantities reported in the article to out/results.json (rounding: shares and pp 1 decimal, coefficients 3, p values 3)."""
import os, sys, json
import numpy as np, pandas as pd
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.join(BASE,'scripts'))
from s1_09_estimate import wild_cluster_boot, randomization_inference, build_panel
RJ=os.path.join(BASE,'out','results.json')
R=json.load(open(RJ))
REGS=['fr','es','steady','noscheme','unver','eea','noneu']
GRP2REG={'EU_aligned':'fr','EU_reversal':'es','EU_pretreated':'steady','EU_noscheme':'noscheme',
         'nonEU':'noneu','EEA_nonEU':'eea','EU_unconfirmed':'unver'}
r1=lambda v: None if v is None else round(float(v),1)
r3=lambda v: None if v is None else round(float(v),3)
def ci(lo,hi): return None if lo is None or hi is None else f'{lo:.1f} to {hi:.1f}'

panel=build_panel(); panel['regime']=panel.grp.map(GRP2REG)
tr=pd.read_csv(os.path.join(BASE,'out','transitions_v2.csv'))
ml=json.load(open(os.path.join(BASE,'out','ml_results.json')))
W=lambda s,w: float(np.average(s,weights=w)) if len(s) and np.sum(w)>0 else None

# ---------------- 표본 ----------------
R['n_units_cov']=R.get('mar_coverage_n_units'); R['cov_ratio_mean']=round(R.get('mar_coverage_ratio_mean'),3)
R['n_units_cov_low']=R.get('mar_coverage_n_units_below_070')
R['priv_units']=12; R['priv_values']=3000
for rg in REGS:
    g=panel[panel.regime==rg]
    R[f'n_units_{rg}']=int(g.unit.nunique())
    R[f'n_list_{rg}']=int(g[g.t_rel==-2].n_listings.sum())
    for lbl,t in (('pre',-2),('t1',1),('t2',2),('march',-1)):
        s=g[g.t_rel==t]
        if s.empty: continue
        R[f'disp_rate_{rg}_{lbl}']=r1(W(s.pct_number,s.n_listings))
        R[f'blank_rate_{rg}_{lbl}']=r1(W(s.pct_blank,s.n_listings))
        R[f'exempt_{rg}_{lbl}']=r1(W(s.pct_exempt,s.n_listings))
        R[f'pending_{rg}_{lbl}']=r1(W(s.pct_pending,s.n_listings))
    if R.get(f'disp_rate_{rg}_pre') is not None:
        for lbl in ('t1','t2'):
            if R.get(f'disp_rate_{rg}_{lbl}') is not None:
                R[f'disp_chg_{rg}' if lbl=='t1' else f'disp_chg_{rg}_t2']=r1(
                    R[f'disp_rate_{rg}_{lbl}']-R[f'disp_rate_{rg}_pre'])
        for nm,a in (('blank_chg','blank_rate'),('exempt_chg','exempt'),('pending_chg','pending')):
            if R.get(f'{a}_{rg}_t2') is not None and R.get(f'{a}_{rg}_pre') is not None:
                R[f'{nm}_{rg}']=r1(R[f'{a}_{rg}_t2']-R[f'{a}_{rg}_pre'])
R['disp_gap_scheme']=r1((R.get('disp_rate_steady_pre') or 0)-(R.get('disp_rate_noscheme_pre') or 0))
sn=pd.read_csv(os.path.join(BASE,'out','city_snapshot_summary.csv'))
R['text_other_max']=r1(sn.pct_text_other.max())
R['cov_disp_gap_steady']=r1((R.get('disp_rate_steady_march') or 0)-(R.get('disp_rate_steady_pre') or 0))

# ---------------- DiD ----------------
def did(treat,ctrl,periods):
    d=panel[panel.regime.isin(list(treat)+list(ctrl))].copy()
    d=d[d.t_rel.isin(periods)]
    ok=d.groupby('unit')['t_rel'].nunique(); d=d[d.unit.isin(ok[ok==len(periods)].index)]
    d['treat']=d.regime.isin(treat).astype(int); d['post']=(d.t_rel>0).astype(int)
    if periods==[-1,1]: d['post']=(d.t_rel>0).astype(int)
    d=d.dropna(subset=['pct_number'])
    if d.empty or d.treat.nunique()<2: return None
    y=d.pct_number.to_numpy(float)
    U=pd.get_dummies(d['unit'],drop_first=True,dtype=float).to_numpy()
    T=pd.get_dummies(d['t_rel'],drop_first=True,dtype=float).to_numpy()
    inter=(d['post']*d['treat']).to_numpy(float).reshape(-1,1)
    X=np.hstack([np.ones((len(d),1)),inter,U,T])
    out=wild_cluster_boot(y,X,d['unit'].to_numpy(),j=1)
    out['randomization']=randomization_inference(d[['unit','t_rel','post','pct_number']].dropna(),
        'pct_number',d[d.treat==1].unit.unique().tolist(),nperm=2000)
    out['n_obs']=int(len(d)); out['n_units']=int(d.unit.nunique())
    return out
def emit(pfx,o):
    if not o:
        for k in ('b','ci_lo','ci_hi','ci','wcb_p','ri_p'): R[f'{pfx}_{k}']=None
        return
    b,se=o['coef'],o['se_cluster']; lo,hi=b-1.96*se,b+1.96*se
    R[f'{pfx}_b']=r1(b); R[f'{pfx}_se']=r3(se)
    R[f'{pfx}_ci_lo']=r1(lo); R[f'{pfx}_ci_hi']=r1(hi); R[f'{pfx}_ci']=ci(lo,hi)
    R[f'{pfx}_wcb_p']=r3(o['p_wildboot'])
    R[f'{pfx}_ri_p']=r3((o.get('randomization') or {}).get('p_randomization'))

for rg in ('fr','es','steady','unver'):
    emit(f'did_disp_{rg}_t1', did([rg],['noneu'],[-2,1]))
    emit(f'did_disp_{rg}_t2', did([rg],['noneu'],[-2,2]))
    o=did([rg],['noneu'],[-2,1]); R[f'n_obs_{rg}']=o['n_obs'] if o else None
# 2026-03 기준판(자료품질 비교용)
o=did(['es'],['noneu'],[-1,1]); R['did_disp_es_t1_b_march']=r1(o['coef']) if o else None
# 플라시보
emit('placebo_noscheme', did(['noscheme'],['noneu'],[-2,1]))
o=did(['noscheme'],['noneu'],[-2,1]); R['n_obs_noscheme']=o['n_obs'] if o else None
emit('placebo_eea', did(['eea'],['noneu'],[-2,1]))
o=did(['eea'],['noneu'],[-2,1]); R['n_obs_eea']=o['n_obs'] if o else None
rng=np.random.default_rng(7)
pl=panel[panel.regime=='noneu'].copy(); us=sorted(pl.unit.unique()); rng.shuffle(us)
pl['regime']=np.where(pl.unit.isin(us[:len(us)//2]),'FAKE','noneu')
sav=panel.copy()
panel=pl; emit('placebo_fake', did(['FAKE'],['noneu'],[-2,1]))
o=did(['FAKE'],['noneu'],[-2,1]); R['n_obs_noneu']=o['n_obs'] if o else None
panel=sav
# 로버스트: 미확인국 포함/제외
o=did(['es'],['noneu','unver'],[-2,1]); R['robust_unver_es_b']=r1(o['coef']) if o else None
o=did(['es','unver'],['noneu'],[-2,1]); R['robust_unver_es_b_alt']=r1(o['coef']) if o else None

# ---------------- 전이 ----------------
def agg(pair,seg,origin,regime,col):
    g=tr[(tr.pair==pair)&(tr.segment==seg)&(tr.origin==origin)&(tr.regime==regime)]
    if g.empty or g.n0.sum()==0: return None
    return W(g['pct_'+col],g.n0.to_numpy(float))
def n0(pair,seg,origin,regime):
    g=tr[(tr.pair==pair)&(tr.segment==seg)&(tr.origin==origin)&(tr.regime==regime)]
    return int(g.n0.sum()) if not g.empty else None

for rg in REGS:
    for col,nm in (('number','num'),('exempt','exempt'),('blank','blank'),('gone','exit')):
        R[f'trans_{rg}_num_{nm}']=r1(agg('main','all','number',rg,col))
    R[f'trans_{rg}_none_acq']=r1(agg('main','all','blank',rg,'number'))
    R[f'trans_{rg}_none_exit']=r1(agg('main','all','blank',rg,'gone'))
    R[f'trans_{rg}_none_same']=r1(agg('main','all','blank',rg,'blank'))
    for seg in ('single','multi'):
        R[f'n_foll_{rg}_{seg}']=(n0('main',seg,'number',rg) or 0)+(n0('main',seg,'blank',rg) or 0)
        for col,nm in (('number','num'),('exempt','exempt'),('blank','blank'),('gone','exit')):
            R[f'tr_{rg}_{seg}_{nm}']=r1(agg('main',seg,'number',rg,col))
        R[f'tr_{rg}_{seg}_acq']=r1(agg('main',seg,'blank',rg,'number'))
# 「번호를 잃었다」 = 살아남은 숙소 중 번호를 더는 표시하지 않는 비율
def lose(pair,seg,regime):
    g=tr[(tr.pair==pair)&(tr.segment==seg)&(tr.origin=='number')&(tr.regime==regime)]
    if g.empty: return None
    surv=(g.n0-g.n_gone).sum()
    if surv<=0: return None
    return 100*float((g.n_exempt+g.n_blank+g.n_other).sum())/float(surv)
for rg in REGS: R[f'lose_{rg}']=r1(lose('main','all',rg))
for seg,key in (('single','lose_single_es'),('multi','lose_multi_es'),
                ('entire','lose_entire_es'),('private','lose_private_es'),
                ('price_q1','lose_q1_es'),('price_q4','lose_q4_es'),
                ('rev_q1','lose_rev_q1_es'),('rev_q4','lose_rev_q4_es')):
    R[key]=r1(lose('main',seg,'es'))
if R.get('lose_single_es') is not None and R.get('lose_multi_es') is not None:
    R['lose_gap_es']=r1(R['lose_single_es']-R['lose_multi_es'])
# lose 격차 검정(도시 군집)
g=tr[(tr.pair=='main')&(tr.origin=='number')&(tr.regime=='es')&(tr.segment.isin(['single','multi']))].copy()
if not g.empty:
    g['lose_pct']=100*(g.n_exempt+g.n_blank+g.n_other)/(g.n0-g.n_gone).replace(0,np.nan)
    g=g.dropna(subset=['lose_pct'])
    g['single']=(g.segment=='single').astype(float)
    y=g.lose_pct.to_numpy(float)
    U=pd.get_dummies(g['unit'],drop_first=True,dtype=float).to_numpy()
    X=np.hstack([np.ones((len(g),1)),g.single.to_numpy().reshape(-1,1),U])
    o=wild_cluster_boot(y,X,g['unit'].to_numpy(),j=1)
    R['lose_gap_es_wcb_p']=r3(o['p_wildboot']); R['lose_gap_es_b']=r1(o['coef'])
# 전이 DiD (도시 수준 수준차)
def trdid(ycol,regime,pair='main',origin='blank',seg='all'):
    s=tr[(tr.pair==pair)&(tr.segment==seg)&(tr.origin==origin)&(tr.regime.isin([regime,'noneu']))].copy()
    if s.empty or s.regime.nunique()<2: return None
    s['treat']=(s.regime==regime).astype(float)
    y=s[ycol].to_numpy(float)
    X=np.hstack([np.ones((len(s),1)),s.treat.to_numpy().reshape(-1,1)])
    return wild_cluster_boot(y,X,s['unit'].to_numpy(),j=1)
emit('did_exit_es', trdid('pct_gone','es'))
emit('did_acquire_es', trdid('pct_number','es'))
emit('did_acquire_fr', trdid('pct_number','fr'))
emit('did_exit_fr', trdid('pct_gone','fr'))
# 「번호 상실」 DiD (origin=number)
s=tr[(tr.pair=='main')&(tr.segment=='all')&(tr.origin=='number')&(tr.regime.isin(['es','noneu']))].copy()
if not s.empty:
    s['lose_pct']=100*(s.n_exempt+s.n_blank+s.n_other)/(s.n0-s.n_gone).replace(0,np.nan)
    s=s.dropna(subset=['lose_pct']); s['treat']=(s.regime=='es').astype(float)
    y=s.lose_pct.to_numpy(float)
    X=np.hstack([np.ones((len(s),1)),s.treat.to_numpy().reshape(-1,1)])
    emit('did_lose_es', wild_cluster_boot(y,X,s['unit'].to_numpy(),j=1))
# 사후 플라시보
R['post_placebo_lose_b']=r1(lose('post_placebo','all','es'))
R['post_placebo_exit_b']=r1(agg('post_placebo','all','blank','es','gone'))
# 사전구간 퇴출률(쓸 수 없음을 보이는 숫자)
R['bad_exit_pre']=r1(agg('pretrend_raw','all','blank','steady','gone'))
R['good_exit_main']=r1(agg('main','all','blank','steady','gone'))

# ---------------- ML ----------------
R['ml_auc_es']=r3((ml.get('cv_es') or {}).get('auc_ovr_macro'))
R['ml_balacc_es']=r3((ml.get('cv_es') or {}).get('bal_acc'))
R['ml_auc_noneu']=r3((ml.get('cv_noneu') or {}).get('auc_ovr_macro'))
R['ml_auc_steady']=r3((ml.get('cv_steady') or {}).get('auc_ovr_macro'))
R['ml_auc_eu']=r3((ml.get('cv_eu') or {}).get('auc_ovr_macro'))
R['ml_balacc_eu']=r3((ml.get('cv_eu') or {}).get('bal_acc'))
json.dump(R,open(RJ,'w'),indent=1,ensure_ascii=False,default=str)
print('WROTE',RJ,'keys=',len(R))
for k in ('did_disp_es_t1_b','did_disp_es_t1_ci','did_disp_es_t2_b','did_disp_fr_t1_b',
          'trans_es_num_exempt','trans_steady_num_exempt','lose_single_es','lose_multi_es',
          'lose_gap_es','lose_gap_es_wcb_p','bad_exit_pre','good_exit_main',
          'did_disp_es_t1_b_march','placebo_eea_b','placebo_fake_b','disp_gap_scheme'):
    print(f'  {k} = {R.get(k)}')
