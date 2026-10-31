#!/usr/bin/env python3
"""Assemble all reported quantities into out/results.json."""
import os, sys, json
import numpy as np, pandas as pd
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.join(BASE,'scripts'))
from s1_09_estimate import GROUP_BY_COUNTRY, wild_cluster_boot, randomization_inference, build_panel

REG={'EU_aligned':'fr','EU_reversal':'es','EU_pretreated':'steady',
     'EU_noscheme':'noscheme','nonEU':'noneu','EEA_nonEU':'eea','EU_unconfirmed':'unconf'}
R={}

def w(series,weights): return float(np.average(series,weights=weights))

def did_core(d, ycol, keep_t, nperm=2000):
    d=d[d.t_rel.isin(keep_t)].copy()
    ok=d.groupby('unit')['t_rel'].nunique(); d=d[d.unit.isin(ok[ok==len(keep_t)].index)]
    d['post']=(d.t_rel>0).astype(int)
    d=d.dropna(subset=[ycol])
    if d.empty or d.treat.nunique()<2: return None
    y=d[ycol].to_numpy(float)
    U=pd.get_dummies(d['unit'],drop_first=True,dtype=float).to_numpy()
    T=pd.get_dummies(d['t_rel'],drop_first=True,dtype=float).to_numpy()
    inter=(d['post']*d['treat']).to_numpy(float).reshape(-1,1)
    X=np.hstack([np.ones((len(d),1)),inter,U,T])
    r=wild_cluster_boot(y,X,d['unit'].to_numpy(),j=1)
    r['randomization']=randomization_inference(d[['unit','t_rel','post',ycol]].dropna(),
        ycol,d[d.treat==1].unit.unique().tolist(),nperm=nperm)
    r['n_treat_clusters']=int(d[d.treat==1].unit.nunique())
    r['n_ctrl_clusters']=int(d[d.treat==0].unit.nunique())
    return r

def two_period(d, ycol, t0, t1, nperm=2000):
    """t0 -> t1 두 기간만 쓰는 DiD(이벤트 스터디 계수용)."""
    s=d[d.t_rel.isin([t0,t1])].copy()
    ok=s.groupby('unit')['t_rel'].nunique(); s=s[s.unit.isin(ok[ok==2].index)]
    s=s.dropna(subset=[ycol])
    if s.empty or s.treat.nunique()<2: return None
    y=s[ycol].to_numpy(float)
    U=pd.get_dummies(s['unit'],drop_first=True,dtype=float).to_numpy()
    p=(s.t_rel==t1).astype(float).to_numpy().reshape(-1,1)
    inter=(p.ravel()*s.treat.to_numpy()).reshape(-1,1)
    X=np.hstack([np.ones((len(s),1)),inter,p,U])
    return wild_cluster_boot(y,X,s['unit'].to_numpy(),j=1)

def put_did(prefix, r):
    if not r:
        for k in ('b','se','ci_lo','ci_hi','wcb_p','ri_p'): R[f'{prefix}_{k}']=None
        return
    b,se=r['coef'],r['se_cluster']
    R[f'{prefix}_b']=round(b,4); R[f'{prefix}_se']=round(se,4)
    R[f'{prefix}_ci_lo']=round(b-1.96*se,4); R[f'{prefix}_ci_hi']=round(b+1.96*se,4)
    R[f'{prefix}_wcb_p']=round(r['p_wildboot'],4)
    ri=r.get('randomization') or {}
    R[f'{prefix}_ri_p']=ri.get('p_randomization')
    R[f'{prefix}_n_treat_units']=r.get('n_treat_clusters')
    R[f'{prefix}_n_ctrl_units']=r.get('n_ctrl_clusters')

def main():
    panel=build_panel(); panel['regime']=panel.grp.map(REG)
    snap=pd.read_csv(os.path.join(BASE,'out','city_snapshot_summary.csv'))
    tr=pd.read_csv(os.path.join(BASE,'out','transitions_final.csv'))
    tr['regime']=tr.grp.map(REG)

    # ---------- 표본 ----------
    R['spec_note']=('주사양: pre=2025-12(t-2), post=2026-06(t+1). 2026-03(t-1) 스냅샷은 체계적 미완이라 '
                    '주분석에서 제외(보충에 포함). 군집=도시, 추론=와일드 클러스터 부트스트랩 + 도시 단위 무작위화.')
    R['cutoff_date']='2026-05-20'
    R['n_units']=int(panel.unit.nunique())
    R['n_countries']=int(panel.country.nunique())
    R['n_snapshots']=int(len(snap))
    R['n_listing_obs']=int(snap.n_listings.sum())
    R['n_units_eu']=int(panel[panel.grp.astype(str).str.startswith('EU')].unit.nunique())
    R['n_units_eea']=int(panel[panel.grp=='EEA_nonEU'].unit.nunique())
    R['n_units_noneu']=int(panel[panel.grp=='nonEU'].unit.nunique())
    R['snapshot_first']=str(snap.publish_date.min()); R['snapshot_last']=str(snap.publish_date.max())
    R['n_panel_listings']=int(tr[(tr.pair=='main_clean')&(tr.segment=='all')].n_t0.sum())
    R['n_panel_blank_t0']=int(tr[(tr.pair=='main_clean')&(tr.segment=='all')].n_blank_t0.sum())
    R['n_units_by_regime']={k:int(panel[panel.regime==k].unit.nunique()) for k in sorted(panel.regime.dropna().unique())}
    R['units_by_regime']={k:sorted(panel[panel.regime==k].unit.unique().tolist()) for k in sorted(panel.regime.dropna().unique())}

    # ---------- 표시율 ----------
    for rg,g in panel.groupby('regime'):
        for lbl,t in (('pre',-2),('post',1),('post2',2),('mar',-1)):
            s=g[g.t_rel==t]
            if s.empty: R[f'disp_rate_{rg}_{lbl}']=None; continue
            R[f'disp_rate_{rg}_{lbl}']=round(w(s.pct_number,s.n_listings),3)
            R[f'blank_rate_{rg}_{lbl}']=round(w(s.pct_blank,s.n_listings),3)
            R[f'exempt_rate_{rg}_{lbl}']=round(w(s.pct_exempt,s.n_listings),3)
            R[f'n_listings_{rg}_{lbl}']=int(s.n_listings.sum())
        if R.get(f'disp_rate_{rg}_pre') is not None and R.get(f'disp_rate_{rg}_post') is not None:
            R[f'disp_change_{rg}']=round(R[f'disp_rate_{rg}_post']-R[f'disp_rate_{rg}_pre'],3)
            R[f'disp_change_{rg}_to_post2']=(round(R[f'disp_rate_{rg}_post2']-R[f'disp_rate_{rg}_pre'],3)
                                             if R.get(f'disp_rate_{rg}_post2') is not None else None)
        # 다물건/1채
        for lbl,t in (('pre',-2),('post',1)):
            s=g[g.t_rel==t]
            if s.empty: continue
            R[f'disp_rate_{rg}_multi_{lbl}']=round(w(s.pct_number_multi.fillna(0),s.n_listings),3)
            R[f'disp_rate_{rg}_single_{lbl}']=round(w(s.pct_number_single.fillna(0),s.n_listings),3)

    # ---------- 주 DiD ----------
    for rg in ('fr','es','steady','unconf'):
        d=panel[panel.regime.isin([rg,'noneu'])].copy()
        d['treat']=(d.regime==rg).astype(int)
        put_did(f'did_disp_{rg}', did_core(d,'pct_number',[-2,1]))
        put_did(f'did_disp_{rg}_t2', did_core(d,'pct_number',[-2,1,2]))
    # EU 전체(확인된 국가) vs 비EU
    d=panel[panel.regime.isin(['fr','es','steady','noscheme','noneu'])].copy()
    d['treat']=(d.regime!='noneu').astype(int)
    put_did('did_disp_euconfirmed', did_core(d,'pct_number',[-2,1]))

    # ---------- 이벤트 스터디 (기준 t-2 = 2025-12) ----------
    for rg in ('fr','es','steady','noscheme','eea'):
        d=panel[panel.regime.isin([rg,'noneu'])].copy()
        d['treat']=(d.regime==rg).astype(int)
        R[f'es_disp_{rg}_m2']=0.0      # 기준
        for rel,t in (('m1',-1),('p1',1),('p2',2)):
            r=two_period(d,'pct_number',-2,t)
            R[f'es_disp_{rg}_{rel}']=round(r['coef'],4) if r else None
            R[f'es_disp_{rg}_{rel}_se']=round(r['se_cluster'],4) if r else None
            R[f'es_disp_{rg}_{rel}_p']=round(r['p_wildboot'],4) if r else None
        R[f'es_disp_{rg}_m3']=None; R[f'es_disp_{rg}_p0']=None   # 해당 스냅샷 없음

    # ---------- 전이 ----------
    for pair,suf in (('main_clean',''),('pretrend_raw','_pre'),('post_placebo','_postpl')):
        sub=tr[tr.pair==pair]
        for rg,g in sub[sub.segment=='all'].groupby('regime'):
            wt=g.n_blank_t0.to_numpy(float)
            if wt.sum()==0: continue
            R[f'trans_acquire_{rg}{suf}']=round(w(g.pct_acquire,wt),3)
            R[f'trans_exit_{rg}{suf}']=round(w(g.pct_gone,wt),3)
            R[f'trans_same_{rg}{suf}']=round(w(g.pct_same,wt),3)
            R[f'trans_exempt_{rg}{suf}']=round(w(g.pct_exempt,wt),3)
            R[f'trans_n_blank_t0_{rg}{suf}']=int(wt.sum())
        # 번호 있던 숙소가 번호를 잃은 비율
        for rg,g in sub[sub.segment=='had_number'].groupby('regime'):
            wt=g.n_blank_t0.to_numpy(float)
            if wt.sum()==0: continue
            R[f'kept_number_{rg}{suf}']=round(w(g.pct_acquire,wt),3)
            R[f'lost_to_blank_{rg}{suf}']=round(w(g.pct_same,wt),3)
            R[f'lost_to_exempt_{rg}{suf}']=round(w(g.pct_exempt,wt),3)
            R[f'hadnum_exit_{rg}{suf}']=round(w(g.pct_gone,wt),3)
            R[f'hadnum_n_t0_{rg}{suf}']=int(wt.sum())
    # 이질성 분할
    for seg in ('multi','single','entire','private'):
        for rg,g in tr[(tr.pair=='main_clean')&(tr.segment==seg)].groupby('regime'):
            wt=g.n_blank_t0.to_numpy(float)
            if wt.sum()==0: continue
            R[f'acquire_{seg}_{rg}']=round(w(g.pct_acquire,wt),3)
            R[f'exit_{seg}_{rg}']=round(w(g.pct_gone,wt),3)
            R[f'n_{seg}_{rg}']=int(wt.sum())

    # 전이 DiD(수준차, 군집=도시) — main_clean
    for rg in ('fr','es','steady'):
        s=tr[(tr.pair=='main_clean')&(tr.segment=='all')&(tr.regime.isin([rg,'noneu']))].copy()
        if s.empty or s.regime.nunique()<2: continue
        s['treat']=(s.regime==rg).astype(float)
        for ycol,nm in (('pct_acquire','did_acquire'),('pct_gone','did_exit')):
            y=s[ycol].to_numpy(float)
            X=np.hstack([np.ones((len(s),1)),s.treat.to_numpy().reshape(-1,1)])
            r=wild_cluster_boot(y,X,s['unit'].to_numpy(),j=1)
            r['n_treat_clusters']=int((s.treat==1).sum()); r['n_ctrl_clusters']=int((s.treat==0).sum())
            put_did(f'{nm}_{rg}', r)

    # ---------- 플라시보 ----------
    for rg in ('fr','es'):
        d=panel[panel.regime.isin([rg,'noneu'])].copy(); d['treat']=(d.regime==rg).astype(int)
        r=two_period(d,'pct_number',-2,-1)     # 둘 다 시행 전 (단 t-1 미완 — 주의)
        R[f'placebo_pre_{rg}_b']=round(r['coef'],4) if r else None
        R[f'placebo_pre_{rg}_p']=round(r['p_wildboot'],4) if r else None
        R[f'pretrend_p_{rg}']=R[f'placebo_pre_{rg}_p']
        r2=two_period(d,'pct_number',1,2)      # 둘 다 시행 후
        R[f'placebo_post_{rg}_b']=round(r2['coef'],4) if r2 else None
        R[f'placebo_post_{rg}_p']=round(r2['p_wildboot'],4) if r2 else None
    # 미시행국 플라시보
    d=panel[panel.regime.isin(['noscheme','noneu'])].copy(); d['treat']=(d.regime=='noscheme').astype(int)
    r=did_core(d,'pct_number',[-2,1])
    R['placebo_noscheme_b']=round(r['coef'],4) if r else None
    R['placebo_noscheme_p']=round(r['p_wildboot'],4) if r else None
    # 비EU 가짜 시행일
    rng=np.random.default_rng(7)
    pl=panel[panel.regime=='noneu'].copy()
    units=sorted(pl.unit.unique()); rng.shuffle(units)
    pl['treat']=pl.unit.isin(units[:len(units)//2]).astype(int)
    r=two_period(pl,'pct_number',-2,1)
    R['placebo_fakedate_noneu_b']=round(r['coef'],4) if r else None
    R['placebo_fakedate_noneu_p']=round(r['p_wildboot'],4) if r else None

    # ---------- 자료 품질 ----------
    piv=panel.pivot_table(index='unit',columns='t_rel',values='n_listings')
    if set([-2,-1,1])<=set(piv.columns):
        ratio=piv[-1]/((piv[-2]+piv[1])/2)
        R['mar_coverage_ratio_mean']=round(float(ratio.mean()),4)
        R['mar_coverage_ratio_median']=round(float(ratio.median()),4)
        R['mar_coverage_ratio_min']=round(float(ratio.min()),4)
        R['mar_coverage_n_units_below_070']=int((ratio<0.70).sum())
        R['mar_coverage_n_units']=int(ratio.notna().sum())
    R['listings_by_period']={f't{int(t):+d}':int(panel[panel.t_rel==t].n_listings.sum())
                             for t in sorted(panel.t_rel.unique())}

    # ---------- ML ----------
    mlp=os.path.join(BASE,'out','ml_results.json')
    if os.path.exists(mlp):
        ml=json.load(open(mlp))
        R['ml_model']=ml.get('model'); R['ml_tabpfn_version']=ml.get('tabpfn_version')
        R['ml_gpu']=ml.get('gpu'); R['ml_torch']=ml.get('torch')
        R['ml_n_total']=(ml.get('sample') or {}).get('n_total')
        for k in ('eu','noneu','es','steady'):
            c=ml.get(f'cv_{k}')
            if c:
                R[f'ml_auc_{k}']=c.get('auc_ovr_macro'); R[f'ml_balacc_{k}']=c.get('bal_acc')
                R[f'ml_acc_{k}']=c.get('accuracy'); R[f'ml_n_{k}']=c.get('n')
                R[f'ml_folds']=c.get('folds'); R[f'ml_fit_secs_{k}']=c.get('fit_secs_mean')
        pe=ml.get('perm_eu')
        if pe: R['ml_perm_top']=pe['importances'][:8]; R['ml_perm_baseline_eu']=pe['baseline_bal_acc']
        pn=ml.get('perm_noneu')
        if pn: R['ml_perm_top_noneu']=pn['importances'][:8]
        R['ml_sample_cap_probe']=ml.get('sample_cap_probe')
        R['ml_the credential file_touched']=ml.get('the credential file_touched')
        if ml.get('TOKEN_REQUIRED'): R['ml_TOKEN_REQUIRED']=True
    out=os.path.join(BASE,'out','results.json')
    json.dump(R,open(out,'w'),indent=1,ensure_ascii=False,default=str)
    print('WROTE',out,'keys=',len(R))
    for k in ('did_disp_fr_b','did_disp_fr_wcb_p','did_disp_es_b','did_disp_es_wcb_p',
              'did_disp_es_t2_b','disp_rate_es_pre','disp_rate_es_post','disp_rate_noscheme_pre',
              'trans_acquire_es','trans_exit_es','lost_to_exempt_es','placebo_noscheme_b'):
        print(f'  {k} = {R.get(k)}')
if __name__=='__main__': main()
