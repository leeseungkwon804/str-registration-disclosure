#!/usr/bin/env python3
"""Main estimates: event study and difference-in-differences against the comparison markets, with wild cluster bootstrap and randomization inference."""
import os, sys, json
import numpy as np, pandas as pd
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RNG = np.random.default_rng(20261030)
NBOOT = 999

# S1-7 1차 출처로 확인된 국가별 시행 상태
GROUP_BY_COUNTRY = {
 'France':'EU_aligned',                                     # 2026-05 전국 확대 (정렬)
 'Italy':'EU_pretreated','Greece':'EU_pretreated','Portugal':'EU_pretreated',
 'Belgium':'EU_pretreated','The Netherlands':'EU_pretreated',
 'Hungary':'EU_pretreated',   # 2026-10-30 NTAK 공식 포털로 2020-10-01 시행 확인
 'Spain':'EU_reversal',                                     # 2025-07 발효 → 2026-05/06 대법원 무효
 'Austria':'EU_noscheme','Ireland':'EU_noscheme',           # 제도 없음
 'Germany':'EU_unconfirmed','Czech Republic':'EU_unconfirmed','Denmark':'EU_unconfirmed',
 'Latvia':'EU_unconfirmed','Malta':'EU_unconfirmed',
 'Sweden':'EU_unconfirmed',
 'Norway':'EEA_nonEU','Switzerland':'EEA_nonEU',
 'United Kingdom':'nonEU','United States':'nonEU','Canada':'nonEU',
 'Australia':'nonEU','New Zealand':'nonEU',
}
CONFIRMED = {'EU_aligned','EU_pretreated','EU_reversal','EU_noscheme','nonEU'}

def wild_cluster_boot(y, X, clusters, j, nboot=NBOOT, rng=RNG):
    """계수 j 에 대한 와일드 클러스터 부트스트랩 p값(귀무 β_j=0 부과) + 신뢰구간용 분포."""
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ (X.T @ y)
    r = y - X @ b
    uc = np.unique(clusters)
    # 군집 강건 SE
    meat = np.zeros((X.shape[1], X.shape[1]))
    for c in uc:
        m = clusters == c
        Xc, rc = X[m], r[m]
        s = Xc.T @ rc
        meat += np.outer(s, s)
    G, K = len(uc), X.shape[1]
    adj = (G/(G-1)) * ((len(y)-1)/(len(y)-K)) if G > 1 else 1.0
    V = XtXi @ meat @ XtXi * adj
    se = float(np.sqrt(max(V[j, j], 0)))
    t = float(b[j]/se) if se > 0 else np.nan
    # 귀무 부과: β_j=0 인 제약모형으로 잔차 생성
    keep = [k for k in range(K) if k != j]
    Xr = X[:, keep]
    br = np.linalg.pinv(Xr.T @ Xr) @ (Xr.T @ y)
    rr = y - Xr @ br
    fit_r = Xr @ br
    ts = []
    for _ in range(nboot):
        w = rng.choice([-1.0, 1.0], size=len(uc))
        wmap = dict(zip(uc, w))
        wv = np.array([wmap[c] for c in clusters])
        yb = fit_r + rr*wv
        bb = XtXi @ (X.T @ yb)
        rb = yb - X @ bb
        meatb = np.zeros((K, K))
        for c in uc:
            m = clusters == c
            s = X[m].T @ rb[m]
            meatb += np.outer(s, s)
        Vb = XtXi @ meatb @ XtXi * adj
        seb = np.sqrt(max(Vb[j, j], 0))
        ts.append(bb[j]/seb if seb > 0 else np.nan)
    ts = np.array([x for x in ts if np.isfinite(x)])
    p = float((np.abs(ts) >= abs(t)).mean()) if len(ts) and np.isfinite(t) else np.nan
    return dict(coef=float(b[j]), se_cluster=se, t=t, p_wildboot=p,
                n_obs=int(len(y)), n_clusters=int(G), n_boot_used=int(len(ts)))

def randomization_inference(d, ycol, treat_units, nperm=2000, rng=RNG):
    """도시 단위 무작위화 추론: 처치 배정을 도시 수준에서 재배정해 DiD 계수 분포를 만든다.
    처치 도시가 적으므로 이것을 주 추론의 하나로 쓴다."""
    units = sorted(d['unit'].unique())
    n_t = len(set(treat_units) & set(units))
    if n_t == 0 or n_t == len(units): return None
    def didcoef(tset):
        sub = d.copy()
        sub['tr'] = sub['unit'].isin(tset).astype(float)
        y = sub[ycol].to_numpy(float)
        U = pd.get_dummies(sub['unit'], drop_first=True, dtype=float).to_numpy()
        T = pd.get_dummies(sub['t_rel'], drop_first=True, dtype=float).to_numpy()
        inter = (sub['post']*sub['tr']).to_numpy(float).reshape(-1,1)
        X = np.hstack([np.ones((len(sub),1)), inter, U, T])
        b = np.linalg.pinv(X.T @ X) @ (X.T @ y)
        return float(b[1])
    obs = didcoef(set(treat_units))
    draws = []
    for _ in range(nperm):
        perm = set(rng.choice(units, size=n_t, replace=False).tolist())
        draws.append(didcoef(perm))
    draws = np.array(draws)
    return dict(coef_obs=round(obs,4), n_perm=int(nperm),
                p_randomization=round(float((np.abs(draws) >= abs(obs)).mean()),4),
                perm_mean=round(float(draws.mean()),4),
                perm_sd=round(float(draws.std(ddof=1)),4),
                perm_q025=round(float(np.quantile(draws,0.025)),4),
                perm_q975=round(float(np.quantile(draws,0.975)),4),
                n_treat_units=int(n_t), n_units=int(len(units)))

def build_panel():
    s = pd.read_csv(os.path.join(BASE,'out','city_snapshot_summary.csv'))
    s['grp'] = s['country'].map(GROUP_BY_COUNTRY)
    assert s['grp'].notna().all(), s.loc[s.grp.isna(),'country'].unique()
    s['unit'] = s['country'] + ' | ' + s['city']
    # 상대 기간: pre 는 뒤에서 -1,-2 ... / post 는 앞에서 +1,+2 ...
    out = []
    for u, g in s.groupby('unit'):
        g = g.sort_values('publish_date').copy()
        pre = g[g.side=='pre'].sort_values('publish_date')
        post = g[g.side=='post'].sort_values('publish_date')
        for i, (_, row) in enumerate(pre.iloc[::-1].iterrows()):   # 가장 가까운 pre = -1
            r = row.copy(); r['t_rel'] = -(i+1); out.append(r)
        for i, (_, row) in enumerate(post.iterrows()):
            r = row.copy(); r['t_rel'] = i+1; out.append(r)
    p = pd.DataFrame(out)
    p['post'] = (p.t_rel > 0).astype(int)
    return p

def did(panel, treat_groups, control_groups, ycol='pct_number', label='', weight=False):
    """y ~ unit FE + t_rel FE + post*treat . 군집=unit."""
    d = panel[panel.grp.isin(list(treat_groups)+list(controls_list(control_groups)))].copy()
    d = d[d.t_rel.isin([-2,-1,1,2])]
    # 전후 모두 있는 단위만
    ok = d.groupby('unit')['post'].agg(['min','max'])
    d = d[d.unit.isin(ok[(ok['min']==0)&(ok['max']==1)].index)]
    d['treat'] = d.grp.isin(list(treat_groups)).astype(int)
    d = d.dropna(subset=[ycol])
    if d.empty or d.treat.nunique() < 2: return None
    y = d[ycol].to_numpy(float)
    U = pd.get_dummies(d['unit'], drop_first=True, dtype=float).to_numpy()
    T = pd.get_dummies(d['t_rel'], drop_first=True, dtype=float).to_numpy()
    inter = (d['post']*d['treat']).to_numpy(float).reshape(-1,1)
    X = np.hstack([np.ones((len(d),1)), inter, U, T])
    res = wild_cluster_boot(y, X, d['unit'].to_numpy(), j=1)
    res.update(label=label, y=ycol,
               treat_groups=list(treat_groups), control_groups=list(controls_list(control_groups)),
               n_treat_clusters=int(d[d.treat==1]['unit'].nunique()),
               n_ctrl_clusters=int(d[d.treat==0]['unit'].nunique()),
               mean_treat_pre=round(float(d[(d.treat==1)&(d.post==0)][ycol].mean()),4),
               mean_treat_post=round(float(d[(d.treat==1)&(d.post==1)][ycol].mean()),4),
               mean_ctrl_pre=round(float(d[(d.treat==0)&(d.post==0)][ycol].mean()),4),
               mean_ctrl_post=round(float(d[(d.treat==0)&(d.post==1)][ycol].mean()),4))
    res['raw_did'] = round((res['mean_treat_post']-res['mean_treat_pre'])
                          -(res['mean_ctrl_post']-res['mean_ctrl_pre']),4)
    res['randomization'] = randomization_inference(
        d[['unit','t_rel','post',ycol]].dropna(), ycol,
        d[d.treat==1]['unit'].unique().tolist(), nperm=2000)
    return res

def controls_list(c): return c if isinstance(c,(list,tuple,set)) else [c]

def event_study(panel, treat_groups, control_groups, ycol='pct_number'):
    """t_rel=-1 기준, 각 기간 처치x기간 계수."""
    d = panel[panel.grp.isin(list(treat_groups)+list(controls_list(control_groups)))].copy()
    d = d[d.t_rel.isin([-2,-1,1,2])]
    ok = d.groupby('unit')['t_rel'].nunique()
    d = d[d.unit.isin(ok[ok>=3].index)]
    d['treat'] = d.grp.isin(list(treat_groups)).astype(int)
    d = d.dropna(subset=[ycol])
    out = {}
    for k in (-2,1,2):
        d['ind'] = ((d.t_rel==k)&(d.treat==1)).astype(float)
        # -1 과 k 만 쓰는 2기간 비교로 각 계수를 따로 추정(해석 단순)
        sub = d[d.t_rel.isin([-1,k])].copy()
        ok2 = sub.groupby('unit')['t_rel'].nunique()
        sub = sub[sub.unit.isin(ok2[ok2==2].index)]
        if sub.empty or sub.treat.nunique()<2: out[f't{k:+d}']=None; continue
        y = sub[ycol].to_numpy(float)
        U = pd.get_dummies(sub['unit'],drop_first=True,dtype=float).to_numpy()
        post_k = (sub.t_rel==k).astype(float).to_numpy().reshape(-1,1)
        inter = (post_k.ravel()*sub.treat.to_numpy()).reshape(-1,1)
        X = np.hstack([np.ones((len(sub),1)), inter, post_k, U])
        r = wild_cluster_boot(y, X, sub['unit'].to_numpy(), j=1)
        r['n_treat_clusters']=int(sub[sub.treat==1]['unit'].nunique())
        r['n_ctrl_clusters']=int(sub[sub.treat==0]['unit'].nunique())
        out[f't{k:+d}'] = r
    return out

def main():
    R = {'meta': {}, 'descriptives': {}, 'event_study': {}, 'did': {}, 'transitions': {}}
    panel = build_panel()
    panel.to_csv(os.path.join(BASE,'out','city_period_panel.csv'),index=False)
    R['meta']['n_units'] = int(panel.unit.nunique())
    R['meta']['n_city_snapshots'] = int(len(panel))
    R['meta']['n_listings_total'] = int(panel.n_listings.sum())
    R['meta']['cutoff'] = '2026-05-20'
    R['meta']['nboot'] = NBOOT
    R['meta']['group_definition_source'] = 'methods/ (1차 출처)'
    R['meta']['groups'] = {g: sorted(panel[panel.grp==g].unit.unique().tolist())
                           for g in sorted(panel.grp.unique())}
    R['meta']['n_units_by_group'] = {g:int(panel[panel.grp==g].unit.nunique())
                                     for g in sorted(panel.grp.unique())}

    # --- 기술통계: 그룹 x 기간 표시율 (가중 = 숙소 수) ---
    for g, gd in panel.groupby('grp'):
        rows={}
        for t, td in gd.groupby('t_rel'):
            w = td.n_listings.to_numpy(float)
            rows[f't{int(t):+d}'] = dict(
                n_units=int(td.unit.nunique()), n_listings=int(w.sum()),
                pct_number_unweighted=round(float(td.pct_number.mean()),3),
                pct_number_weighted=round(float(np.average(td.pct_number,weights=w)),3),
                pct_blank_weighted=round(float(np.average(td.pct_blank,weights=w)),3),
                pct_exempt_weighted=round(float(np.average(td.pct_exempt,weights=w)),3),
                pct_pending_weighted=round(float(np.average(td.pct_pending,weights=w)),3),
                pct_text_other_weighted=round(float(np.average(td.pct_text_other,weights=w)),3),
                pct_number_multi=round(float(np.average(td.pct_number_multi.fillna(0),weights=w)),3),
                pct_number_single=round(float(np.average(td.pct_number_single.fillna(0),weights=w)),3))
        R['descriptives'][g]=rows

    # --- 주 비교 ---
    specs = [
      ('MAIN_FRvsNonEU',      ['EU_aligned'], ['nonEU']),
      ('FRvsEUnoscheme',      ['EU_aligned'], ['EU_noscheme']),
      ('EUconfirmedVsNonEU',  ['EU_aligned','EU_pretreated'], ['nonEU']),
      ('ESreversalVsNonEU',   ['EU_reversal'], ['nonEU']),
      ('PretreatedVsNonEU',   ['EU_pretreated'], ['nonEU']),
      ('ROBUST_allEUvsNonEU', ['EU_aligned','EU_pretreated','EU_reversal','EU_unconfirmed'], ['nonEU']),
      ('PLACEBO_EEAvsNonEU',  ['EEA_nonEU'], ['nonEU']),
    ]
    for name, tg, cg in specs:
        R['did'][name] = did(panel, tg, cg, 'pct_number', name)
        R['event_study'][name] = event_study(panel, tg, cg, 'pct_number')

    # --- 플라시보: 비EU 에 가짜 시행일 (t=-1 을 가짜 cutoff 로) ---
    pl = panel[panel.grp=='nonEU'].copy()
    pl = pl[pl.t_rel.isin([-2,-1,1])]
    pl['fake_post'] = (pl.t_rel>=-1).astype(int)   # -2 vs (-1,+1)
    # 비EU 내부를 임의 반분해 가짜 처치
    units = sorted(pl.unit.unique()); RNG.shuffle(units)
    fake_t = set(units[:len(units)//2])
    pl['treat'] = pl.unit.isin(fake_t).astype(int)
    d = pl.dropna(subset=['pct_number'])
    y=d.pct_number.to_numpy(float)
    U=pd.get_dummies(d['unit'],drop_first=True,dtype=float).to_numpy()
    T=pd.get_dummies(d['t_rel'],drop_first=True,dtype=float).to_numpy()
    inter=(d.fake_post*d.treat).to_numpy(float).reshape(-1,1)
    X=np.hstack([np.ones((len(d),1)),inter,U,T])
    R['did']['PLACEBO_fake_cutoff_nonEU']=wild_cluster_boot(y,X,d['unit'].to_numpy(),j=1)
    R['did']['PLACEBO_fake_cutoff_nonEU']['label']='비EU 도시 임의 반분 + 가짜 시행일'

    # --- 숙소 수준 전이 ---
    tr = pd.read_csv(os.path.join(BASE,'out','panel_transitions.csv'))
    tr['grp']=tr['country'].map(GROUP_BY_COUNTRY)
    tr['unit']=tr['country']+' | '+tr['city']
    tr.to_csv(os.path.join(BASE,'out','panel_transitions_grouped.csv'),index=False)
    for seg in ('all','multi','single'):
        R['transitions'][seg]={}
        for g, gd in tr[tr.segment==seg].groupby('grp'):
            rows={}
            for pair, pdta in gd.groupby('pair'):
                w=pdta.n_blank_t0.to_numpy(float)
                if w.sum()==0: continue
                rows[pair]=dict(n_units=int(pdta.unit.nunique()),
                    n_blank_t0=int(w.sum()),
                    pct_got_number=round(float(np.average(pdta.pct_got_number,weights=w)),3),
                    pct_gone=round(float(np.average(pdta.pct_gone,weights=w)),3),
                    pct_still_blank=round(float(np.average(pdta.pct_still_blank,weights=w)),3),
                    pct_exempt=round(float(np.average(pdta.pct_exempt,weights=w)),3),
                    pct_pending=round(float(np.average(pdta.pct_pending,weights=w)),3))
            R['transitions'][seg][g]=rows
    # 전이 DiD: (main - pretrend) 를 EU vs 비EU 로
    R['transitions']['did']={}
    for seg in ('all','multi','single'):
        for ycol in ('pct_got_number','pct_gone'):
            w = tr[(tr.segment==seg)&tr.pair.isin(['main','pretrend'])]
            piv = w.pivot_table(index=['unit','grp'],columns='pair',values=ycol)
            piv = piv.dropna().reset_index()
            piv['delta']=piv['main']-piv['pretrend']
            for name,tg,cg in [('FRvsNonEU',['EU_aligned'],['nonEU']),
                               ('EUconfirmedVsNonEU',['EU_aligned','EU_pretreated'],['nonEU']),
                               ('ESreversalVsNonEU',['EU_reversal'],['nonEU'])]:
                sub=piv[piv.grp.isin(tg+cg)].copy()
                if sub.empty or sub.grp.isin(tg).nunique()<2: continue
                sub['treat']=sub.grp.isin(tg).astype(int)
                if sub.treat.nunique()<2: continue
                y=sub.delta.to_numpy(float)
                X=np.hstack([np.ones((len(sub),1)),sub.treat.to_numpy(float).reshape(-1,1)])
                r=wild_cluster_boot(y,X,sub['unit'].to_numpy(),j=1)
                r.update(label=f'{ycol} Δ(main-pretrend) {name} seg={seg}',
                         mean_treat=round(float(sub[sub.treat==1].delta.mean()),3),
                         mean_ctrl=round(float(sub[sub.treat==0].delta.mean()),3))
                R['transitions']['did'][f'{seg}|{ycol}|{name}']=r

    out=os.path.join(BASE,'out','results.json')
    json.dump(R,open(out,'w'),indent=1,ensure_ascii=False,default=str)
    print('WROTE',out)
    print(json.dumps(R['meta']['n_units_by_group'],indent=1))
    for k,v in R['did'].items():
        if v: print(f"  {k}: raw_did={v.get('raw_did')} coef={v['coef']:.3f} p={v['p_wildboot']} G={v['n_clusters']}")
if __name__=='__main__': main()
