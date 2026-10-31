#!/usr/bin/env python3
"""Coverage diagnostics showing that the March 2026 snapshot is incomplete, and estimates with December 2025 as the reference."""
import os, sys, csv, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_04_summarise import load
from s1_09_estimate import (GROUP_BY_COUNTRY, wild_cluster_boot,
                            randomization_inference, build_panel)
from s1_05_panel import transitions
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RNG = np.random.default_rng(20261030)

def coverage_diag(panel):
    piv = panel.pivot_table(index='unit', columns='t_rel', values='n_listings')
    piv.columns = [f't{int(c):+d}' for c in piv.columns]
    piv = piv.copy()
    need = {'t-2','t-1','t+1'}
    if need <= set(piv.columns):
        piv['ratio_tm1'] = piv['t-1'] / ((piv['t-2'] + piv['t+1'])/2)
    grp = panel.groupby('unit')['grp'].first()
    piv['grp'] = grp
    piv.to_csv(os.path.join(BASE,'out','coverage_diagnostics.csv'))
    return piv

def did_on(panel, treat, ctrl, ycol, keep_t, label):
    d = panel[panel.grp.isin(list(treat)+list(ctrl))].copy()
    d = d[d.t_rel.isin(keep_t)]
    ok = d.groupby('unit')['t_rel'].nunique()
    d = d[d.unit.isin(ok[ok == len(keep_t)].index)]
    d['treat'] = d.grp.isin(list(treat)).astype(int)
    d['post'] = (d.t_rel > 0).astype(int)
    d = d.dropna(subset=[ycol])
    if d.empty or d.treat.nunique() < 2: return None
    y = d[ycol].to_numpy(float)
    U = pd.get_dummies(d['unit'], drop_first=True, dtype=float).to_numpy()
    T = pd.get_dummies(d['t_rel'], drop_first=True, dtype=float).to_numpy()
    inter = (d['post']*d['treat']).to_numpy(float).reshape(-1,1)
    X = np.hstack([np.ones((len(d),1)), inter, U, T])
    r = wild_cluster_boot(y, X, d['unit'].to_numpy(), j=1)
    r.update(label=label, y=ycol, periods=keep_t,
             treat_groups=list(treat), control_groups=list(ctrl),
             n_treat_clusters=int(d[d.treat==1].unit.nunique()),
             n_ctrl_clusters=int(d[d.treat==0].unit.nunique()),
             mean_treat_pre=round(float(d[(d.treat==1)&(d.post==0)][ycol].mean()),4),
             mean_treat_post=round(float(d[(d.treat==1)&(d.post==1)][ycol].mean()),4),
             mean_ctrl_pre=round(float(d[(d.treat==0)&(d.post==0)][ycol].mean()),4),
             mean_ctrl_post=round(float(d[(d.treat==0)&(d.post==1)][ycol].mean()),4))
    r['raw_did'] = round((r['mean_treat_post']-r['mean_treat_pre'])
                        -(r['mean_ctrl_post']-r['mean_ctrl_pre']),4)
    r['randomization'] = randomization_inference(
        d[['unit','t_rel','post',ycol]].dropna(), ycol,
        d[d.treat==1].unit.unique().tolist(), nperm=2000)
    return r

def main():
    R = {'why': ('2026-03 스냅샷이 체계적으로 미완. 기준기간으로 쓰면 사전 표시율이 과대평가된다. '
                 '따라서 2025-12(pre) vs 2026-06/09(post) 만 사용한다.')}
    panel = build_panel()
    cov = coverage_diag(panel)
    R['coverage'] = dict(
        n_units=int(len(cov)),
        ratio_tm1_mean=round(float(cov['ratio_tm1'].mean()),4) if 'ratio_tm1' in cov else None,
        ratio_tm1_median=round(float(cov['ratio_tm1'].median()),4) if 'ratio_tm1' in cov else None,
        ratio_tm1_min=round(float(cov['ratio_tm1'].min()),4) if 'ratio_tm1' in cov else None,
        n_units_below_0_7=int((cov['ratio_tm1']<0.7).sum()) if 'ratio_tm1' in cov else None,
        listings_by_period={f't{int(t):+d}': int(panel[panel.t_rel==t].n_listings.sum())
                            for t in sorted(panel.t_rel.unique())},
        pct_number_weighted_by_period={
            f't{int(t):+d}': round(float(np.average(panel[panel.t_rel==t].pct_number,
                                   weights=panel[panel.t_rel==t].n_listings)),3)
            for t in sorted(panel.t_rel.unique())})
    # 보정 DiD: t-2 (2025-12) vs t+1,t+2
    R['did_clean'] = {}
    for name, tg, cg in [('MAIN_FRvsNonEU',['EU_aligned'],['nonEU']),
                         ('FRvsEUnoscheme',['EU_aligned'],['EU_noscheme']),
                         ('EUconfirmedVsNonEU',['EU_aligned','EU_pretreated'],['nonEU']),
                         ('ESreversalVsNonEU',['EU_reversal'],['nonEU']),
                         ('PretreatedVsNonEU',['EU_pretreated'],['nonEU']),
                         ('ROBUST_allEUvsNonEU',['EU_aligned','EU_pretreated','EU_reversal','EU_unconfirmed'],['nonEU']),
                         ('PLACEBO_EEAvsNonEU',['EEA_nonEU'],['nonEU'])]:
        R['did_clean'][name] = did_on(panel, tg, cg, 'pct_number', [-2,1], name+' (t-2 vs t+1)')
        R['did_clean'][name+'_withT2'] = did_on(panel, tg, cg, 'pct_number', [-2,1,2], name+' (t-2 vs t+1,+2)')
    # 사후 플라시보: t+1 vs t+2 (둘 다 시행 후 → 효과 0 이어야)
    pl = panel[panel.t_rel.isin([1,2])].copy()
    pl['post'] = (pl.t_rel==2).astype(int)
    for name, tg, cg in [('POSTPLACEBO_FRvsNonEU',['EU_aligned'],['nonEU']),
                         ('POSTPLACEBO_EUconfVsNonEU',['EU_aligned','EU_pretreated'],['nonEU'])]:
        d = pl[pl.grp.isin(tg+cg)].copy()
        ok = d.groupby('unit')['t_rel'].nunique(); d = d[d.unit.isin(ok[ok==2].index)]
        d['treat'] = d.grp.isin(tg).astype(int)
        d = d.dropna(subset=['pct_number'])
        if d.empty or d.treat.nunique()<2: continue
        y=d.pct_number.to_numpy(float)
        U=pd.get_dummies(d['unit'],drop_first=True,dtype=float).to_numpy()
        inter=(d.post*d.treat).to_numpy(float).reshape(-1,1)
        X=np.hstack([np.ones((len(d),1)),inter,d.post.to_numpy(float).reshape(-1,1),U])
        r=wild_cluster_boot(y,X,d['unit'].to_numpy(),j=1); r['label']=name
        R['did_clean'][name]=r

    # ---- 전이 재계산: 2025-12 -> 2026-06 (시행일 사이), 2026-06 -> 2026-09 (사후 플라시보)
    plan = list(csv.DictReader(open(os.path.join(BASE,'data','catalog','download_plan.csv'))))
    byunit={}
    for p in plan: byunit.setdefault((p['country'],p['city']),[]).append(p)
    rows=[]
    for k,ps in sorted(byunit.items()):
        ps.sort(key=lambda r:r['publish_date'])
        pres=[p for p in ps if p['side']=='pre']; posts=[p for p in ps if p['side']=='post']
        if not pres or not posts: continue
        meta=dict(country=k[0],city=k[1],role=ps[0]['role'],
                  grp=GROUP_BY_COUNTRY.get(k[0],'?'))
        pairs=[('main_clean',pres[0],posts[0])]          # 가장 이른 pre(2025-12) -> 첫 post(2026-06)
        if len(posts)>=2: pairs.append(('post_placebo',posts[0],posts[-1]))
        cache={}
        for lbl,a,b in pairs:
            for p in (a,b):
                fp=os.path.join(BASE,p['dest'])
                if p['dest'] not in cache and os.path.exists(fp) and os.path.getsize(fp)>0:
                    cache[p['dest']]=load(fp)
            if a['dest'] in cache and b['dest'] in cache:
                m=dict(meta); m.update(t0=a['publish_date'],t1=b['publish_date'])
                rows+=transitions(cache[a['dest']],cache[b['dest']],lbl,m)
        print('tr',k,len(rows),flush=True); cache.clear()
    tdf=pd.DataFrame(rows); tdf.to_csv(os.path.join(BASE,'out','panel_transitions_clean.csv'),index=False)
    R['transitions_clean']={}
    for seg in ('all','multi','single'):
        R['transitions_clean'][seg]={}
        for g,gd in tdf[tdf.segment==seg].groupby('grp'):
            rr={}
            for pair,pd_ in gd.groupby('pair'):
                w=pd_.n_blank_t0.to_numpy(float)
                if w.sum()==0: continue
                rr[pair]=dict(n_units=int(pd_.unit.nunique()) if 'unit' in pd_ else int(len(pd_)),
                  n_blank_t0=int(w.sum()),
                  pct_got_number=round(float(np.average(pd_.pct_got_number,weights=w)),3),
                  pct_gone=round(float(np.average(pd_.pct_gone,weights=w)),3),
                  pct_still_blank=round(float(np.average(pd_.pct_still_blank,weights=w)),3),
                  pct_exempt=round(float(np.average(pd_.pct_exempt,weights=w)),3),
                  pct_pending=round(float(np.average(pd_.pct_pending,weights=w)),3))
            R['transitions_clean'][seg][g]=rr
    # 전이 DiD: main_clean 의 got/gone 을 EU vs 비EU 로 (수준 비교)
    R['transitions_clean']['did_levels']={}
    for seg in ('all','multi','single'):
        for ycol in ('pct_got_number','pct_gone'):
            sub=tdf[(tdf.segment==seg)&(tdf.pair=='main_clean')].copy()
            for name,tg,cg in [('FRvsNonEU',['EU_aligned'],['nonEU']),
                               ('EUconfirmedVsNonEU',['EU_aligned','EU_pretreated'],['nonEU']),
                               ('ESreversalVsNonEU',['EU_reversal'],['nonEU'])]:
                s2=sub[sub.grp.isin(tg+cg)].copy()
                if s2.empty: continue
                s2['treat']=s2.grp.isin(tg).astype(int)
                if s2.treat.nunique()<2: continue
                s2['unit']=s2.country+' | '+s2.city
                y=s2[ycol].to_numpy(float)
                X=np.hstack([np.ones((len(s2),1)),s2.treat.to_numpy(float).reshape(-1,1)])
                r=wild_cluster_boot(y,X,s2['unit'].to_numpy(),j=1)
                r.update(label=f'{ycol} level diff {name} seg={seg} (pair=main_clean)',
                         mean_treat=round(float(s2[s2.treat==1][ycol].mean()),3),
                         mean_ctrl=round(float(s2[s2.treat==0][ycol].mean()),3))
                R['transitions_clean']['did_levels'][f'{seg}|{ycol}|{name}']=r
    out=os.path.join(BASE,'out','results_coverage.json')
    json.dump(R,open(out,'w'),indent=1,ensure_ascii=False,default=str)
    print('WROTE',out)
    for k,v in R['did_clean'].items():
        if v: print(f"  {k}: raw={v.get('raw_did')} coef={v['coef']:+.3f} pw={v['p_wildboot']} pr={(v.get('randomization') or {}).get('p_randomization')}")
if __name__=='__main__': main()
