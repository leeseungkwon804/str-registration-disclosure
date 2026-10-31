#!/usr/bin/env python3
"""Listing transitions December 2025 -> June 2026 (main) and June -> September 2026, by regime and subgroup."""
import os, sys, csv
import numpy as np, pandas as pd
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE,'scripts'))
from s1_04_summarise import load
from s1_09_estimate import GROUP_BY_COUNTRY

SEGMENTS = {
 'all':     lambda a: a['multi'].notna() | a['multi'].isna(),
 'multi':   lambda a: a['multi']==1,
 'single':  lambda a: a['multi']==0,
 'entire':  lambda a: a['entire']==1,
 'private': lambda a: a['entire']==0,
}

def trans(d0, d1, pair, meta):
    a = d0.dropna(subset=['id']).drop_duplicates('id').set_index('id')
    b = d1.dropna(subset=['id']).drop_duplicates('id').set_index('id')
    out=[]
    for seg, fn in SEGMENTS.items():
        try: mask = fn(a)
        except Exception: continue
        base = a[(a['lic_class']=='blank') & mask]
        n0 = len(base)
        if n0 == 0: continue
        present = base.index.intersection(b.index)
        gone = n0 - len(present)
        bb = b.loc[present]
        got   = int((bb['lic_class']=='number').sum())
        exem  = int((bb['lic_class']=='exempt').sum())
        pend  = int((bb['lic_class']=='pending').sum())
        still = int((bb['lic_class']=='blank').sum())
        r=dict(meta); r.update(pair=pair, segment=seg, n_blank_t0=n0,
            n_gone=gone, pct_gone=round(100*gone/n0,4),
            n_acquire=got, pct_acquire=round(100*got/n0,4),
            n_exempt=exem, pct_exempt=round(100*exem/n0,4),
            n_pending=pend, pct_pending=round(100*pend/n0,4),
            n_same=still, pct_same=round(100*still/n0,4),
            n_t0=len(a), n_t1=len(b),
            pct_number_t0=round(100*(a['lic_class']=='number').mean(),4),
            pct_number_t1=round(100*(b['lic_class']=='number').mean(),4))
        out.append(r)
    # 번호 있던 숙소가 번호를 잃었나 (스페인 해석에 필요)
    hadnum = a[a['lic_class']=='number']
    if len(hadnum):
        pres2 = hadnum.index.intersection(b.index)
        b2 = b.loc[pres2]
        r=dict(meta); r.update(pair=pair, segment='had_number',
            n_blank_t0=len(hadnum),
            n_gone=len(hadnum)-len(pres2), pct_gone=round(100*(len(hadnum)-len(pres2))/len(hadnum),4),
            n_acquire=int((b2['lic_class']=='number').sum()),
            pct_acquire=round(100*(b2['lic_class']=='number').sum()/len(hadnum),4),
            n_exempt=int((b2['lic_class']=='exempt').sum()),
            pct_exempt=round(100*(b2['lic_class']=='exempt').sum()/len(hadnum),4),
            n_pending=int((b2['lic_class']=='pending').sum()),
            pct_pending=round(100*(b2['lic_class']=='pending').sum()/len(hadnum),4),
            n_same=int((b2['lic_class']=='blank').sum()),
            pct_same=round(100*(b2['lic_class']=='blank').sum()/len(hadnum),4),
            n_t0=len(a), n_t1=len(b),
            pct_number_t0=round(100*(a['lic_class']=='number').mean(),4),
            pct_number_t1=round(100*(b['lic_class']=='number').mean(),4))
        out.append(r)
    return out

def main():
    plan=list(csv.DictReader(open(os.path.join(BASE,'data','catalog','download_plan.csv'))))
    byunit={}
    for p in plan: byunit.setdefault((p['country'],p['city']),[]).append(p)
    rows=[]
    for k,ps in sorted(byunit.items()):
        ps.sort(key=lambda r:r['publish_date'])
        pres=[p for p in ps if p['side']=='pre']; posts=[p for p in ps if p['side']=='post']
        if not pres or not posts: continue
        meta=dict(country=k[0], city=k[1], unit=k[0]+' | '+k[1],
                  role=ps[0]['role'], grp=GROUP_BY_COUNTRY.get(k[0],'?'))
        pairs=[('main_clean',pres[0],posts[0])]
        if len(posts)>=2: pairs.append(('post_placebo',posts[0],posts[-1]))
        if len(pres)>=2:  pairs.append(('pretrend_raw',pres[0],pres[-1]))
        cache={}
        for lbl,x,y in pairs:
            for p in (x,y):
                fp=os.path.join(BASE,p['dest'])
                if p['dest'] not in cache and os.path.exists(fp) and os.path.getsize(fp)>0:
                    cache[p['dest']]=load(fp)
            if x['dest'] in cache and y['dest'] in cache:
                m=dict(meta); m.update(t0=x['publish_date'], t1=y['publish_date'])
                rows+=trans(cache[x['dest']], cache[y['dest']], lbl, m)
        print('ok',k,len(rows),flush=True); cache.clear()
    df=pd.DataFrame(rows)
    df.to_csv(os.path.join(BASE,'out','transitions_final.csv'),index=False)
    print('WROTE out/transitions_final.csv rows=',len(df))
if __name__=='__main__': main()
