#!/usr/bin/env python3
"""Listing-level panel across snapshots: what listings without a number did (acquired a number, left, unchanged)."""
import os, sys, csv
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_04_summarise import load
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def transitions(d0, d1, label, meta):
    """d0 에서 번호 없던(blank) 숙소가 d1 에서 어떻게 됐는지."""
    a = d0.dropna(subset=['id']).drop_duplicates('id').set_index('id')
    b = d1.dropna(subset=['id']).drop_duplicates('id').set_index('id')
    out = []
    # 기준 집합: d0 에서 lic_class=='blank' (번호 없음)
    for seg, mask in (('all', a['multi'].notna()),
                      ('multi', a['multi']==1),
                      ('single', a['multi']==0)):
        base = a[(a['lic_class']=='blank') & mask]
        n0 = len(base)
        if n0 == 0:
            continue
        present = base.index.intersection(b.index)
        gone = n0 - len(present)
        bb = b.loc[present]
        got = int((bb['lic_class']=='number').sum())
        exem = int((bb['lic_class']=='exempt').sum())
        pend = int((bb['lic_class']=='pending').sum())
        still = int((bb['lic_class']=='blank').sum())
        other = len(present) - got - exem - pend - still
        r = dict(meta)
        r.update(pair=label, segment=seg, n_blank_t0=n0,
                 n_gone=gone, pct_gone=round(100*gone/n0,3),
                 n_got_number=got, pct_got_number=round(100*got/n0,3),
                 n_exempt=exem, pct_exempt=round(100*exem/n0,3),
                 n_pending=pend, pct_pending=round(100*pend/n0,3),
                 n_still_blank=still, pct_still_blank=round(100*still/n0,3),
                 n_other=other, pct_other=round(100*other/n0,3),
                 n_t0_total=len(a), n_t1_total=len(b),
                 pct_blank_t0=round(100*(a['lic_class']=='blank').mean(),3),
                 pct_number_t0=round(100*(a['lic_class']=='number').mean(),3),
                 pct_number_t1=round(100*(b['lic_class']=='number').mean(),3))
        out.append(r)
    return out

def main():
    plan = list(csv.DictReader(open(os.path.join(BASE,'data','catalog','download_plan.csv'))))
    byunit = {}
    for p in plan:
        byunit.setdefault((p['country'],p['city']), []).append(p)
    rows = []
    for k, ps in sorted(byunit.items()):
        ps.sort(key=lambda r: r['publish_date'])
        pres  = [p for p in ps if p['side']=='pre']
        posts = [p for p in ps if p['side']=='post']
        if not pres or not posts:
            print('skip (no pre/post)', k, flush=True); continue
        meta = dict(country=k[0], city=k[1], role=ps[0]['role'], group=ps[0]['group'])
        need = {}
        pairs = []
        pairs.append(('main', pres[-1], posts[0]))
        if len(pres) >= 2: pairs.append(('pretrend', pres[-2], pres[-1]))
        if len(posts) >= 2: pairs.append(('postpost', posts[0], posts[-1]))
        cache = {}
        ok = True
        for _,a,b in pairs:
            for p in (a,b):
                fp = os.path.join(BASE,p['dest'])
                if p['dest'] not in cache:
                    if not (os.path.exists(fp) and os.path.getsize(fp)>0):
                        print('MISSING',p['dest'],flush=True); ok=False; continue
                    try: cache[p['dest']] = load(fp)
                    except Exception as e:
                        print('ERR',p['dest'],type(e).__name__,e,flush=True); ok=False
        for lbl,a,b in pairs:
            if a['dest'] in cache and b['dest'] in cache:
                m = dict(meta); m.update(t0=a['publish_date'], t1=b['publish_date'])
                rows += transitions(cache[a['dest']], cache[b['dest']], lbl, m)
        print('done', k, len(rows), flush=True)
        cache.clear()
    pd.DataFrame(rows).to_csv(os.path.join(BASE,'out','panel_transitions.csv'),index=False)
    print('WROTE out/panel_transitions.csv rows=',len(rows))
if __name__=='__main__': main()
