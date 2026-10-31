#!/usr/bin/env python3
"""General transition table: market x window x subgroup x origin state -> destination shares."""
import os, sys, csv
import numpy as np, pandas as pd
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.join(BASE,'scripts'))
from s1_06_make_ml_sample import load_ml
from s1_09_estimate import GROUP_BY_COUNTRY
REG={'EU_aligned':'fr','EU_reversal':'es','EU_pretreated':'steady','EU_noscheme':'noscheme',
     'nonEU':'noneu','EEA_nonEU':'eea','EU_unconfirmed':'unver'}

def segments(a):
    segs={'all':pd.Series(True,index=a.index),
          'multi':a['multi']==1,'single':a['multi']==0,
          'entire':a['entire']==1,'private':a['entire']==0}
    for col,pre in (('price','price'),('n_reviews','rev')):
        v=pd.to_numeric(a[col],errors='coerce')
        try:
            q=pd.qcut(v,4,labels=False,duplicates='drop')
        except Exception:
            q=pd.Series(np.nan,index=a.index)
        for i in range(4): segs[f'{pre}_q{i+1}']=(q==i)
    return segs

def main():
    plan=list(csv.DictReader(open(os.path.join(BASE,'data','catalog','download_plan.csv'))))
    byunit={}
    for p in plan: byunit.setdefault((p['country'],p['city']),[]).append(p)
    rows=[]
    for k,ps in sorted(byunit.items()):
        ps.sort(key=lambda r:r['publish_date'])
        pres=[p for p in ps if p['side']=='pre']; posts=[p for p in ps if p['side']=='post']
        if not pres or not posts: continue
        pairs=[('main',pres[0],posts[0])]                 # 2025-12 -> 2026-06
        if len(posts)>=2: pairs.append(('post_placebo',posts[0],posts[-1]))
        if len(pres)>=2:  pairs.append(('pretrend_raw',pres[0],pres[-1]))
        if len(posts)>=2: pairs.append(('main_t2',pres[0],posts[-1]))  # 2025-12 -> 2026-09
        cache={}
        for lbl,x,y in pairs:
            for p in (x,y):
                if p['dest'] not in cache:
                    fp=os.path.join(BASE,p['dest'])
                    if os.path.exists(fp) and os.path.getsize(fp)>0: cache[p['dest']]=load_ml(fp)
            if x['dest'] not in cache or y['dest'] not in cache: continue
            a=cache[x['dest']].dropna(subset=['id']).drop_duplicates('id').set_index('id')
            b=cache[y['dest']].dropna(subset=['id']).drop_duplicates('id').set_index('id')
            segs=segments(a)
            bcls=b['lic_class']
            for seg,mask in segs.items():
                for origin in ('number','blank'):
                    base=a[(a['lic_class']==origin)&mask.fillna(False)]
                    n0=len(base)
                    if n0==0: continue
                    present=base.index.intersection(b.index)
                    dest=bcls.loc[present]
                    rows.append(dict(country=k[0],city=k[1],unit=k[0]+' | '+k[1],
                        grp=GROUP_BY_COUNTRY.get(k[0],'?'),regime=REG.get(GROUP_BY_COUNTRY.get(k[0],'?'),'?'),
                        pair=lbl,t0=x['publish_date'],t1=y['publish_date'],segment=seg,origin=origin,
                        n0=n0,n_gone=n0-len(present),
                        n_number=int((dest=='number').sum()),n_exempt=int((dest=='exempt').sum()),
                        n_pending=int((dest=='pending').sum()),n_blank=int((dest=='blank').sum()),
                        n_other=int((dest=='text_other').sum())))
        print('ok',k,len(rows),flush=True); cache.clear()
    df=pd.DataFrame(rows)
    for c in ('gone','number','exempt','pending','blank','other'):
        df['pct_'+c]=(100*df['n_'+c]/df['n0']).round(4)
    df.to_csv(os.path.join(BASE,'out','transitions_v2.csv'),index=False)
    print('WROTE out/transitions_v2.csv rows=',len(df))
if __name__=='__main__': main()
