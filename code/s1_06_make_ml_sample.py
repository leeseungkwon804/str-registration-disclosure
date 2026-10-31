#!/usr/bin/env python3
"""Sample for the heterogeneity model: listings without a number in December 2025 and their June 2026 outcome (three classes)."""
import os, sys, csv
import numpy as np, pandas as pd
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE,'scripts'))
from s1_lic import classify
from s1_09_estimate import GROUP_BY_COUNTRY
RNG = np.random.default_rng(20261030)
PER_UNIT = 1200

COLS = ['id','license','room_type','calculated_host_listings_count','host_id',
        'host_is_superhost','host_since','availability_365','number_of_reviews',
        'number_of_reviews_ltm','price','minimum_nights','reviews_per_month',
        'latitude','longitude','accommodates']

def load_ml(path):
    df = pd.read_csv(path, usecols=lambda c: c in COLS, low_memory=False,
                     dtype={'license':'string'})
    for c in COLS:
        if c not in df.columns: df[c] = pd.NA
    d = pd.DataFrame(index=df.index)
    d['id'] = pd.to_numeric(df['id'], errors='coerce').astype('Int64')
    cl = df['license'].map(classify)      # 🔴 원문은 즉시 버린다
    d['lic_class'] = [c['lic_class'] for c in cl]
    d['n_tokens']  = [c['n_tokens'] for c in cl]
    d['has_br']    = [c['has_br'] for c in cl]
    chlc = pd.to_numeric(df['calculated_host_listings_count'], errors='coerce')
    d['host_listings'] = chlc
    d['multi'] = (chlc >= 2).astype('Int8')
    d['entire'] = df['room_type'].astype('string').str.startswith('Entire').astype('Int8')
    d['private_room'] = df['room_type'].astype('string').str.contains('Private room', na=False).astype('Int8')
    d['superhost'] = (df['host_is_superhost'].astype('string')=='t').astype('Int8')
    hs = pd.to_datetime(df['host_since'], errors='coerce')
    d['host_tenure_yrs'] = ((pd.Timestamp('2026-05-20')-hs).dt.days/365.25)
    d['avail365'] = pd.to_numeric(df['availability_365'], errors='coerce')
    d['n_reviews'] = pd.to_numeric(df['number_of_reviews'], errors='coerce')
    d['n_reviews_ltm'] = pd.to_numeric(df['number_of_reviews_ltm'], errors='coerce')
    d['reviews_per_month'] = pd.to_numeric(df['reviews_per_month'], errors='coerce')
    pr = df['price'].astype('string').str.replace(r'[^0-9.]','',regex=True)
    d['price'] = pd.to_numeric(pr, errors='coerce')
    d['min_nights'] = pd.to_numeric(df['minimum_nights'], errors='coerce')
    d['accommodates'] = pd.to_numeric(df['accommodates'], errors='coerce')
    d['lat'] = pd.to_numeric(df['latitude'], errors='coerce')
    d['lon'] = pd.to_numeric(df['longitude'], errors='coerce')
    return d

def main():
    plan = list(csv.DictReader(open(os.path.join(BASE,'data','catalog','download_plan.csv'))))
    byunit = {}
    for p in plan: byunit.setdefault((p['country'],p['city']),[]).append(p)
    out=[]
    for k,ps in sorted(byunit.items()):
        ps.sort(key=lambda r: r['publish_date'])
        pres=[p for p in ps if p['side']=='pre']; posts=[p for p in ps if p['side']=='post']
        if not pres or not posts: continue
        a,b = pres[0], posts[0]           # 2025-12 -> 2026-06 (미완 3월 회피)
        fa,fb = os.path.join(BASE,a['dest']), os.path.join(BASE,b['dest'])
        if not all(os.path.exists(f) and os.path.getsize(f)>0 for f in (fa,fb)):
            print('skip',k,flush=True); continue
        d0 = load_ml(fa); d1 = load_ml(fb)
        d0 = d0.dropna(subset=['id']).drop_duplicates('id').set_index('id')
        d1 = d1.dropna(subset=['id']).drop_duplicates('id').set_index('id')
        base = d0[d0.lic_class=='blank'].copy()           # 시행 전 번호 없던 숙소
        if base.empty: continue
        present = base.index.intersection(d1.index)
        y = pd.Series(2, index=base.index, dtype=int)      # 기본 = 퇴출
        y.loc[present] = 0
        got = present[d1.loc[present,'lic_class']=='number']
        y.loc[got] = 1
        base['y3'] = y
        base = base.reset_index()
        if len(base) > PER_UNIT:
            base = base.iloc[RNG.choice(len(base), PER_UNIT, replace=False)]
        base['country']=k[0]; base['city']=k[1]
        base['grp']=GROUP_BY_COUNTRY.get(k[0],'?')
        base['eu'] = int(base['grp'].iloc[0].startswith('EU'))
        base['t0']=a['publish_date']; base['t1']=b['publish_date']
        out.append(base)
        print(f"ok {k} n={len(base)} still={100*(base.y3==0).mean():.1f} got={100*(base.y3==1).mean():.1f} gone={100*(base.y3==2).mean():.1f}",flush=True)
        del d0,d1,base
    df=pd.concat(out,ignore_index=True)
    df=df.drop(columns=[c for c in df.columns if c=='index'])
    os.makedirs(os.path.join(BASE,'out'),exist_ok=True)
    df.to_csv(os.path.join(BASE,'out','ml_sample.csv'),index=False)
    print('WROTE out/ml_sample.csv n=',len(df))
    print(df.groupby('grp').y3.value_counts(normalize=True).round(3).to_string())
if __name__=='__main__': main()
