#!/usr/bin/env python3
"""Market x snapshot summaries of the registration field. Only derived categories are kept; no field value is written."""
import os, sys, csv, gzip
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from s1_lic import classify, CLASSES
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USE = ['id','license','room_type','calculated_host_listings_count','host_id',
       'host_is_superhost','host_since','last_scraped','availability_365']

def load(path):
    df = pd.read_csv(path, usecols=lambda c: c in USE, low_memory=False,
                     dtype={'license':'string'})
    for c in USE:
        if c not in df.columns: df[c] = pd.NA
    d = pd.DataFrame(index=df.index)
    d['id'] = pd.to_numeric(df['id'], errors='coerce').astype('Int64')
    cl = df['license'].map(classify)          # 🔴 원문은 여기서만 쓰고 바로 버린다
    d['lic_class'] = [c['lic_class'] for c in cl]
    d['has_number'] = [c['has_number'] for c in cl]
    d['n_tokens'] = [c['n_tokens'] for c in cl]
    d['has_br'] = [c['has_br'] for c in cl]
    d['room_type'] = df['room_type']
    chlc = pd.to_numeric(df['calculated_host_listings_count'], errors='coerce')
    d['multi'] = (chlc >= 2).astype('Int8')
    d['host_id'] = pd.to_numeric(df['host_id'], errors='coerce').astype('Int64')
    d['superhost'] = (df['host_is_superhost'].astype('string') == 't').astype('Int8')
    d['avail365'] = pd.to_numeric(df['availability_365'], errors='coerce')
    d['entire'] = df['room_type'].astype('string').str.startswith('Entire').astype('Int8')
    del df, cl
    return d

def main():
    plan = list(csv.DictReader(open(os.path.join(BASE,'data','catalog','download_plan.csv'))))
    os.makedirs(os.path.join(BASE,'out'), exist_ok=True)
    rows, rtrows = [], []
    for i,p in enumerate(plan,1):
        fp = os.path.join(BASE, p['dest'])
        if not (os.path.exists(fp) and os.path.getsize(fp)>0):
            print(f"[{i}/{len(plan)}] MISSING {p['dest']}", flush=True); continue
        try:
            d = load(fp)
        except Exception as e:
            print(f"[{i}/{len(plan)}] ERR {p['dest']} {type(e).__name__}: {e}", flush=True); continue
        n = len(d)
        base = dict(role=p['role'], group=p['group'], country=p['country'], city=p['city'],
                    publish_date=p['publish_date'], side=p['side'], n_listings=n,
                    src_file=p['dest'])
        vc = d['lic_class'].value_counts()
        for c in CLASSES:
            base[f'n_{c}'] = int(vc.get(c,0))
            base[f'pct_{c}'] = round(100*vc.get(c,0)/n,3) if n else None
        base['pct_multi'] = round(100*d['multi'].mean(),3) if n else None
        base['pct_entire'] = round(100*d['entire'].mean(),3) if n else None
        base['pct_has_br'] = round(100*d['has_br'].mean(),3) if n else None
        base['mean_tokens_if_number'] = round(d.loc[d.has_number==1,'n_tokens'].mean(),3) if (d.has_number==1).any() else None
        # 다물건/1채 별 번호표시율
        for lbl,msk in (('multi', d['multi']==1), ('single', d['multi']==0)):
            sub=d[msk]; base[f'n_{lbl}']=len(sub)
            base[f'pct_number_{lbl}'] = round(100*sub['has_number'].mean(),3) if len(sub) else None
        rows.append(base)
        # room_type 별
        for rt,sub in d.groupby(d['room_type'].astype('string'),dropna=False):
            nn=len(sub)
            r=dict(country=p['country'],city=p['city'],publish_date=p['publish_date'],
                   role=p['role'],side=p['side'],room_type=(rt if rt==rt else 'NA'),n_listings=nn)
            v2=sub['lic_class'].value_counts()
            for c in CLASSES: r[f'pct_{c}']=round(100*v2.get(c,0)/nn,3) if nn else None
            rtrows.append(r)
        print(f"[{i}/{len(plan)}] {p['country']}/{p['city']} {p['publish_date']} n={n} num%={base['pct_number']}", flush=True)
        del d
    pd.DataFrame(rows).to_csv(os.path.join(BASE,'out','city_snapshot_summary.csv'),index=False)
    pd.DataFrame(rtrows).to_csv(os.path.join(BASE,'out','city_snapshot_by_roomtype.csv'),index=False)
    print('WROTE out/city_snapshot_summary.csv rows=',len(rows))
if __name__=='__main__': main()
