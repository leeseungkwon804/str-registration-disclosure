#!/usr/bin/env python3
"""Pre-period checks with the snapshots available before December 2025."""
import os, sys, csv, json
import numpy as np, pandas as pd
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.join(BASE,'scripts'))
from s1_04_summarise import load
from s1_lic import CLASSES
RJ=os.path.join(BASE,'out','results.json'); R=json.load(open(RJ))

UNITS={('New Zealand','New Zealand (nationwide)'):'noneu',
       ('United Kingdom','Edinburgh'):'noneu',
       ('Switzerland','Vaud'):'eea'}

def main():
    snaps=list(csv.DictReader(open(os.path.join(BASE,'data','catalog','snapshots.csv'))))
    rows=[]
    for s in snaps:
        k=(s['country'],s['city'])
        if k not in UNITS: continue
        if not ('2025-07-01'<=s['publish_date']<='2026-10-31'): continue
        dest='data/raw/%s__%s__%s.csv.gz'%(k[0].replace(' ','_'),
             k[1].replace(' ','_').replace(',','').replace('(','').replace(')',''),s['publish_date'])
        fp=os.path.join(BASE,dest)
        if not (os.path.exists(fp) and os.path.getsize(fp)>0): continue
        d=load(fp); n=len(d)
        rows.append(dict(country=k[0],city=k[1],regime=UNITS[k],publish_date=s['publish_date'],
            n_listings=n, pct_number=round(100*d.has_number.mean(),4),
            pct_blank=round(100*(d.lic_class=='blank').mean(),4),
            pct_exempt=round(100*(d.lic_class=='exempt').mean(),4)))
        print('  ',k[1],s['publish_date'],n,rows[-1]['pct_number'],flush=True)
        del d
    df=pd.DataFrame(rows).sort_values(['city','publish_date'])
    df.to_csv(os.path.join(BASE,'out','pretrend_controls.csv'),index=False)

    R['pretrend_eu_units_available']=0
    R['pretrend_earliest_eu_snapshot']='2025-12-11'
    R['pretrend_control_units_available']=int(df.groupby(['country','city']).ngroups)
    R['pretrend_note']=('Inside Airbnb 무료 카탈로그는 「최근 1년 분기 자료」라서 EU 단위의 2025-12 이전 스냅샷이 '
        '하나도 없다(확인: EU 최초 스냅샷 2025-12-11). 따라서 처치군의 온전한 사전 구간은 만들 수 없고, '
        '추가 자료는 data request 가 필요하다. 아래 수치는 대조군 3개 단위의 사전 표류만을 말한다.')
    # 대조군 사전 표류: 2025-12 이전 마지막 -> 2025-12 -> 2026-06
    for (c,city),g in df.groupby(['country','city']):
        g=g.sort_values('publish_date')
        early=g[g.publish_date<'2025-12-01']
        dec=g[(g.publish_date>='2025-12-01')&(g.publish_date<'2026-01-01')]
        jun=g[(g.publish_date>='2026-05-20')&(g.publish_date<'2026-07-15')]
        tag=city.split(' ')[0].lower().replace('(','')
        if len(early): R[f'pretrend_{tag}_early_date']=early.publish_date.iloc[-1]; R[f'pretrend_{tag}_early_disp']=round(float(early.pct_number.iloc[-1]),1)
        if len(dec):   R[f'pretrend_{tag}_dec_disp']=round(float(dec.pct_number.iloc[0]),1)
        if len(jun):   R[f'pretrend_{tag}_jun_disp']=round(float(jun.pct_number.iloc[0]),1)
        if len(early) and len(dec):
            R[f'pretrend_{tag}_drift_pre']=round(float(dec.pct_number.iloc[0]-early.pct_number.iloc[-1]),1)
        if len(dec) and len(jun):
            R[f'pretrend_{tag}_drift_post']=round(float(jun.pct_number.iloc[0]-dec.pct_number.iloc[0]),1)
    # 커버리지 점검 (사전 스냅샷도 미완인가)
    cov={}
    for (c,city),g in df.groupby(['country','city']):
        g=g.sort_values('publish_date')
        e=g[g.publish_date<'2025-12-01']; d2=g[(g.publish_date>='2025-12-01')&(g.publish_date<'2026-01-01')]
        if len(e) and len(d2):
            cov[city]=round(float(e.n_listings.iloc[-1]/d2.n_listings.iloc[0]),3)
    R['pretrend_coverage_ratio_vs_dec']=cov
    json.dump(R,open(RJ,'w'),indent=1,ensure_ascii=False,default=str)
    print('WROTE results.json pretrend_* keys; coverage ratios:',cov)
if __name__=='__main__': main()
