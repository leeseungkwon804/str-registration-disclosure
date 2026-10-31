#!/usr/bin/env python3
"""Select the snapshots to download for each market (up to two before and two after the application date)."""
import csv, os
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 비유럽 대조군 14개 (영어권 4개국 — 2024/1028 적용 대상 아님)
CONTROLS = {
 ('United Kingdom','London'),('United Kingdom','Edinburgh'),
 ('United Kingdom','Greater Manchester'),('United Kingdom','Bristol'),
 ('United States','New York City'),('United States','Los Angeles'),
 ('United States','San Francisco'),('United States','Boston'),
 ('Canada','Toronto'),('Canada','Vancouver'),('Canada','Montreal'),
 ('Australia','Sydney'),('Australia','Melbourne'),
 ('New Zealand','New Zealand (nationwide)'),
}

def main():
    snaps=list(csv.DictReader(open(os.path.join(BASE,'data','catalog','snapshots.csv'))))
    elig=list(csv.DictReader(open(os.path.join(BASE,'data','catalog','city_eligibility.csv'))))
    byunit={}
    for s in snaps: byunit.setdefault((s['country'],s['city']),[]).append(s)

    plan=[]
    for e in elig:
        k=(e['country'],e['city'])
        if e['eligible_ba']!='1': continue
        if e['group']=='EU' or e['group']=='EEA_nonEU':
            role='treat_eu' if e['group']=='EU' else 'eea_noneu'
        elif k in CONTROLS:
            role='control_noneu'
        else:
            continue
        posts=sorted(d for d in e['post_dates'].split('|') if d)
        want=[d for d in (e['prev_pre'],e['last_pre'],e['first_post'],posts[-1]) if d]
        want=sorted(set(want))
        for s in byunit[k]:
            if s['publish_date'] in want:
                plan.append(dict(role=role, group=e['group'], country=e['country'], city=e['city'],
                    link=e['link'], publish_date=s['publish_date'], side=s['side'],
                    unit_level=s['unit_level'], url=s['listings_url'],
                    dest='data/raw/%s__%s__%s.csv.gz' % (
                        e['country'].replace(' ','_'), e['city'].replace(' ','_').replace(',','').replace('(','').replace(')',''), s['publish_date'])))
    plan.sort(key=lambda r:(r['role'],r['country'],r['city'],r['publish_date']))
    out=os.path.join(BASE,'data','catalog','download_plan.csv')
    with open(out,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(plan[0].keys())); w.writeheader(); w.writerows(plan)
    from collections import Counter
    print('plan rows=',len(plan))
    print(Counter(r['role'] for r in plan))
    print('units:',len({(r['country'],r['city']) for r in plan}))
    missing=CONTROLS-{(r['country'],r['city']) for r in plan if r['role']=='control_noneu'}
    print('controls not found:',missing)
if __name__=='__main__': main()
