#!/usr/bin/env python3
"""Build the catalogue of Inside Airbnb snapshots (all cities, all published dates) and mark which markets have snapshots before and after 20 May 2026."""
import json, csv, os
from datetime import date

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUTOFF = date(2026, 5, 20)  # EU Reg 2024/1028 적용일

# EU27
EU27 = {
 'Austria','Belgium','Bulgaria','Croatia','Cyprus','Czech Republic','Czechia','Denmark',
 'Estonia','Finland','France','Germany','Greece','Hungary','Ireland','Italy','Latvia',
 'Lithuania','Luxembourg','Malta','The Netherlands','Netherlands','Poland','Portugal',
 'Romania','Slovakia','Slovenia','Spain','Sweden'}
# EEA 비EU + 스위스(EFTA)
EEA_NONEU = {'Norway','Iceland','Liechtenstein','Switzerland'}

def group(country):
    if country in EU27: return 'EU'
    if country in EEA_NONEU: return 'EEA_nonEU'
    return 'nonEU'

def main():
    src = os.path.join(BASE,'data','catalog','insideairbnb_datasets_raw_20261030.json')
    ds = json.load(open(src))['data']['allData']['datasets']
    rows = []
    for d in ds:
        pdte = d['publishDate']
        # city 가 None 인 레코드 = 국가 단위 데이터셋(Ireland/Malta/New Zealand 등)
        cityname = d['city'] or (d['country'] + ' (nationwide)')
        y,m,dd = (int(x) for x in pdte.split('-'))
        url = d['dataRoot'].rstrip('/') + '/' + pdte + '/data/listings.csv.gz'
        rows.append(dict(
            city=cityname, country=d['country'], region=d.get('region') or '',
            region_short=d.get('regionShort') or '', link=d['link'],
            unit_level='country' if d['city'] is None else 'city',
            group=group(d['country']), publish_date=pdte,
            side='post' if date(y,m,dd) >= CUTOFF else 'pre',
            scrape_id=d['scrapeID'], listings_url=url))
    rows.sort(key=lambda r:(r['country'],r['city'],r['publish_date']))
    out = os.path.join(BASE,'data','catalog','snapshots.csv')
    with open(out,'w',newline='',encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('snapshots.csv rows=',len(rows))

    # 도시 단위 집계
    cities = {}
    for r in rows:
        k = (r['country'], r['city'])
        c = cities.setdefault(k, dict(country=r['country'], city=r['city'],
            group=r['group'], link=r['link'], n_snap=0, pre=[], post=[]))
        c['n_snap'] += 1
        c[r['side']].append(r['publish_date'])
    crows = []
    for k,c in sorted(cities.items()):
        c['pre'].sort(); c['post'].sort()
        crows.append(dict(country=c['country'], city=c['city'], group=c['group'], link=c['link'],
            n_snapshots=c['n_snap'], n_pre=len(c['pre']), n_post=len(c['post']),
            pre_dates='|'.join(c['pre']), post_dates='|'.join(c['post']),
            last_pre=c['pre'][-1] if c['pre'] else '',
            prev_pre=c['pre'][-2] if len(c['pre'])>=2 else '',
            first_post=c['post'][0] if c['post'] else '',
            eligible_ba=int(bool(c['pre']) and bool(c['post'])),
            eligible_pretrend=int(len(c['pre'])>=2 and bool(c['post']))))
    out2 = os.path.join(BASE,'data','catalog','city_eligibility.csv')
    with open(out2,'w',newline='',encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(crows[0].keys())); w.writeheader(); w.writerows(crows)
    print('city_eligibility.csv rows=',len(crows))

    # 요약
    from collections import Counter
    print('\n--- 스냅샷 날짜 분포 ---')
    for d,n in sorted(Counter(r['publish_date'] for r in rows).items()): print(' ',d,n)
    print('\n--- 그룹별 도시 수 / 전후 모두 있는 도시 수 ---')
    for g in ('EU','EEA_nonEU','nonEU'):
        sub=[c for c in crows if c['group']==g]
        print(f"  {g}: 도시 {len(sub)}, 전후 모두 {sum(c['eligible_ba'] for c in sub)}, 사전추세가능 {sum(c['eligible_pretrend'] for c in sub)}")
    print('\n--- EU 국가별 (전후 모두 있는 도시) ---')
    cc=Counter(c['country'] for c in crows if c['group']=='EU' and c['eligible_ba'])
    for k,v in sorted(cc.items()): print(' ',k,v)

if __name__=='__main__': main()
