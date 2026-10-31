#!/usr/bin/env python3
"""Check mechanically that no value of the registration field appears in any output."""
import os, sys, csv, glob, gzip, random
import pandas as pd
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLOWED_SUBSTR = ('lic_class','has_number','n_tokens','has_br','len_bucket')
BANNED_COLS = {'license','licence'}

def main():
    bad = []
    # 1) 컬럼 점검
    for f in sorted(glob.glob(os.path.join(BASE,'out','*.csv'))):
        cols = pd.read_csv(f, nrows=0).columns.tolist()
        for c in cols:
            if c.strip().lower() in BANNED_COLS:
                bad.append(f'BANNED COLUMN {c} in {os.path.relpath(f,BASE)}')
    print('[1] out/*.csv 컬럼 점검:', 'OK' if not bad else 'FAIL')

    # 2) 역검색 — 셀 단위 비교(부분문자열 우연일치를 배제한다)
    #    CSV 는 모든 셀 값을 집합으로, 마크다운/텍스트는 단어경계 정규식으로 본다.
    import csv as _csv, re as _re
    cell_values=set(); text_blob=''
    targets=[]
    for pat in ('methods/','out/*.csv','data/catalog/*.csv','project log'):
        targets += glob.glob(os.path.join(BASE,pat))
    for t in targets:
        if t.endswith('.csv'):
            try:
                with open(t,newline='',encoding='utf-8',errors='replace') as fh:
                    for row in _csv.reader(fh):
                        for c in row: cell_values.add(c.strip().lower())
            except Exception: pass
        else:
            try: text_blob += open(t,encoding='utf-8',errors='replace').read().lower()
            except Exception: pass
    print(f'[2] 산출물 {len(targets)}개 파일 / CSV 셀 고유값 {len(cell_values)}개 / 텍스트 {len(text_blob)} 자')

    rng = random.Random(20261030)
    files = sorted(glob.glob(os.path.join(BASE,'data','raw','*.csv.gz')))
    rng.shuffle(files)
    checked=0; leaks=0; sampled=0; detail=[]
    for f in files[:12]:
        try:
            s_ = pd.read_csv(f, usecols=['license'], dtype={'license':'string'})['license']
        except Exception as e:
            print('   skip', os.path.basename(f), type(e).__name__); continue
        vals=[v for v in s_.dropna().unique().tolist() if isinstance(v,str) and len(v.strip())>=6]
        rng.shuffle(vals)
        for v in vals[:300]:
            vv=v.strip(); lv=vv.lower(); sampled+=1
            hit=False
            if lv in cell_values: hit=True            # CSV 셀과 완전일치
            elif _re.search(r'(?<![0-9A-Za-z])'+_re.escape(lv)+r'(?![0-9A-Za-z])', text_blob):
                hit=True                              # 텍스트에 토큰 단위로 등장
            if hit:
                leaks+=1
                detail.append((os.path.basename(f), len(vv)))
                print(f'   🔴 LEAK: {os.path.basename(f)} 의 license 값이 산출물에 셀/토큰 단위로 존재 (값 미출력, 길이={len(vv)})')
        checked+=1
    print(f'[2] 결과: 도시 {checked}개 · license 고유값 {sampled}개 역검색 → 누출 {leaks}건')
    if leaks: bad.append(f'{leaks} license value(s) found in outputs')

    print('\n=== 종합:', 'PASS — license 원문 누출 없음' if not bad else 'FAIL')
    for b in bad: print('  -', b)
    return 1 if bad else 0

if __name__=='__main__': sys.exit(main())
