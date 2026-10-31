#!/usr/bin/env python3
"""Download the earlier snapshots available for the comparison markets (pre-period check), with the same procedure."""
import csv, os, sys, time, urllib.request, urllib.error, hashlib, urllib.parse
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UA=("STR-Disclosure-Study/1.0 (academic research; EU Reg 2024/1028 short-term rental "
    "registration-disclosure study; contact the corresponding author)")
DELAY=2.5

def main():
    plan=list(csv.DictReader(open(os.path.join(BASE,'data','catalog','download_plan_pretrend.csv'))))
    os.makedirs(os.path.join(BASE,'data','raw'),exist_ok=True)
    logp=os.path.join(BASE,'data','catalog','download_log.csv')
    done=set()
    if os.path.exists(logp):
        for r in csv.DictReader(open(logp)):
            if r['status']=='ok': done.add(r['dest'])
    newlog = not os.path.exists(logp)
    lf=open(logp,'a',newline='',encoding='utf-8')
    lw=csv.DictWriter(lf,fieldnames=['dest','url','status','bytes','md5','http','secs'])
    if newlog: lw.writeheader()
    ok=skip=fail=0
    for i,r in enumerate(plan,1):
        dest=os.path.join(BASE,r['dest'])
        if r['dest'] in done and os.path.exists(dest) and os.path.getsize(dest)>0:
            skip+=1; continue
        t0=time.time()
        try:
            # 비ASCII 경로(zürich, pyrénées-atlantiques 등) 퍼센트 인코딩
            pr=urllib.parse.urlsplit(r['url'])
            safe=urllib.parse.urlunsplit((pr.scheme,pr.netloc,
                 urllib.parse.quote(pr.path,safe='/'),pr.query,pr.fragment))
            req=urllib.request.Request(safe,headers={'User-Agent':UA,'Accept-Encoding':'identity'})
            with urllib.request.urlopen(req,timeout=180) as resp:
                data=resp.read(); code=resp.status
            tmp=dest+'.part'
            open(tmp,'wb').write(data); os.replace(tmp,dest)
            md5=hashlib.md5(data).hexdigest()
            lw.writerow(dict(dest=r['dest'],url=r['url'],status='ok',bytes=len(data),
                             md5=md5,http=code,secs=round(time.time()-t0,2)))
            ok+=1
            print(f"[{i}/{len(plan)}] ok {r['dest']} {len(data)/1e6:.1f}MB",flush=True)
        except Exception as e:
            code=getattr(e,'code','')
            lw.writerow(dict(dest=r['dest'],url=r['url'],status='fail:%s'%type(e).__name__,
                             bytes=0,md5='',http=code,secs=round(time.time()-t0,2)))
            fail+=1
            print(f"[{i}/{len(plan)}] FAIL {r['dest']} {type(e).__name__} {code}",flush=True)
        lf.flush()
        time.sleep(DELAY)
    lf.close()
    print(f"DONE ok={ok} skip={skip} fail={fail}")
if __name__=='__main__': main()
