#!/usr/bin/env python3
"""Heterogeneity model: tabular foundation model with city-grouped cross-validation and permutation importance."""
import os, sys, json, time, traceback
WEIGHTS='<local model weights>'
for k,v in dict(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',
                TABPFN_ALLOW_CLI_INTERACTION='False',TABPFN_DISABLE_TELEMETRY='1',
                TABPFN_MODEL_CACHE_DIR='<local model weights>').items():
    os.environ.setdefault(k,v)

TOKEN_SIGNS=('the credential file','access token','api token','token not found','authenticat',
             'api key','tabpfn_token','unauthor','log in','login','sign up','subscription')
def tripwire(msg):
    low=str(msg).lower(); hits=[s for s in TOKEN_SIGNS if s in low]
    if hits:
        print('\n🔴 STOP: 토큰/인증 요구 신호 감지 →',hits)
        print('🔴 지침대로 토큰을 찾거나 입력하지 않고 멈춘다. 메시지:',str(msg)[:500])
        return True
    return False

import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RNG=np.random.default_rng(20261030)
FEATS=['n_tokens','has_br','host_listings','multi','entire','private_room','superhost',
       'host_tenure_yrs','avail365','n_reviews','n_reviews_ltm','reviews_per_month',
       'price','min_nights','accommodates','lat','lon']
MAXN=int(os.environ.get('MAXN','8000'))

def prep(df):
    X=df[FEATS].astype('float64').copy()
    X=X.replace([np.inf,-np.inf],np.nan)
    for c in X.columns: X[c]=X[c].fillna(X[c].median())
    return X.to_numpy(dtype='float64'), df['y3'].astype(int).to_numpy(), df['unit'].to_numpy()

def cv_eval(make, X, y, groups, folds=4, tag=''):
    """도시(unit) 군집을 섞지 않는 교차검증. AUC(ovr macro) + balanced accuracy."""
    cls=np.unique(y)
    if len(cls)<2 or len(np.unique(groups))<folds:
        return None
    sgk=StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=20261030)
    aucs=[];bas=[];accs=[];secs=[]
    oof=np.full((len(y),len(cls)),np.nan)
    for i,(tr,te) in enumerate(sgk.split(X,y,groups)):
        if len(np.unique(y[tr]))<2: continue
        t0=time.time(); m=make(); m.fit(X[tr],y[tr])
        p=m.predict_proba(X[te]); secs.append(round(time.time()-t0,2))
        # 클래스 정렬
        pm=np.zeros((len(te),len(cls)))
        for j,c in enumerate(m.classes_):
            pm[:,list(cls).index(c)]=p[:,j]
        oof[te]=pm
        try: aucs.append(roc_auc_score(y[te],pm,multi_class='ovr',average='macro'))
        except Exception: pass
        pred=cls[pm.argmax(1)]
        bas.append(balanced_accuracy_score(y[te],pred)); accs.append(accuracy_score(y[te],pred))
        print(f'   [{tag}] fold{i} n_tr={len(tr)} n_te={len(te)} auc={aucs[-1] if aucs else float("nan"):.4f} ba={bas[-1]:.4f} {secs[-1]}s',flush=True)
        del m
    if not bas: return None
    return dict(auc_ovr_macro=round(float(np.mean(aucs)),4) if aucs else None,
                auc_sd=round(float(np.std(aucs)),4) if aucs else None,
                bal_acc=round(float(np.mean(bas)),4), accuracy=round(float(np.mean(accs)),4),
                folds=int(len(bas)), fit_secs_mean=round(float(np.mean(secs)),2),
                n=int(len(y)), class_share={int(c):round(float((y==c).mean()),4) for c in cls}), oof

def perm_importance(make, X, y, groups, nrep=3, tag=''):
    """순열 중요도(단일 분할, balanced accuracy 감소량)."""
    sgk=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=20261030)
    tr,te=next(iter(sgk.split(X,y,groups)))
    m=make(); m.fit(X[tr],y[tr])
    cls=m.classes_
    base=balanced_accuracy_score(y[te],cls[m.predict_proba(X[te]).argmax(1)])
    out=[]
    for j,f in enumerate(FEATS):
        drops=[]
        for r in range(nrep):
            Xp=X[te].copy()
            Xp[:,j]=RNG.permutation(Xp[:,j])
            b=balanced_accuracy_score(y[te],cls[m.predict_proba(Xp).argmax(1)])
            drops.append(base-b)
        out.append(dict(feature=f, drop_bal_acc=round(float(np.mean(drops)),5),
                        sd=round(float(np.std(drops)),5)))
        print(f'   [{tag}] perm {f}: {out[-1]["drop_bal_acc"]:+.5f}',flush=True)
    out.sort(key=lambda r:-r['drop_bal_acc'])
    return dict(baseline_bal_acc=round(float(base),4), importances=out)

def main():
    R={'weights_file':os.path.basename(WEIGHTS),'offline_env':{k:os.environ[k] for k in
        ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','TABPFN_ALLOW_CLI_INTERACTION')},
       'the credential file_touched':False,'note':'the credential file 파일은 열거나 읽지 않았다.'}
    import torch
    R['torch']=torch.__version__; R['cuda']=bool(torch.cuda.is_available())
    R['gpu']=torch.cuda.get_device_name(0) if R['cuda'] else None
    print('torch',R['torch'],'cuda',R['cuda'],R['gpu'])
    print('weights',os.path.exists(WEIGHTS),os.path.getsize(WEIGHTS))

    df=pd.read_csv(os.path.join(BASE,'out','ml_sample.csv'))
    df['unit']=df['country']+' | '+df['city']
    R['sample']=dict(n_total=int(len(df)),n_units=int(df.unit.nunique()),
                     by_grp=df.grp.value_counts().to_dict(),
                     y3_share={int(k):round(float(v),4) for k,v in df.y3.value_counts(normalize=True).items()})
    print('sample',R['sample']['n_total'],'units',R['sample']['n_units'])

    model='TabPFN'
    try:
        from tabpfn import TabPFNClassifier
        import tabpfn; R['tabpfn_version']=getattr(tabpfn,'__version__','?')
        def make():
            return TabPFNClassifier(model_path=WEIGHTS, device='cuda' if R['cuda'] else 'cpu',
                                    n_estimators=1, random_state=20261030,
                                    ignore_pretraining_limits=True, show_progress_bar=False)
        # 연기 테스트
        sub=df.sample(min(600,len(df)),random_state=1)
        Xs,ys,gs=prep(sub)
        t0=time.time(); m=make(); m.fit(Xs,ys); _=m.predict_proba(Xs[:100])
        R['smoke_fit_secs']=round(time.time()-t0,2); del m
        print('TabPFN smoke OK',R['smoke_fit_secs'],'s')
    except Exception as e:
        tb=traceback.format_exc()
        if tripwire(str(e)+tb):
            R['TOKEN_REQUIRED']=True; R['stopped']='토큰 요구로 중단'
            json.dump(R,open(os.path.join(BASE,'out','ml_results.json'),'w'),indent=1,ensure_ascii=False)
            sys.exit(3)
        print('TabPFN 불가 → GBM 대체:',type(e).__name__,str(e)[:300])
        R['tabpfn_error']=f'{type(e).__name__}: {str(e)[:300]}'
        model='GBM(HistGradientBoosting)'
        from sklearn.ensemble import HistGradientBoostingClassifier
        def make(): return HistGradientBoostingClassifier(random_state=20261030,max_iter=300)
    R['model']=model
    print('MODEL =',model)

    # EU / 비EU 분리 학습
    for name,mask in [('eu', df.grp.astype(str).str.startswith('EU')),
                      ('noneu', df.grp.isin(['nonEU','EEA_nonEU'])),
                      ('es', df.grp=='EU_reversal'),
                      ('steady', df.grp=='EU_pretreated')]:
        sub=df[mask]
        if len(sub)<400 or sub.y3.nunique()<2:
            R[f'cv_{name}']=None; print('skip',name,len(sub)); continue
        if len(sub)>MAXN: sub=sub.sample(MAXN,random_state=20261030)
        X,y,g=prep(sub)
        print(f'--- CV {name}: n={len(y)} units={len(np.unique(g))} classes={dict(zip(*np.unique(y,return_counts=True)))}')
        try:
            r=cv_eval(make,X,y,g,folds=4,tag=name)
        except Exception as e:
            if tripwire(e): R['TOKEN_REQUIRED']=True; break
            print('CV fail',name,type(e).__name__,str(e)[:200]); r=None
        if r: R[f'cv_{name}']=r[0]
        if name in ('eu','noneu') and r:
            try: R[f'perm_{name}']=perm_importance(make,X,y,g,tag=name)
            except Exception as e:
                if tripwire(e): R['TOKEN_REQUIRED']=True; break
                print('perm fail',name,type(e).__name__,str(e)[:200])

    # 표본 상한·속도 탐색
    caps=[]
    Xa,ya,ga=prep(df if len(df)<=20000 else df.sample(20000,random_state=3))
    for n in (1000,2000,4000,8000,16000):
        if n>len(ya): break
        try:
            t0=time.time(); m=make(); m.fit(Xa[:n],ya[:n]); _=m.predict_proba(Xa[:500])
            caps.append(dict(n=n,secs=round(time.time()-t0,2),status='ok'))
            print(f'  cap n={n} ok {caps[-1]["secs"]}s',flush=True); del m
            if R['cuda']:
                import torch; torch.cuda.empty_cache()
        except Exception as e:
            if tripwire(e): R['TOKEN_REQUIRED']=True; break
            caps.append(dict(n=n,secs=None,status=f'{type(e).__name__}: {str(e)[:150]}'))
            print(f'  cap n={n} FAIL {type(e).__name__}',flush=True); break
    R['sample_cap_probe']=caps
    json.dump(R,open(os.path.join(BASE,'out','ml_results.json'),'w'),indent=1,ensure_ascii=False,default=str)
    print('WROTE out/ml_results.json')
if __name__=='__main__': main()
