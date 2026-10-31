#!/usr/bin/env python3
"""Check that the tabular foundation model loads from local weights and runs offline on the GPU."""
import os, sys, time, json, traceback

WEIGHTS = '<local model weights>'
# 오프라인 강제 — 어떤 다운로드/인증 경로도 타지 않게 한다
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('HF_DATASETS_OFFLINE','1')
os.environ.setdefault('TABPFN_MODEL_CACHE_DIR','<local model weights>')
os.environ.setdefault('TABPFN_ALLOW_CLI_INTERACTION','False')
os.environ.setdefault('TABPFN_DISABLE_TELEMETRY','1')

TOKEN_SIGNS = ('the credential file','access token','api token','token not found','authenticat',
               'login','api key','TABPFN_TOKEN','use_server','unauthor')

def token_tripwire(msg):
    low = str(msg).lower()
    hits = [s for s in TOKEN_SIGNS if s.lower() in low]
    if hits:
        print('\n🔴 STOP: 토큰/인증을 요구하는 신호가 감지됐다 →', hits)
        print('🔴 지침대로 토큰을 찾거나 입력하지 않고 멈춘다. 메시지 원문:')
        print('   ', str(msg)[:600])
        return True
    return False

def main():
    rep = {}
    import numpy as np, pandas as pd, torch
    rep['torch'] = torch.__version__
    rep['cuda_available'] = bool(torch.cuda.is_available())
    rep['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    print('torch', rep['torch'], 'cuda', rep['cuda_available'], rep['gpu'])
    print('weights exists:', os.path.exists(WEIGHTS), os.path.getsize(WEIGHTS) if os.path.exists(WEIGHTS) else 0)

    import tabpfn
    rep['tabpfn_version'] = getattr(tabpfn,'__version__','?')
    print('tabpfn', rep['tabpfn_version'])
    from tabpfn import TabPFNClassifier
    import inspect
    sig = list(inspect.signature(TabPFNClassifier.__init__).parameters)
    rep['init_params'] = sig
    print('TabPFNClassifier params:', sig)

    # 데이터
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(BASE,'out','ml_sample.csv'))
    df = df.dropna(subset=['y_gone'])
    feats = ['has_number','n_tokens','has_br','multi','superhost','avail365','entire']
    X = pd.get_dummies(df[feats+['lic_class']], columns=['lic_class'], dtype=float)
    X = X.fillna(X.median(numeric_only=True)).to_numpy(dtype='float32')
    y = df['y_gone'].astype(int).to_numpy()
    rep['n_total'] = int(len(y)); rep['n_features'] = int(X.shape[1])
    rep['pos_rate'] = round(float(y.mean()),4)
    print('sample', X.shape, 'pos_rate', rep['pos_rate'])

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score

    # 가중치 경로를 받는 인자 이름을 찾아 넣는다
    kw = {}
    for cand in ('model_path','model_paths','checkpoint_path','path'):
        if cand in sig: kw[cand] = WEIGHTS; rep['weights_kwarg']=cand; break
    if 'device' in sig: kw['device'] = 'cuda' if rep['cuda_available'] else 'cpu'
    if 'n_estimators' in sig: kw['n_estimators'] = 1
    print('ctor kwargs:', {k:(v if k!='model_path' else '<local weights>') for k,v in kw.items()})

    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.3,random_state=20261030,stratify=y)
    t0=time.time(); clf = TabPFNClassifier(**kw); rep['ctor_secs']=round(time.time()-t0,2)
    t0=time.time(); clf.fit(Xtr,ytr); rep['fit_secs']=round(time.time()-t0,2)
    t0=time.time(); p = clf.predict_proba(Xte)[:,1]; rep['predict_secs']=round(time.time()-t0,2)
    rep['n_train']=int(len(ytr)); rep['n_test']=int(len(yte))
    rep['auc']=round(float(roc_auc_score(yte,p)),4)
    rep['acc']=round(float(accuracy_score(yte,(p>=0.5).astype(int))),4)
    print('FIT OK', json.dumps({k:rep[k] for k in ('ctor_secs','fit_secs','predict_secs','auc','acc','n_train','n_test')}))

    # 표본 상한 탐색 (fit 가능한 최대 train 크기)
    caps=[]
    for n in (1000,2000,4000,8000,16000,32000):
        if n > len(ytr): break
        try:
            t0=time.time(); c2=TabPFNClassifier(**kw); c2.fit(Xtr[:n],ytr[:n])
            _=c2.predict_proba(Xte[:500]); el=round(time.time()-t0,2)
            caps.append(dict(n=n,secs=el,status='ok')); print(f'  cap n={n} ok {el}s',flush=True)
            del c2
            if rep['cuda_available']: torch.cuda.empty_cache()
        except Exception as e:
            if token_tripwire(e): rep['TOKEN_REQUIRED']=True; break
            caps.append(dict(n=n,secs=None,status='fail:'+type(e).__name__+':'+str(e)[:200]))
            print(f'  cap n={n} FAIL {type(e).__name__}: {str(e)[:200]}',flush=True)
            break
    rep['sample_cap_probe']=caps
    out=os.path.join(BASE,'out','tabpfn_check.json')
    json.dump(rep,open(out,'w'),indent=1,ensure_ascii=False)
    print('WROTE',out)

if __name__=='__main__':
    try:
        main()
    except Exception as e:
        tb=traceback.format_exc()
        if token_tripwire(str(e)+tb):
            sys.exit(3)
        print('FAILED:',type(e).__name__,e); print(tb)
        sys.exit(1)
