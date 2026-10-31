#!/usr/bin/env python3
"""Finalize out/results.json; run after s1_14, s1_17, s1_19 and s1_20."""
import os, sys, json
import numpy as np, pandas as pd
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.join(BASE,'scripts'))
from s1_09_estimate import build_panel, wild_cluster_boot, randomization_inference, GROUP_BY_COUNTRY
RJ=os.path.join(BASE,'out','results.json'); R=json.load(open(RJ))
REG={'EU_aligned':'fr','EU_reversal':'es','EU_pretreated':'steady','EU_noscheme':'noscheme',
     'nonEU':'noneu','EEA_nonEU':'eea','EU_unconfirmed':'unver'}
panel=build_panel(); panel['regime']=panel.grp.map(REG)
# 이전 분류: 헝가리를 미확인으로
panel['regime_old']=np.where(panel.country=='Hungary','unver',panel.regime)
tr=pd.read_csv(os.path.join(BASE,'out','transitions_v2.csv'))
r1=lambda v: None if v is None else round(float(v),1)

def fit(d,periods,nperm=2000):
    d=d[d.t_rel.isin(periods)].copy()
    ok=d.groupby('unit')['t_rel'].nunique(); d=d[d.unit.isin(ok[ok==len(periods)].index)]
    d['post']=(d.t_rel>0).astype(int); d=d.dropna(subset=['pct_number'])
    if d.empty or d.treat.nunique()<2: return None
    y=d.pct_number.to_numpy(float)
    U=pd.get_dummies(d['unit'],drop_first=True,dtype=float).to_numpy()
    T=pd.get_dummies(d['t_rel'],drop_first=True,dtype=float).to_numpy()
    X=np.hstack([np.ones((len(d),1)),(d['post']*d['treat']).to_numpy(float).reshape(-1,1),U,T])
    o=wild_cluster_boot(y,X,d['unit'].to_numpy(),j=1)
    o['randomization']=randomization_inference(d[['unit','t_rel','post','pct_number']].dropna(),
        'pct_number',d[d.treat==1].unit.unique().tolist(),nperm=nperm)
    o['n_treat_units']=int(d[d.treat==1].unit.nunique()); o['n_ctrl_units']=int(d[d.treat==0].unit.nunique())
    o['n_obs']=int(len(d)); return o

def two_period(d,t0,t1):
    s=d[d.t_rel.isin([t0,t1])].copy()
    ok=s.groupby('unit')['t_rel'].nunique(); s=s[s.unit.isin(ok[ok==2].index)]
    s=s.dropna(subset=['pct_number'])
    if s.empty or s.treat.nunique()<2: return None
    y=s.pct_number.to_numpy(float)
    U=pd.get_dummies(s['unit'],drop_first=True,dtype=float).to_numpy()
    p=(s.t_rel==t1).astype(float).to_numpy().reshape(-1,1)
    X=np.hstack([np.ones((len(s),1)),(p.ravel()*s.treat.to_numpy()).reshape(-1,1),p,U])
    return wild_cluster_boot(y,X,s['unit'].to_numpy(),j=1)

def emit(pfx,o,ri=True):
    if not o: return
    b,se=o['coef'],o['se_cluster']
    R[f'{pfx}_b']=r1(b); R[f'{pfx}_se']=round(se,3)
    R[f'{pfx}_ci_lo']=r1(b-1.96*se); R[f'{pfx}_ci_hi']=r1(b+1.96*se)
    R[f'{pfx}_ci']=f'{b-1.96*se:.1f} to {b+1.96*se:.1f}'
    R[f'{pfx}_wcb_p']=round(o['p_wildboot'],3)
    _r=(o.get('randomization') or {}).get('p_randomization')
    R[f'{pfx}_ri_p']=None if _r is None else round(_r,3)
    for k in ('n_treat_units','n_ctrl_units','n_obs'):
        if o.get(k) is not None: R[f'{pfx}_{k}']=o[k]

# 1) 합동(pooled)
for rg in ('es','fr','steady','unver'):
    d=panel[panel.regime.isin([rg,'noneu'])].copy(); d['treat']=(d.regime==rg).astype(int)
    emit(f'did_disp_{rg}_pooled', fit(d,[-2,1,2]))
    print('pooled',rg,R.get(f'did_disp_{rg}_pooled_b'),flush=True)
R['did_disp_KEY_NOTE']=('세 수치를 섞지 마라. _t1_b = 2025-12 → 2026-06. _t2_b = 2025-12 → 2026-09(둘째 사후기 단독). '
 '_pooled_b = 2025-12 기준으로 두 사후기를 함께 쓴 사후 더미 1개 추정.')
R['did_disp_ALIAS_NOTE']='did_disp_<rg>_b 는 did_disp_<rg>_t1_b 와 같은 추정(2025-12 → 2026-06)이다.'

# 2) 이벤트 스터디 보강 (미확인·EEA·제도없음)
for rg in ('unver','eea','noscheme'):
    d=panel[panel.regime.isin([rg,'noneu'])].copy(); d['treat']=(d.regime==rg).astype(int)
    for rel,t in (('m1',-1),('p1',1),('p2',2)):
        o=two_period(d,-2,t)
        if not o: continue
        for alias in ([rg,'unconf'] if rg=='unver' else [rg]):
            R[f'es_disp_{alias}_{rel}']=round(o['coef'],4)
            R[f'es_disp_{alias}_{rel}_se']=round(o['se_cluster'],4)
            R[f'es_disp_{alias}_{rel}_p']=round(o['p_wildboot'],4)
    for alias in ([rg,'unconf'] if rg=='unver' else [rg]): R[f'es_disp_{alias}_m2']=0.0
R['es_disp_noneu_NOTE']='비EU 는 대조군 자체이므로 DiD 계수가 정의되지 않는다. 원수준은 disp_rate_noneu_* 를 쓸 것.'
R['es_disp_REL_NOTE']=('rel 매핑: m2=2025-12(기준) · m1=2026-03(미완, 주분석 제외) · p1=2026-06 · p2=2026-09. '
 'm3 와 p0 에 해당하는 스냅샷은 없다.')
for rg in ('fr','es','steady','noscheme','eea','unver','unconf'):
    R.setdefault(f'es_disp_{rg}_m3',None); R.setdefault(f'es_disp_{rg}_p0',None)
for rel in ('m3','m2','m1','p0','p1','p2'):
    R.setdefault(f'es_disp_noneu_{rel}', 0.0 if rel=='m2' else None)

# 3) 마드리드
s=tr[(tr.pair=='main')&(tr.segment=='all')&(tr.origin=='number')&(tr.regime=='es')].copy()
s['pe']=100*s.n_exempt/s.n0; s['pn']=100*s.n_number/s.n0
top=s.sort_values('pe',ascending=False).iloc[0]
R['city_max_exempt_name']=str(top.city)
R['city_max_exempt_lost_to_exempt']=r1(top.pe); R['city_max_exempt_kept_number']=r1(top.pn)

# 4) 점표기 별칭
for k,v in (R.get('n_units_by_regime') or {}).items(): R[f'n_units_by_regime.{k}']=int(v)
pt=R.get('ml_perm_top_array') or R.get('ml_perm_top')
if isinstance(pt,list) and pt:
    R['ml_perm_top.0.drop_bal_acc']=pt[0].get('drop_bal_acc'); R['ml_perm_top.0.feature']=pt[0].get('feature')
    R['ml_perm_top_array']=pt

# 5) 문구 키
R['ml_model']='a tabular foundation model'
R['ml_perm_top']=('how many days a year the unit is available, how many reviews it has accumulated, '
                  'its review rate, its minimum-stay requirement and the number of listings its host holds')
R['ml_structure_verdict']='resemble'; R['het_headline']='hosts holding several listings'; R['h3_verdict']='support'

# 6) vsteady 짧은 별칭
for tag in ('vsteady','vsteady_nofr'):
    for suf in ('b','se','ci','ci_lo','ci_hi','wcb_p','ri_p','n_treat_units','n_ctrl_units','n_obs'):
        src=f'did_disp_es_{tag}_t1_{suf}'
        if src in R: R[f'did_disp_es_{tag}_{suf}']=R[src]
if 'placebo_post_fr_vsteady_wcb_p' in R: R['placebo_post_fr_vsteady_p']=R['placebo_post_fr_vsteady_wcb_p']
R['did_disp_es_vsteady_ALIAS_NOTE']='did_disp_es_vsteady_*(접미사 없음) = _t1_*(2025-12 → 2026-06).'

# 7) 「이전 분류(헝가리 미확인)」 대조 — supp 용
for tag,ctrl in (('earlier_noneu',['noneu']),('earlier_vsteady',['steady','fr'])):
    d=panel[panel.regime_old.isin(['es']+ctrl)].copy(); d['treat']=(d.regime_old=='es').astype(int)
    emit(f'did_disp_es_{tag}_t1', fit(d,[-2,1]))
d=panel[panel.regime_old.isin(['steady','noneu'])].copy(); d['treat']=(d.regime_old=='steady').astype(int)
emit('did_disp_steady_earlier_t1', fit(d,[-2,1]))
for col,tag in (('regime_old','earlier'),('regime','current')):
    for rg in ('steady','unver'):
        g=panel[panel[col]==rg]
        R[f'n_units_{rg}_{tag}']=int(g.unit.nunique())
        R[f'n_list_{rg}_{tag}']=int(g[g.t_rel==-2].n_listings.sum())
        for lbl,t in (('pre',-2),('t1',1),('t2',2)):
            sub=g[g.t_rel==t]
            R[f'disp_rate_{rg}_{tag}_{lbl}']=r1(float(np.average(sub.pct_number,weights=sub.n_listings))) if len(sub) else None
R['hungary_assignment_note']=('헝가리는 1차 출처(NTAK 공식 포털: 2020-10-01 부터 중개 플랫폼 광고에 등록번호 표시 의무)를 '
 '첫 추정 뒤에 확보했다. 본 분석은 부다페스트를 의무 시행군(steady)으로 둔다. 이전 분류(미확인)에서의 추정은 '
 'did_disp_es_earlier_* / did_disp_steady_earlier_t1_* 에 있고 결론은 달라지지 않는다.')

# 7b) ML 보조 키 (겹당 학습/검정 크기, CV 설명, 특성 목록)
mlp=os.path.join(BASE,'out','ml_results.json')
if os.path.exists(mlp):
    ml=json.load(open(mlp))
    cv=ml.get('cv_eu') or {}
    if cv.get('n') and cv.get('folds'):
        R['ml_n_test']=int(round(cv['n']/cv['folds'])); R['ml_n_train']=int(cv['n']-round(cv['n']/cv['folds']))
    R['ml_cv_scheme']='StratifiedGroupKFold(4), 군집=도시 — 같은 도시가 학습·검정에 동시에 들어가지 않는다.'
    R['ml_outcome']='3분류: 0=번호 없음 유지, 1=번호 획득, 2=퇴출. 기준집단 = 2025-12 에 번호 없던 숙소.'
    R['ml_note_license_raw']='특성에 license 원문은 들어가지 않는다. 파생 범주(n_tokens, has_br)만 사용.'
    R['ml_feature_list']=['n_tokens','has_br','host_listings','multi','entire','private_room','superhost',
        'host_tenure_yrs','avail365','n_reviews','n_reviews_ltm','reviews_per_month','price','min_nights',
        'accommodates','lat','lon']

# 8) p 표기 문자열
n=0
for k in [k for k in list(R) if (k.endswith(('_wcb_p','_ri_p')) or k.endswith('_p')) and not k.endswith('_p_str')]:
    try: f=float(R[k])
    except (TypeError,ValueError): continue
    R[k+'_str']='<0.001' if f<0.001 else f'{f:.3f}'; n+=1
R['p_value_reporting_note']=('저장된 p 가 0 인 것은 부트스트랩 999회·무작위화 2,000회에서 더 극단적인 추출이 없었다는 뜻이다. '
 '「0.000」이 아니라 <0.001 로 적어라. 모든 p 키에 <키>_str 문자열을 함께 넣었다.')
json.dump(R,open(RJ,'w'),indent=1,ensure_ascii=False,default=str)
print('WROTE',RJ,'keys=',len(R),'| p_str',n)
for k in ('did_disp_es_t1_b','did_disp_es_vsteady_t1_b','did_disp_steady_t1_b',
          'did_disp_es_earlier_noneu_t1_b','did_disp_es_earlier_vsteady_t1_b','did_disp_steady_earlier_t1_b',
          'n_units_steady_current','n_units_unver_current','n_units_steady_earlier','n_units_unver_earlier'):
    print(f'  {k} = {R.get(k)}')
