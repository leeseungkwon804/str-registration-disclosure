#!/usr/bin/env python3
"""Tables of the article (out/table*.md, .csv) from out/results.json."""
import os, json
import numpy as np, pandas as pd
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R=json.load(open(os.path.join(BASE,'out','results.json')))
OUT=os.path.join(BASE,'out')
LBL={'es':'EU · single registration procedure annulled (ES)',
     'fr':'EU · already in force (FR)',
     'steady':'EU · already in force (IT, GR, PT, BE, NL, HU)',
     'noscheme':'EU · no registration scheme (AT, IE)',
     'unconf':'EU · national status unverified',
     'eea':'EEA non-EU (NO, CH)',
     'noneu':'Non-EU control (UK, US, CA, AU, NZ)'}
ORD=['es','fr','steady','noscheme','unconf','eea','noneu']

def fmt_p(v):
    if v is None or (isinstance(v,float) and v!=v): return '—'
    return '<0.001' if float(v)<0.001 else f'{float(v):.3f}'

def md(df,path,title,note):
    df=df.copy()
    for c in df.columns:
        if 'p' == c.split()[-1].strip('()') or c.startswith(('Wild-boot','Randomization')):
            df[c]=df[c].map(fmt_p)
    def _f1(v):
        if v is None or (isinstance(v,float) and v!=v): return '—'
        if isinstance(v,str): return v
        return f'{float(v):.1f}'
    for c in df.columns:
        if any(t in str(c) for t in ('(%)','(pp)','Difference','Estimate','Cluster SE')):
            df[c]=df[c].map(_f1).astype('object')
    for c in ('Treated cities','Control cities','Cities','Listings 2025-12','Listings 2026-06','Listings'):
        if c in df.columns:
            df[c]=df[c].map(lambda v:'—' if v is None or (isinstance(v,float) and v!=v) else f'{int(v):,}')
    df=df.where(pd.notna(df),'—')
    df=df.replace({None:'—'})
    with open(path,'w',encoding='utf-8') as f:
        f.write(f'**{title}**\n\n'); f.write(df.to_markdown(index=False, disable_numparse=True)); f.write(f'\n\n{note}\n')
    df.to_csv(path.replace('.md','.csv'),index=False)
    print('WROTE',path)

# ---- 표1: 표본과 표시율
rows=[]
for g in ORD:
    if R.get(f'disp_rate_{g}_pre') is None: continue
    rows.append({'Regulatory status':LBL[g],
      'Cities':R['n_units_by_regime'].get(g),
      'Listings 2025-12':R.get(f'n_listings_{g}_pre'),
      'Listings 2026-06':R.get(f'n_listings_{g}_post'),
      'Disclosure 2025-12 (%)':R.get(f'disp_rate_{g}_pre'),
      'Disclosure 2026-06 (%)':R.get(f'disp_rate_{g}_post'),
      'Disclosure 2026-09 (%)':R.get(f'disp_rate_{g}_post2'),
      'Change to 2026-06 (pp)':R.get(f'disp_change_{g}'),
      'Change to 2026-09 (pp)':R.get(f'disp_change_{g}_to_post2'),
      'Exempt 2025-12 (%)':R.get(f'exempt_rate_{g}_pre'),
      'Exempt 2026-06 (%)':R.get(f'exempt_rate_{g}_post')})
md(pd.DataFrame(rows),os.path.join(OUT,'table1_sample_and_rates.md'),
   'Table 1. Sample and registration-number disclosure, by national regulatory status',
   'Shares are weighted by the number of listings in the snapshot. The 2026-03 snapshot is '
   'systematically incomplete and is excluded (see Table 3). Source: Inside Airbnb; authors’ calculations.')

# ---- 표2: 주 추정
rows=[]
for g,nm in (('es','Spain — procedure annulled'),('fr','France — already in force'),
             ('steady','EU already in force'),('unconf','EU status unverified')):
    b=R.get(f'did_disp_{g}_b')
    if b is None: continue
    rows.append({'Treated group':nm,
      'DiD 2025-12 → 2026-06 (pp)':round(b,2),
      'Cluster SE':round(R.get(f'did_disp_{g}_se'),2) if R.get(f'did_disp_{g}_se') is not None else None,
      '95% CI':f"{R.get(f'did_disp_{g}_ci_lo'):.1f} to {R.get(f'did_disp_{g}_ci_hi'):.1f}",
      'Wild-boot p':R.get(f'did_disp_{g}_wcb_p'),
      'Randomization p':R.get(f'did_disp_{g}_ri_p'),
      'DiD incl. 2026-09 (pp)':round(R.get(f'did_disp_{g}_t2_b'),2) if R.get(f'did_disp_{g}_t2_b') is not None else None,
      'Wild-boot p (incl. 09)':R.get(f'did_disp_{g}_t2_wcb_p'),
      'Treated cities':R.get(f'did_disp_{g}_n_treat_units'),
      'Control cities':R.get(f'did_disp_{g}_n_ctrl_units')})
for nm,bk,pk in (('PLACEBO: EU no-scheme (AT, IE) vs non-EU','placebo_noscheme_b','placebo_noscheme_p'),
                 ('PLACEBO: non-EU split, fake date','placebo_fakedate_noneu_b','placebo_fakedate_noneu_p'),
                 ('PLACEBO: Spain, both periods post (06→09)','placebo_post_es_b','placebo_post_es_p'),
                 ('PLACEBO: France, both periods post (06→09)','placebo_post_fr_b','placebo_post_fr_p')):
    if R.get(bk) is None: continue
    rows.append({'Treated group':nm,'DiD 2025-12 → 2026-06 (pp)':round(R[bk],2),'Cluster SE':None,
                 '95% CI':None,'Wild-boot p':R.get(pk),'Randomization p':None,
                 'DiD incl. 2026-09 (pp)':None,'Wild-boot p (incl. 09)':None,
                 'Treated cities':None,'Control cities':None})
md(pd.DataFrame(rows),os.path.join(OUT,'table2_did.md'),
   'Table 2. Difference-in-differences in disclosure, relative to the non-EU control',
   'Outcome: share of listings displaying a registration number, city × snapshot, weighted equally across cities. '
   'All specifications include city and snapshot fixed effects; standard errors are clustered on the city. '
   f"Inference is by wild cluster bootstrap ({R.get('nboot','999')} replications, Rademacher weights, null imposed) "
   'and by city-level randomization inference (2,000 re-assignments). Placebo rows report the point estimate and '
   'wild-bootstrap p-value only. Two placebo rows are not valid falsification tests and are reported for '
   'transparency: Austria and Ireland display a registration number on 0.00% of listings in every wave, so that '
   'group cannot fall (a floor, not a test); and for Spain both post-period waves follow the judgments of May-June '
   '2026, published in the BOE in July 2026, so the treatment is still unfolding across them. The significant '
   'France post-period row indicates a residual differential trend between the EU and the non-EU control, which '
   'the small estimates should be read against.')

# ---- 표3: 자료 품질
LB={'t-2':'2025-12','t-1':'2026-03','t+1':'2026-06','t+2':'2026-09'}
rows=[{'Snapshot':LB.get(k,k),'Listings':v} for k,v in R['listings_by_period'].items()]
d3=pd.DataFrame(rows)
d3['Weighted disclosure (%)']=[R.get(f'disp_rate_all_{s}') for s in ['x']*len(d3)]
d3=d3.drop(columns=['Weighted disclosure (%)'])
d3['Role in the analysis']=['Pre-period (used)','INCOMPLETE — excluded from main spec','Post-period (used)','Post-period (used)'][:len(d3)]
md(d3,os.path.join(OUT,'table3_coverage.md'),
   'Table 3. Snapshot coverage and the exclusion of the 2026-03 wave',
   f"Across {R.get('mar_coverage_n_units')} units the 2026-03 listing count equals on average "
   f"{R.get('mar_coverage_ratio_mean')} (median {R.get('mar_coverage_ratio_median')}, minimum "
   f"{R.get('mar_coverage_ratio_min')}) of the mean of the adjacent 2025-12 and 2026-06 waves; "
   f"{R.get('mar_coverage_n_units_below_070')} units fall below 0.70. Because the shortfall is not random with "
   'respect to disclosure — the 2026-03 wave over-represents listings that display a number — the main '
   'specification uses 2025-12 as the pre-period.')
