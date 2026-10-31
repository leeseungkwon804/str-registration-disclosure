#!/usr/bin/env python3
"""Supplementary tables S1-S7."""
import os, json, csv
import numpy as np, pandas as pd
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUPP=os.path.join(BASE,'supp'); os.makedirs(os.path.join(SUPP,'figs'),exist_ok=True)
R=json.load(open(os.path.join(BASE,'out','results.json')))
L=[]
def fp(v):
    if v is None or (isinstance(v,float) and v!=v): return '—'
    return '<0.001' if float(v)<0.001 else f'{float(v):.3f}'

def f1(v):
    return '—' if v is None else f'{float(v):.1f}'

def tbl(cap, df, note=None):
    df=df.copy()
    for c in df.columns:
        if 'p' == str(c).split()[-1] or str(c).endswith(' p'):
            df[c]=df[c].map(fp)
        if '(pp)' in str(c) or 'Difference' in str(c) or 'Estimate' in str(c):
            df[c]=df[c].map(lambda v: v if isinstance(v,str) else f1(v))
    for c in df.columns:
        if 'Listings' in str(c) or str(c)=='Markets':
            df[c]=df[c].map(lambda v: v if isinstance(v,str) else
                            ('—' if v is None or (isinstance(v,float) and v!=v) else f'{int(v):,}'))
    df=df.where(pd.notna(df),'—')
    L.append(f'TABLE: {cap}')
    L.append(df.to_markdown(index=False, disable_numparse=True))
    if note: L.append(f'NOTE: {note}')
    L.append('')

L.append('TITLE: Supplementary Materials')
L.append('')
L.append('## S1. Status of a Registration-Number Display Duty, by Country')
L.append('')
rows=[
 ['France','In force; extended to all municipalities by May 2026','Previously only municipalities operating a change-of-use procedure. Paris has required a number since 1 December 2017','Ministry practical guide, 1 September 2025; Code du tourisme Arts. L. 324-1-1 III and L. 324-2; City of Paris declaration page; Pays Basque agglomeration procedure page','30 October 2026'],
 ['Spain','In force from 1 July 2025; provisions establishing the single registration procedure annulled May-June 2026','Royal Decree 1312/2024, Art. 6 on platform duties, effects from 1 July 2025. Judgments of 19 May, 21 May and 1 June 2026 annul Arts. 5, 8, 9, 10 and 12(b),(c) in full and Arts. 1, 2 and 6 only in so far as they refer to the single registration procedure and the registration number','State gazette: the royal decree and the three judgments','30 October 2026'],
 ['Italy','In force','National identification code; the provisions apply from the sixtieth day after the ministerial notice of 3 September 2024','Official gazette, Part Two No. 103, 3 September 2024; national accommodation database portal','30 October 2026'],
 ['Greece','In force before 2022','Registry number must be displayed visibly in platform listings','Tax authority document on core obligations for short-term rental through digital platforms','30 October 2026'],
 ['Portugal','In force before 2026','Registration number required in advertising and commercial documentation','National tourism authority technical guide, January 2025','30 October 2026'],
 ['Belgium','In force, regionally','Brussels-Capital Region under the ordinance of 8 May 2014 and its implementing decree of 24 March 2016; Flanders under the accommodation decree','Regional government guidance for both regions','30 October 2026'],
 ['The Netherlands','In force in all three sampled cities','Amsterdam from 1 April 2021; The Hague from 2 August 2021, where a registration number is required for every form of tourist rental and must be stated in an advertisement; Rotterdam under a municipal ordinance introducing the registration duty under Article 23a of the Housing Act 2014','City of Amsterdam holiday-rental page and municipal housing regulation 2024; City of The Hague permit page; Rotterdam municipal ordinance as published in the government gazette','30 October 2026'],
 ['Austria','Not in force','The federal ministry states that the Regulation will initially not be implemented in any federal province as of 20 May 2026, that there is no registration requirement, and that no registration number need be indicated in platform listings','Federal ministry page on the short-term rental Regulation','30 October 2026'],
 ['Ireland','Not in force','Registration is not yet open and will open once the enabling legislation has been passed','National tourism development authority register page','30 October 2026'],
 ['Hungary','In force from 1 October 2020','The registration number and the accommodation type must be shown in listings of accommodation offered through intermediary platforms','Official portal of the national tourism information system','30 October 2026'],
 ['Germany, Czechia, Denmark, Latvia, Malta, Sweden','Not established','No primary source establishing the status of a display duty as of May 2026 could be located. A European Parliament analysis notes that no public information exists on how many Member States were implementing the Regulation. Display is observed at 0.0 percent in the Czech, Danish, Latvian, Maltese and Swedish markets at every snapshot, but that is the outcome variable of this study and is therefore not used to classify them','European Parliament in-depth analysis PE 759.356, September 2025','30 October 2026'],
]
tbl('Table S1. Status of a duty to display a short-term rental registration number in platform listings, by country, as of May 2026, with the primary source consulted and the date of consultation.',
    pd.DataFrame(rows,columns=['Country','Status in May 2026','Detail','Primary source','Accessed']),
    'Classification rests on primary legal and government sources. Where no primary source could be located the status is recorded as not established rather than inferred. Markets in the last row are held out of the main estimates and enter only as a robustness check.')

L.append('## S2. Snapshot Coverage and the Exclusion of the March 2026 Wave')
L.append('')
cov=pd.read_csv(os.path.join(BASE,'out','coverage_diagnostics.csv'))
cov=cov.rename(columns={'unit':'Market','t-2':'2025-12','t-1':'2026-03','t+1':'2026-06','t+2':'2026-09',
                        'ratio_tm1':'2026-03 / mean(2025-12, 2026-06)','grp':'Regime'})
keep=[c for c in ['Market','Regime','2025-12','2026-03','2026-06','2026-09','2026-03 / mean(2025-12, 2026-06)'] if c in cov.columns]
cov=cov[keep].copy()
for c in ['2025-12','2026-03','2026-06','2026-09']:
    if c in cov: cov[c]=cov[c].map(lambda v:'' if pd.isna(v) else f'{int(v):,}')
if '2026-03 / mean(2025-12, 2026-06)' in cov:
    cov['2026-03 / mean(2025-12, 2026-06)']=cov['2026-03 / mean(2025-12, 2026-06)'].map(
        lambda v:'' if pd.isna(v) else f'{v:.3f}')
cov=cov.sort_values('2026-03 / mean(2025-12, 2026-06)')
tbl('Table S2. Number of listings in each snapshot, by market, and the ratio of the March 2026 count to the mean of the adjacent waves.',
    cov,
    f"Across {R.get('mar_coverage_n_units')} markets the ratio averages {float(R['mar_coverage_ratio_mean']):.3f} "
    f"(median {float(R['mar_coverage_ratio_median']):.3f}, minimum {float(R['mar_coverage_ratio_min']):.3f}); "
    f"{R.get('mar_coverage_n_units_below_070')} markets fall below 0.70. The shortfall is not random with respect "
    "to the outcome: the weighted display share in the March wave exceeds the December wave by "
    f"{float(R['cov_disp_gap_steady']):.1f} percentage points among markets where the duty was already in force, so the "
    "March wave over-represents listings that display a number. The main estimates therefore use December 2025 "
    "as the pre-period.")

L.append('## S3. Classification of the Registration Field')
L.append('')
rules=[
 ['Preparation','HTML line breaks are replaced by a separator, remaining tags are stripped, HTML entities are resolved, non-breaking spaces are normalized, repeated whitespace is collapsed and the string is trimmed.'],
 ['Empty','Missing, zero length, punctuation and whitespace only, or an explicit placeholder for absence in any of the sampled languages.'],
 ['Exempt','Contains an exemption marker in any of the sampled languages. Evaluated before the pending category.'],
 ['Pending','Contains a marker for an application in progress in any of the sampled languages.'],
 ['Number present','Contains at least one token of three or more characters that includes at least two digits.'],
 ['Other text','None of the above. Reported separately; the maximum share in any market and snapshot is '+str(R.get('text_other_max'))+' percent.'],
 ['Priority','A single string is assigned to exactly one category in the order empty, exempt, pending, number present, other text. A field containing both an exemption marker and a number is therefore classified as exempt.'],
 ['Multiple numbers','The field is split on line breaks and on comma, semicolon, slash and vertical bar to count tokens. The count is retained as a derived variable; it does not change the category.'],
 ['Derived variables retained','Category, an indicator for a number being present, the token count, an indicator for a line break in the original field, and a length band.'],
]
tbl('Table S3. Rules used to classify the public registration field, fixed before any field values were inspected.',
    pd.DataFrame(rules,columns=['Step','Rule']),
    'The rules were written and frozen before classification and were applied without modification. No value of '
    'the field is reproduced here or anywhere in this study. The field can contain personal data, so only the '
    'derived variables in the last row were retained, and a mechanical audit confirmed that no field value '
    f"appears in any output: {R.get('priv_values')} distinct values drawn from {R.get('priv_units')} markets were "
    'searched for in every published table, figure input and intermediate file, at cell and token level, with no match.')

L.append('## S4. Estimates Including the Incomplete March 2026 Wave')
L.append('')
rows=[]
for rg,nm in (('es','Spain, procedure annulled'),('fr','France, already in force'),
              ('steady','EU, duty in force'),('unver','EU, status not established')):
    rows.append([nm, R.get(f'did_disp_{rg}_t1_b'), R.get(f'did_disp_{rg}_t1_wcb_p'),
                 R.get('did_disp_es_t1_b_march') if rg=='es' else '', ''])
d4=pd.DataFrame(rows,columns=['Treated group','Main estimate, December 2025 reference (pp)','Wild-boot p',
                              'Estimate with March 2026 as reference (pp)','Note'])
d4=d4.fillna('')
tbl('Table S4. The Spanish estimate under the two possible pre-periods.',
    d4,
    'The main estimate uses December 2025 as the reference. Using the incomplete March 2026 wave as the reference '
    f"instead gives {R.get('did_disp_es_t1_b_march')} percentage points for Spain. Both are reported because the "
    'choice of pre-period is the single discretionary decision in the design; the direction and significance of '
    'the Spanish estimate do not depend on it.')

L.append('## S5. Control Group of European Markets Where the Duty Remained in Force')
L.append('')
rows=[]
for tag,nm in (('vsteady','EU markets where the duty stayed in force, France included'),
               ('vsteady_nofr','EU markets where the duty stayed in force, France excluded')):
    for per,pl in (('t1','To 2026-06'),('t2','To 2026-09'),('pooled','Both post-period waves')):
        rows.append([nm,pl,R.get(f'did_disp_es_{tag}_{per}_b'),R.get(f'did_disp_es_{tag}_{per}_ci'),
                     R.get(f'did_disp_es_{tag}_{per}_wcb_p'),R.get(f'did_disp_es_{tag}_{per}_ri_p'),
                     R.get(f'did_disp_es_{tag}_{per}_n_treat_units'),R.get(f'did_disp_es_{tag}_{per}_n_ctrl_units')])
d5=pd.DataFrame(rows,columns=['Control group','Horizon','Estimate (pp)','95% CI','Wild-boot p','Randomization p','Treated markets','Control markets']).fillna('')
tbl('Table S5. The Spanish estimate against European markets where the display duty remained in force.',
    d5,
    'The comparison holds the continent and the Regulation constant and removes the drift between European and '
    'non-European markets that Section 4.2 of the article reports. Under this control the placebo that compares '
    f"the French markets across the two post-period waves returns {R.get('placebo_post_fr_vsteady_b')} percentage "
    f"points with a wild-bootstrap p of {fp(R.get('placebo_post_fr_vsteady_wcb_p'))}, against "
    f"{f1(R.get('placebo_post_fr_b'))} and a p of {fp(R.get('placebo_post_fr_p'))} when markets outside Europe are the control, "
    'so the residual trend is a property of that control group rather than of the European markets.')
rows=[['Switched to exempt among listings that showed a number',R.get('did_exempt_es_vsteady_b'),R.get('did_exempt_es_vsteady_wcb_p')],
      ['No longer shows a number, among surviving listings that showed one',R.get('did_lose_es_vsteady_b'),R.get('did_lose_es_vsteady_wcb_p')],
      ['Acquired a number, among listings with none',R.get('did_acquire_es_vsteady_b'),R.get('did_acquire_es_vsteady_wcb_p')],
      ['No longer present, among listings with none',R.get('did_exit_es_vsteady_b'),R.get('did_exit_es_vsteady_wcb_p')]]
tbl('Table S5b. Listing-level transitions in Spain against European markets where the duty remained in force.',
    pd.DataFrame(rows,columns=['Transition','Difference (pp)','Wild-boot p']).fillna(''),
    'Market-level differences with clustering on the market, December 2025 to the first wave after the '
    'application date.')

L.append('## S6. Model Used for the Heterogeneity Analysis')
L.append('')
rows=[['Model',str(R.get('ml_model'))+' (version '+str(R.get('ml_tabpfn_version'))+'), run locally on the published weights'],
 ['Hardware',str(R.get('ml_gpu'))+'; '+str(R.get('ml_torch'))],
 ['Outcome','Three categories for listings with no number in December 2025: unchanged, acquired a number, no longer present'],
 ['Cross-validation','Stratified four-fold cross-validation with folds grouped by market, so that no market appears in both the training and the held-out part of a fold'],
 ['Training and test size per fold',f"{int(R['ml_n_train']):,} and {int(R['ml_n_test']):,}"],
 ['Features','Token count and line-break indicator from the registration field, host listing count, multi-listing indicator, entire-home and private-room indicators, superhost indicator, host tenure, days available, review counts and review rate, nightly price, minimum nights, guests accommodated, latitude and longitude'],
 ['Fit time','Between 0.8 and 1.7 seconds for training sets from 1,000 to 16,000 rows'],
 ['AUC, one-vs-rest macro',f"Spain {float(R['ml_auc_es']):.3f}; EU duty in force {float(R['ml_auc_steady']):.3f}; EU overall {float(R['ml_auc_eu']):.3f}; outside Europe {float(R['ml_auc_noneu']):.3f}"],
 ['Balanced accuracy',f"Spain {float(R['ml_balacc_es']):.3f}; EU overall {float(R['ml_balacc_eu']):.3f}; chance is 0.333 over three categories"],
 ['Excluded by design','No value of the registration field enters the features; only the derived token count and line-break indicator do']]
tbl('Table S6. Configuration of the model reported in Section 4.4 of the article.',
    pd.DataFrame(rows,columns=['Item','Setting']),
    'Permutation importance is the loss in balanced accuracy when a single feature is permuted in the held-out '
    'fold, averaged over three permutations. The ranking of features is similar inside and outside the European '
    'Union, so the model supports no claim that the response follows a structure specific to the Union.')

L.append('## S7. Assignment of the Hungarian Market')
L.append('')
rows=[]
for tag,nm in (('current','Budapest assigned to the established-duty group (reported)'),
               ('earlier','Budapest assigned to status not established (earlier)')):
    for rg,rn in (('steady','Duty in force'),('unver','Status not established')):
        rows.append([nm,rn,R.get(f'n_units_{rg}_{tag}'),R.get(f'n_list_{rg}_{tag}'),
                     R.get(f'disp_rate_{rg}_{tag}_pre'),R.get(f'disp_rate_{rg}_{tag}_t1'),
                     R.get(f'disp_rate_{rg}_{tag}_t2')])
tbl('Table S7. Composition of the two European groups under each assignment of the Hungarian market.',
    pd.DataFrame(rows,columns=['Assignment','Group','Markets','Listings 2025-12',
                               'Display 2025-12 (%)','Display 2026-06 (%)','Display 2026-09 (%)']).fillna('—'),
    'A primary source for Hungary was located after the first round of estimation. The reported analysis assigns '
    'Budapest to the group whose duty remained in force; the earlier assignment placed it with the markets whose '
    'status could not be established. Only the composition of the two European groups changes.')
rows=[['Spain against markets outside Europe',R.get('did_disp_es_t1_b'),R.get('did_disp_es_t1_wcb_p'),
       R.get('did_disp_es_earlier_noneu_t1_b'),R.get('did_disp_es_earlier_noneu_t1_wcb_p')],
      ['Spain against EU markets where the duty stayed in force',R.get('did_disp_es_vsteady_t1_b'),
       R.get('did_disp_es_vsteady_t1_wcb_p'),R.get('did_disp_es_earlier_vsteady_t1_b'),
       R.get('did_disp_es_earlier_vsteady_t1_wcb_p')],
      ['EU duty in force against markets outside Europe',R.get('did_disp_steady_t1_b'),
       R.get('did_disp_steady_t1_wcb_p'),R.get('did_disp_steady_earlier_t1_b'),
       R.get('did_disp_steady_earlier_t1_wcb_p')]]
tbl('Table S7b. Estimates under each assignment of the Hungarian market.',
    pd.DataFrame(rows,columns=['Estimate','Reported (pp)','Wild-boot p','Earlier assignment (pp)','Wild-boot p ']).fillna('—'),
    'December 2025 to the first wave after the application date. The assignment of one market does not change any '
    'estimate to one decimal place, so the conclusions do not depend on it.')

L.append('FIGURE: FigureS1 | Figure S1. Share of listings displaying a registration number in each market, in the December 2025 wave against the first wave after the application date, by regulatory status. Marker area scales with the number of listings in the earlier wave.')
L.append('')
import re as _re
_txt='\n'.join(L)+'\n'
_txt=_re.sub(r'(?<![\w/])-(?=\d)', '\u2212', _txt)   # 음수 기호만 치환
_han=[l for l in _txt.split('\n') if any('\uac00'<=c<='\ud7a3' for c in l)]
assert not _han, '보충자료는 영문 문서다. 한국어가 섞인 줄: '+repr(_han[:3])
open(os.path.join(SUPP,'Supplementary_Materials.md'),'w',encoding='utf-8').write(_txt)
print('WROTE supp/Supplementary_Materials.md')
