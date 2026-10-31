#!/usr/bin/env python3
"""Figures 1-4."""
import os, json, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(BASE,'figs')
plt.rcParams.update({
 'font.family':'sans-serif','font.sans-serif':['Arial','Liberation Sans','DejaVu Sans'],
 'font.size':8,'axes.labelsize':8.5,'axes.titlesize':9,'xtick.labelsize':8,
 'ytick.labelsize':8,'legend.fontsize':7.5,'axes.linewidth':0.6,
 'xtick.major.width':0.6,'ytick.major.width':0.6,'figure.dpi':200,
 'savefig.dpi':400,'axes.spines.top':False,'axes.spines.right':False,
 'axes.grid':True,'grid.linewidth':0.4,'grid.alpha':0.35,'grid.color':'#b8b7b0'})

# 검증 통과 팔레트 (dataviz validate_palette.js, light, ALL CHECKS PASS)
C = {'fr':'#2a78d6','steady':'#1baf7a','es':'#eb6834','noscheme':'#6250d6',
     'eea':'#e34948','noneu':'#52514e','unconf':'#e87ba4','unver':'#e87ba4'}
MK = {'fr':'o','steady':'s','es':'D','noscheme':'^','eea':'v','noneu':'X','unconf':'P','unver':'P'}
LBL = {'fr':'France · already in force','steady':'EU · in force (IT GR PT BE NL HU)',
       'unconf':'EU · status unverified',
       'es':'Spain · procedure annulled','noscheme':'EU · no scheme (AT, IE)',
       'eea':'EEA non-EU (NO, CH)','noneu':'Non-EU control','unconf':'EU · status unverified'}
# 색 순서 = dataviz validate_palette.js 통과 배열(인접쌍 기준)
ORDER = ['es','fr','steady','eea','noscheme','unconf','noneu']
# fig3 는 grp 원래 이름을 쓰므로 매핑을 둔다
GRP2REG = {'EU_aligned':'fr','EU_reversal':'es','EU_pretreated':'steady',
           'EU_noscheme':'noscheme','nonEU':'noneu','EEA_nonEU':'eea','EU_unconfirmed':'unconf'}
INK, INK2 = '#0b0b0b', '#52514e'
FEATLBL = {'avail365':'Days available (next 365)','n_reviews':'Number of reviews',
 'reviews_per_month':'Reviews per month','min_nights':'Minimum nights',
 'host_listings':"Host's listing count",'multi':'Multi-listing host','lon':'Longitude',
 'lat':'Latitude','superhost':'Superhost','host_tenure_yrs':'Host tenure (years)',
 'price':'Nightly price','accommodates':'Guests accommodated','entire':'Entire home',
 'private_room':'Private room','n_reviews_ltm':'Reviews, last 12 months',
 'n_tokens':'Tokens in registration field','has_br':'Line break in registration field'}

def save(fig,stem):
    for ext in ('png','pdf'):
        fig.savefig(os.path.join(FIGS,f'{stem}.{ext}'),bbox_inches='tight',facecolor='white')
    n={'fig1_disclosure_rates':'Figure1','fig2_event_study':'Figure2',
       'fig3_transitions':'Figure3','fig4_ml_heterogeneity':'Figure4'}.get(stem)
    if n:
        for ext in ('png','pdf'):
            fig.savefig(os.path.join(FIGS,f'{n}.{ext}'),bbox_inches='tight',facecolor='white')

def R():
    return json.load(open(os.path.join(BASE,'out','results.json')))

# ---------- 그림 1 ----------
def fig1():
    r = R(); panel = pd.read_csv(os.path.join(BASE,'out','city_period_panel.csv'))
    fig, axes = plt.subplots(1,2, figsize=(7.1,3.3))
    ax = axes[0]
    SUF = {-2:'pre', -1:'mar', 1:'post', 2:'post2'}
    groups = [g for g in ORDER if r.get(f'disp_rate_{g}_pre') is not None]
    for i,g in enumerate(groups):
        xs = [-2,-1,1,2]
        ys = [r.get(f'disp_rate_{g}_{SUF[t]}', np.nan) for t in xs]
        ys = [np.nan if v is None else v for v in ys]
        # 2026-03(t=-1) 은 미완 → 속이 빈 마커 + 점선 연결
        ax.plot([-2,-1], ys[:2], color=C[g], lw=1.2, ls=(0,(3,2)), zorder=3, clip_on=False)
        ax.plot([-1,1],  ys[1:3], color=C[g], lw=1.2, ls=(0,(3,2)), zorder=3, clip_on=False)
        ax.plot([-2]+xs[2:], [ys[0]]+ys[2:], marker=MK[g], color=C[g], lw=1.8, ms=5.5,
                mec='white', mew=0.8, label=LBL[g], zorder=4, clip_on=False)
        ax.plot([-1], [ys[1]], marker=MK[g], color='white', mec=C[g], mew=1.3, ms=5.5,
                lw=0, zorder=4, clip_on=False)
        if ys[-1]==ys[-1]:
            ax.annotate(f'{ys[-1]:.1f}', (xs[-1], ys[-1]), textcoords='offset points',
                        xytext=(6,0), va='center', fontsize=7, color=C[g], weight='bold')
    ax.axvline(0, color=INK2, lw=0.9, ls=(0,(4,2)), zorder=1)
    ax.text(0.12, 46, 'Reg. 2024/1028\napplies\n20 May 2026', ha='center', va='center',
            fontsize=6.6, color=INK2, linespacing=1.3,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#b8b7b0', lw=0.5))
    # 0% 에 겹쳐 있는 두 군을 글로 드러낸다
    ax.annotate('AT, IE (no scheme) and NO, CH: 0.0% at every snapshot',
                xy=(-2,0), xytext=(-1.9,7.5), fontsize=6.4, color=INK2,
                arrowprops=dict(arrowstyle='-', color=INK2, lw=0.6,
                                shrinkA=0, shrinkB=2))
    ax.set_xticks([-2,-1,1,2])
    ax.set_xticklabels(['2025-12','2026-03\n(partial)','2026-06','2026-09'],fontsize=7.4)
    ax.set_xlim(-2.35,2.55); ax.set_ylim(-5,97)
    ax.set_xlabel('Snapshot relative to application date'); ax.set_ylabel('Listings showing a registration number (%)')
    ax.set_title('(a) By regulatory status of the country', loc='left', color=INK)

    ax2 = axes[1]
    panel['regime'] = panel['grp'].map(GRP2REG)
    pre = panel[panel.t_rel==-2].set_index('unit'); post = panel[panel.t_rel==1].set_index('unit')
    common = pre.index.intersection(post.index)
    ax2.plot([0,100],[0,100], color=INK2, lw=0.8, ls=(0,(4,2)), zorder=1)
    for g in groups:
        u=[x for x in common if pre.loc[x,'regime']==g]
        if not u: continue
        ax2.scatter(pre.loc[u,'pct_number'], post.loc[u,'pct_number'], s=np.clip(pre.loc[u,'n_listings']/260,10,150),
                    facecolor=C[g], edgecolor='white', linewidth=0.7, marker=MK[g],
                    alpha=0.9, zorder=3, label=LBL[g])
    ax2.set_xlim(-4,104); ax2.set_ylim(-4,104)
    ax2.set_xlabel('Before — 2025-12 (%)'); ax2.set_ylabel('After — 2026-06 (%)')
    ax2.set_title('(b) Each city (marker area scales with listings)', loc='left', color=INK)
    ax2.text(58,6,'below the line =\ndisclosure fell', fontsize=6.8, color=INK2, linespacing=1.25)
    h=[Line2D([0],[0],marker=MK[g],color=C[g],lw=1.6,ms=5,mec='white',mew=0.7,label=LBL[g]) for g in groups]
    fig.legend(handles=h, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5,-0.02),
               handletextpad=0.5, columnspacing=1.3)
    fig.tight_layout(rect=[0,0.12,1,0.97])
    save(fig,'fig1_disclosure_rates')
    plt.close(fig); print('fig1 ok')

# ---------- 그림 2 ----------
def fig2():
    r=R()
    specs=[('es','Spain · procedure annulled'),('fr','France · already in force'),
           ('steady','EU · in force'),('unconf','EU · status unverified')]
    specs=[(g,t) for g,t in specs if r.get('es_disp_%s_p1'%g) is not None]
    fig,axes=plt.subplots(1,len(specs),figsize=(2.15*len(specs)+0.5,3.5),sharey=True)
    if len(specs)==1: axes=[axes]
    for ax,(g,ttl) in zip(axes,specs):
        xs=[-2];ys=[0.0];lo=[0.0];hi=[0.0]
        for rel,t in (('m1',-1),('p1',1),('p2',2)):
            b=r.get(f'es_disp_{g}_{rel}'); se=r.get(f'es_disp_{g}_{rel}_se')
            if b is None: continue
            xs.append(t); ys.append(b)
            lo.append(b-1.96*(se or 0)); hi.append(b+1.96*(se or 0))
        ax.axhline(0,color=INK2,lw=0.8)
        ax.axvline(0,color=INK2,lw=0.9,ls=(0,(4,2)))
        for x,l,h2 in zip(xs,lo,hi):
            if x!=-2: ax.vlines(x,l,h2,color=C[g],lw=1.6,zorder=2)
        solid=[(x,y) for x,y in zip(xs,ys) if x!=-1]
        ax.plot([x for x,_ in solid],[y for _,y in solid],marker=MK[g],color=C[g],
                lw=0,ms=6,mec='white',mew=0.9,zorder=3)
        hollow=[(x,y) for x,y in zip(xs,ys) if x==-1]
        if hollow:
            ax.plot([hollow[0][0]],[hollow[0][1]],marker=MK[g],color='white',
                    mec=C[g],mew=1.3,ms=6,lw=0,zorder=3)
        ax.annotate('ref',(-2,0),textcoords='offset points',xytext=(0,-14),ha='center',
                    fontsize=6.4,color=INK2)
        ax.set_xticks([-2,-1,1,2])
        ax.set_xticklabels(['2025-12','2026-03\n(partial)','2026-06','2026-09'],
                           fontsize=6.8, rotation=40, ha='right',
                           rotation_mode='anchor')
        ax.set_xlim(-2.7,3.0); ax.set_title(ttl,loc='left',fontsize=7.8,color=INK)
    axes[0].set_ylabel('Diff-in-diff vs non-EU control\n(percentage points)')
    fig.tight_layout()
    save(fig,'fig2_event_study')
    plt.close(fig); print('fig2 ok')

# ---------- 그림 3 ----------
def fig3():
    """핵심 기제: 번호를 갖고 있던 숙소가 어디로 갔나 + 번호 없던 숙소의 전이."""
    import pandas as pd
    tr=pd.read_csv(os.path.join(BASE,'out','transitions_final.csv'))
    tr['regime']=tr.grp.map(GRP2REG)
    order=['es','fr','steady','noscheme','noneu']
    def agg(seg,col):
        out={}
        for rg,g in tr[(tr.segment==seg)&(tr.pair=='main_clean')].groupby('regime'):
            wt=g.n_blank_t0.to_numpy(float)
            if wt.sum()==0: continue
            out[rg]=float(np.average(g[col],weights=wt))
        return out
    fig,axes=plt.subplots(1,2,figsize=(7.4,4.3),sharey=True)
    # (a) 번호 있던 숙소
    ax=axes[0]
    parts=[('pct_acquire','Still shows a number','#1baf7a','' ),
           ('pct_exempt','Switched to “exempt”','#eb6834','' ),
           ('pct_pending','Marked as pending','#eda100','////'),
           ('pct_same','Became blank','#a8a7a0',''),
           ('pct_gone','Delisted','#52514e','....')]
    have_a=agg('had_number','pct_acquire'); have_b=agg('all','pct_acquire')
    rgs=[r for r in order if (r in have_a) or (r in have_b)]   # 두 패널 행을 맞춘다
    ys=np.arange(len(rgs)); left=np.zeros(len(rgs))
    for col,lab,c,h in parts:
        v=agg('had_number',col); vals=np.array([v.get(r,0) for r in rgs])
        ax.barh(ys,vals,left=left,color=c,edgecolor='white',linewidth=1.1,hatch=h,label=lab,zorder=3)
        for y,x0,x1 in zip(ys,left,vals):
            if col=='pct_gone':
                if rgs[int(y)] in have_a:
                    ax.text(101.5,y,f'{x1:.0f}',ha='left',va='center',fontsize=6.8,color=INK2,weight='bold')
                else:
                    ax.text(2,y,'no listings showed a number',ha='left',va='center',
                            fontsize=6.4,color=INK2,style='italic')
            elif x1>=5.5:
                ax.text(x0+x1/2,y,f'{x1:.0f}',ha='center',va='center',fontsize=6.8,
                        color='white' if c=='#eb6834' else INK,weight='bold')
        left=left+vals
    ax.set_yticks(ys); ax.set_yticklabels([LBL[r] for r in rgs]); ax.invert_yaxis()
    ax.set_xlim(0,112); ax.set_xticks([0,20,40,60,80,100])
    ax.set_xlabel('% of listings that showed a number in 2025-12')
    ax.set_title('(a) Listings that already showed a number',loc='left',color=INK)
    ax.grid(axis='y',visible=False)
    ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.18),ncol=2,frameon=False,
              fontsize=7,handlelength=1.4,handletextpad=0.5,columnspacing=1.1)
    # (b) 번호 없던 숙소
    ax2=axes[1]
    parts2=[('pct_acquire','Obtained a number','#2a78d6',''),
            ('pct_exempt','Marked as “exempt”','#eb6834',''),
            ('pct_pending','Marked as pending','#eda100','////'),
            ('pct_same','Still no number','#a8a7a0',''),
            ('pct_gone','Delisted','#52514e','....')]
    rgs2=rgs
    ys=np.arange(len(rgs2)); left=np.zeros(len(rgs2))
    for col,lab,c,h in parts2:
        v=agg('all',col); vals=np.array([v.get(r,0) for r in rgs2])
        ax2.barh(ys,vals,left=left,color=c,edgecolor='white',linewidth=1.1,hatch=h,label=lab,zorder=3)
        for y,x0,x1 in zip(ys,left,vals):
            if col=='pct_gone':
                ax2.text(101.5,y,f'{x1:.0f}',ha='left',va='center',fontsize=6.8,color=INK2,weight='bold')
            elif x1>=5.5:
                ax2.text(x0+x1/2,y,f'{x1:.0f}',ha='center',va='center',fontsize=6.8,
                         color='white' if c=='#eb6834' else INK,weight='bold')
        left=left+vals
    ax2.set_yticks(ys); ax2.invert_yaxis()
    ax2.set_xlim(0,112); ax2.set_xticks([0,20,40,60,80,100])
    ax2.set_xlabel('% of listings with no number in 2025-12')
    ax2.set_title('(b) Listings with no number',loc='left',color=INK)
    ax2.grid(axis='y',visible=False)
    ax2.legend(loc='upper center',bbox_to_anchor=(0.5,-0.18),ncol=2,frameon=False,
               fontsize=7,handlelength=1.4,handletextpad=0.5,columnspacing=1.1)
    fig.tight_layout(); fig.subplots_adjust(bottom=0.27,wspace=0.10)
    # 합이 100 인지 확인(반올림 오차 허용)
    for nm,segn,pl in (('(a)','had_number',parts),('(b)','all',parts2)):
        for rg in rgs:
            tot=sum((agg(segn,c) or {}).get(rg,0) for c,_,_,_ in pl)
            if rg in (have_a if segn=='had_number' else have_b):
                assert abs(tot-100)<0.6, f'fig3 {nm} {rg} sums to {tot:.2f}, not 100'
    save(fig,'fig3_transitions')
    plt.close(fig); print('fig3 ok')

# ---------- 그림 4 ----------
def fig4():
    ml=json.load(open(os.path.join(BASE,'out','ml_results.json')))
    if not ml.get('perm_eu'): print('fig4 skipped (no perm)'); return
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.3),gridspec_kw={'width_ratios':[1.5,1]})
    ax=axes[0]
    eu={i['feature']:i['drop_bal_acc'] for i in ml['perm_eu']['importances']}
    ne={i['feature']:i['drop_bal_acc'] for i in (ml.get('perm_noneu') or {}).get('importances',[])}
    feats=[f for f,_ in sorted(eu.items(),key=lambda kv:-kv[1])[:9]][::-1]
    ys=np.arange(len(feats)); h=0.38
    ax.barh(ys+h/2,[eu.get(f,0) for f in feats],height=h,color='#2a78d6',
            edgecolor='white',linewidth=0.9,label='EU',zorder=3)
    ax.barh(ys-h/2,[ne.get(f,0) for f in feats],height=h,color='white',
            edgecolor='#52514e',linewidth=0.9,hatch='////',label='Non-EU',zorder=3)
    for y,f in zip(ys,feats):
        ax.annotate(f'{eu.get(f,0):.3f}',(eu.get(f,0),y+h/2),xytext=(3,0),
                    textcoords='offset points',va='center',fontsize=6.4,color=INK)
    ax.set_yticks(ys); ax.set_yticklabels([FEATLBL.get(f,f) for f in feats])
    ax.axvline(0,color=INK2,lw=0.7)
    ax.set_xlabel('Permutation importance\n(drop in balanced accuracy)')
    ax.set_title(f"(a) What predicts the transition — {ml['model']}",loc='left',color=INK)
    ax.grid(axis='y',visible=False); ax.legend(frameon=False,loc='lower right')
    ax2=axes[1]
    keys=[('es','EU · annulled (ES)'),('steady','EU · in force'),('eu','EU (all)'),('noneu','Non-EU')]
    keys=[(k,l) for k,l in keys if ml.get(f'cv_{k}')]
    xs=np.arange(len(keys))
    vals=[ml[f'cv_{k}']['auc_ovr_macro'] for k,_ in keys]
    cols=['#eb6834','#1baf7a','#a8a7a0','#52514e'][:len(keys)]
    ax2.bar(xs,vals,color=cols,edgecolor='white',linewidth=1.1,width=0.62,zorder=3)
    for x,v in zip(xs,vals):
        ax2.annotate(f'{v:.3f}',(x,v),xytext=(0,3),textcoords='offset points',
                     ha='center',fontsize=7,color=INK,weight='bold')
    ax2.axhline(0.5,color=INK2,lw=0.8,ls=(0,(4,2)))
    ax2.annotate('chance',(len(keys)-0.5,0.5),xytext=(0,3),textcoords='offset points',
                 ha='right',fontsize=6.6,color=INK2)
    ax2.set_xticks(xs); ax2.set_xticklabels([l for _,l in keys],rotation=20,ha='right')
    ax2.set_ylim(0,1.0); ax2.set_ylabel('AUC (one-vs-rest, macro)')
    ax2.set_title('(b) Out-of-sample fit, city-grouped CV',loc='left',color=INK)
    fig.tight_layout()
    save(fig,'fig4_ml_heterogeneity')
    plt.close(fig); print('fig4 ok')

if __name__=='__main__':
    which = sys.argv[1:] or ['1','2','3','4']
    if '1' in which: fig1()
    if '2' in which: fig2()
    if '3' in which: fig3()
    if '4' in which: fig4()
