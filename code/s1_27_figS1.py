#!/usr/bin/env python3
"""Figure S1: share displaying a number in each market before and after the application date."""
import os, sys, json
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.join(BASE,'scripts'))
from s1_09_estimate import GROUP_BY_COUNTRY
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Liberation Sans','DejaVu Sans'],
 'font.size':8,'axes.labelsize':8.5,'axes.titlesize':9,'figure.dpi':200,'savefig.dpi':400,
 'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.linewidth':0.4,
 'grid.alpha':0.35,'grid.color':'#b8b7b0','axes.linewidth':0.6})
C={'es':'#eb6834','fr':'#2a78d6','steady':'#1baf7a','eea':'#e34948','noscheme':'#6250d6','unver':'#e87ba4','noneu':'#52514e'}
MK={'es':'D','fr':'o','steady':'s','eea':'v','noscheme':'^','unver':'P','noneu':'X'}
LBL={'es':'Spain · procedure annulled','fr':'France · already in force','steady':'EU · in force',
 'eea':'EEA non-EU','noscheme':'EU · no scheme','unver':'EU · status unverified','noneu':'Non-EU control'}
ORDER=['es','fr','steady','eea','noscheme','unver','noneu']
REG={'EU_aligned':'fr','EU_reversal':'es','EU_pretreated':'steady','EU_noscheme':'noscheme',
     'nonEU':'noneu','EEA_nonEU':'eea','EU_unconfirmed':'unver'}
p=pd.read_csv(os.path.join(BASE,'out','city_period_panel.csv'))
p['regime']=p.country.map(lambda c: REG.get(GROUP_BY_COUNTRY.get(c,'?'),'?'))
pre=p[p.t_rel==-2].set_index('unit'); post=p[p.t_rel==1].set_index('unit')
common=pre.index.intersection(post.index)
fig,ax=plt.subplots(figsize=(5.6,5.2))
ax.plot([0,100],[0,100],color='#52514e',lw=0.8,ls=(0,(4,2)),zorder=1)
for g in ORDER:
    u=[x for x in common if pre.loc[x,'regime']==g]
    if not u: continue
    ax.scatter(pre.loc[u,'pct_number'],post.loc[u,'pct_number'],
        s=np.clip(pre.loc[u,'n_listings']/180,14,340),facecolor=C[g],edgecolor='white',
        linewidth=0.8,marker=MK[g],alpha=0.9,zorder=3)
for x in common:
    d=post.loc[x,'pct_number']-pre.loc[x,'pct_number']
    if abs(d)>=8:
        ax.annotate(x.split(' | ')[1],(pre.loc[x,'pct_number'],post.loc[x,'pct_number']),
            textcoords='offset points',xytext=(7,-3),fontsize=6.3,color='#0b0b0b')
ax.set_xlim(-4,104); ax.set_ylim(-4,104)
ax.set_xlabel('Display share, 2025-12 (%)'); ax.set_ylabel('Display share, 2026-06 (%)')
ax.text(56,7,'below the line = disclosure fell',fontsize=6.9,color='#52514e')
h=[Line2D([0],[0],marker=MK[g],color=C[g],lw=0,ms=5.5,mec='white',mew=0.7,label=LBL[g]) for g in ORDER]
ax.legend(handles=h,loc='upper left',frameon=False,fontsize=7,handletextpad=0.4)
fig.tight_layout()
os.makedirs(os.path.join(BASE,'supp','figs'),exist_ok=True)
for ext in ('png','pdf'):
    fig.savefig(os.path.join(BASE,'supp','figs',f'FigureS1.{ext}'),bbox_inches='tight',facecolor='white')
print('FigureS1 regenerated')
