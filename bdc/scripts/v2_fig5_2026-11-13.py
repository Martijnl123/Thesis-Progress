# Figure 5 rebuilt 13 Nov 2026 (panel (a) unchanged from v2_figures_5_6.py; panel (b) uses the cleaned traded/private classes).
# Was: Figures 5-6 for the thesis (6 Nov 2026). Data through Jun 2026 only (no Sep 2026, no scaled rank).
# Fig 5: N-PORT same-loan benchmark, frozen Sep 2025 cohort (LLM labels).
#   (a) positions with a Level 2 (market) price at every quarter-end Sep 2025-Jun 2026: mean change since Sep 2025 in the BDC
#       mark and in the N-PORT market price, software vs other.
#   (b) BDC marks of software loans that trade (Level 2 price at Sep 2025) vs software loans no registered fund holds.
# Fig 6: H3, gaps between two managers' marks on the same loan (cohort of pairs within 98+ at Sep 2025, as v2_h3.py).
#   (a) mean absolute gap by quarter; (b) Jun 2026 gap by the higher of the two marks (level-dependent dispersion).
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..')))  # repo root
import sys, warnings, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
V=(_ROOT+'/bdc/v2'); F=(_ROOT+'/bdc/industry_llm/full')
OUT=(_ROOT+'/outputs/figures-2026-11-13')
SW,OT='#eb6834','#2a78d6'; INK,INK2,MUTED,GRID,AXIS='#0b0b0b','#52514e','#898781','#e1e0d9','#c3c2b7'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.edgecolor':AXIS,'axes.labelcolor':INK2,
    'xtick.color':MUTED,'ytick.color':MUTED,'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.color':GRID,'grid.linewidth':0.6,'axes.axisbelow':True,'legend.frameon':False})
def save(fig,name):
    fig.savefig(f'{OUT}/{name}.pdf',bbox_inches='tight'); fig.savefig(f'{OUT}/{name}.png',dpi=200,bbox_inches='tight'); plt.close(fig)
Q=['2025-09','2025-12','2026-03','2026-06']; QL=['Sep 25','Dec 25','Mar 26','Jun 26']
lines=[]
# ---------- cohort (as v2_figures.py) ----------
p=pd.read_parquet(f'{V}/panel.parquet')
L=pd.read_parquet(f'{F}/labels.parquet')[['bid','group']]
pb=pd.read_parquet(f'{F}/position_borrower.parquet').merge(L,on='bid').merge(p[['pos_id','pkey']],on='pos_id').drop_duplicates('pkey').set_index('pkey')
p['grp']=p.pkey.map(pb.group); p['bid']=p.pkey.map(pb.bid)
lab=p.grp.notna()&~p.grp.isin(['UNKN','FUND'])
keys=set(p.loc[(p.ddate=='2025-09-30')&p.ok&lab,'pkey'])
fz=p[p.pkey.isin(keys)&lab]; post=fz.ddate>'2025-09-30'
d=fz[~post&fz.ok | post&fz.mark.between(0,110)&(fz.principal>=250e3)].copy()
d['sw']=d.grp.eq('SOFT'); d['q']=d.ddate.dt.strftime('%Y-%m'); d=d[d.q.isin(Q)]
# ---------- Fig 5 ----------
M=pd.read_parquet(f'{V}/nport_matches.parquet'); M=M[M.verified]
l2=M[M.lvl=='2'].groupby('pos_id').price_median.median().rename('px')
anyh=set(M.pos_id)
d['px']=d.pos_id.map(l2); d['held']=d.pos_id.isin(anyh)
w=d.pivot_table(index=['pkey','sw','bid'],columns='q',values=['mark','px'])
bal=w.dropna(subset=[('px',q) for q in Q]+[('mark',q) for q in Q])
fig,axs=plt.subplots(1,2,figsize=(7.2,2.9))
ax=axs[0]
for s,c,nm in [(True,SW,'Software'),(False,OT,'Other')]:
    x=bal.xs(s,level='sw')
    dm=[(x[('mark',q)]-x[('mark','2025-09')]).mean() for q in Q]; dp=[(x[('px',q)]-x[('px','2025-09')]).mean() for q in Q]
    ax.plot(QL,dm,color=c,lw=2,marker='o',ms=3.5); ax.plot(QL,dp,color=c,lw=1.5,ls='--',marker='s',ms=3,mfc='white')
    ax.annotate(f'{nm} (n={len(x)})',(3,dm[-1]),xytext=(5,0),textcoords='offset points',va='center',fontsize=7.5,color=INK2)
    lines.append(f'Fig5a {nm}: {len(x)} positions, {x.index.get_level_values("bid").nunique()} borrowers; change since Sep 2025, mark '+', '.join(f'{v:+.2f}' for v in dm)+' | price '+', '.join(f'{v:+.2f}' for v in dp))
ax.axhline(0,color=AXIS,lw=1)
ax.plot([],[],color=INK2,lw=2,label='BDC mark'); ax.plot([],[],color=INK2,lw=1.5,ls='--',marker='s',ms=3,mfc='white',label='N-PORT market price')
fig.legend(*ax.get_legend_handles_labels(),fontsize=7.5,loc='lower center',ncol=2,bbox_to_anchor=(0.3,-0.06)); ax.set_ylabel('Change since Sep 2025 (points)')
ax.set_title('(a) Same loan, BDC mark vs market price',fontsize=9,color=INK,loc='left')
ax=axs[1]
# Panel (b) rebuilt 13 Nov 2026 from the cleaned classes of v2_traded_untraded_2026-11-13.py (loan-level traded status,
# private = in the N-PORT match universe and no traded loan for the borrower, hand-checked). Means on positions held at each date.
T=pd.read_parquet(f'{V}/traded_untraded_sep25_2026-11-13.parquet'); oth=[]
for sw,cls,c,ls,lw,nm in [(1,'traded',SW,'-',2,'Software, traded'),(1,'untraded',SW,':',2,'Software, private'),
                          (0,'traded',OT,'-',1.3,'Other, traded'),(0,'untraded',OT,':',1.3,'Other, private')]:
    x=T[(T.sw==sw)&(T.cls==cls)]
    dm=[0.0]+[x[f'dm_{q}'].mean() for q in Q[1:]]; n=int(x['dm_2026-06'].notna().sum())
    ax.plot(QL,dm,color=c,lw=lw,ls=ls,marker='o',ms=3 if sw else 2.5)
    if sw: ax.annotate(f'{nm} (n={n})',(3,dm[-1]),xytext=(5,0),textcoords='offset points',va='center',fontsize=7,color=INK2)
    else: oth.append((dm[-1],n))
    lines.append(f'Fig5b {nm}: Sep 2025 positions {len(x)}, held Jun 2026 {n}; mean mark change '+', '.join(f'{v:+.2f}' for v in dm[1:]))
ax.annotate(f'Other, both (n={oth[0][1]}, {oth[1][1]})',(3,np.mean([v for v,_ in oth])),xytext=(5,0),textcoords='offset points',va='center',fontsize=7,color=INK2)
ax.plot([],[],color=INK2,lw=2,label='Traded (market price)'); ax.plot([],[],color=INK2,lw=2,ls=':',label='Private (no traded debt)')
ax.legend(fontsize=7,loc='center left',bbox_to_anchor=(0.0,0.42))
ax.axhline(0,color=AXIS,lw=1); ax.set_title('(b) BDC marks, traded vs private loans',fontsize=9,color=INK,loc='left')
for a in axs: a.tick_params(axis='x',labelsize=7.5)
fig.tight_layout(w_pad=6); save(fig,'fig5_nport_same_loan')
open(f'{OUT}/results_fig5_2026-11-13.txt','w').write('\n'.join(lines)+'\n'); print('\n'.join(lines))
