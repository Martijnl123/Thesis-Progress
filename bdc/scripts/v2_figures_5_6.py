# Figures 5-6 for the thesis (6 Nov 2026). Data through Jun 2026 only (no Sep 2026, no scaled rank).
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
OUT=(_ROOT+'/outputs/figures-2026-11-06')
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
s0=d[d.sw&(d.q=='2025-09')]; trad=set(s0.loc[s0.px.notna(),'pkey']); priv=set(s0.loc[~s0.held,'pkey'])
for ks,c,ls,nm in [(trad,SW,'-','Traded (market price)'),(priv,SW,':','Not held by any registered fund')]:
    x=d[d.pkey.isin(ks)].pivot_table(index='pkey',columns='q',values='mark').dropna()
    dm=[(x[q]-x['2025-09']).mean() for q in Q]
    ax.plot(QL,dm,color=c,lw=2,ls=ls,marker='o',ms=3.5); ax.annotate(f'{nm}\n(n={len(x)})',(3,dm[-1]),xytext=(5,0),textcoords='offset points',va='center',fontsize=7,color=INK2)
    lines.append(f'Fig5b software {nm}: {len(x)} positions held at all four dates; mark change '+', '.join(f'{v:+.2f}' for v in dm))
ax.axhline(0,color=AXIS,lw=1); ax.set_title('(b) BDC marks on software loans',fontsize=9,color=INK,loc='left')
for a in axs: a.tick_params(axis='x',labelsize=7.5)
fig.tight_layout(w_pad=6); save(fig,'fig5_nport_same_loan')
# ---------- Fig 6 (pairs as v2_h3.py) ----------
q=p.set_index('pos_id')
pr=pd.read_parquet(f'{V}/pairs_0.5.parquet'); pr=pr[(pr.cos>=0.6)&(pr.pa<pr.pc)]
ok=q.ok; pr=pr[ok.reindex(pr.pa).values & ok.reindex(pr.pc).values]
for s,c in [('a','pa'),('c','pc')]:
    for col in ['family','mark','ddate','loan','sector']: pr[f'{col}_{s}']=q[col].reindex(pr[c]).values
pr['sector']=pr.sector_a.fillna(pr.sector_c); pr=pr[pr.sector.notna()&~pr.sector.isin(['it_services','tech_other'])&(pr.family_a!=pr.family_c)]
pr['fa']=np.minimum(pr.family_a,pr.family_c); pr['fc']=np.maximum(pr.family_a,pr.family_c)
pr['m_fa']=np.where(pr.family_a==pr.fa,pr.mark_a,pr.mark_c); pr['m_fc']=np.where(pr.family_a==pr.fa,pr.mark_c,pr.mark_a)
g=pr.groupby(['loan_a','ddate_a','fa','fc']).agg(ma=('m_fa','mean'),mc=('m_fc','mean'),sector=('sector','first')).reset_index()
g['sw']=g.sector.eq('software'); g['absd']=(g.ma-g.mc).abs(); g['q']=g.ddate_a.dt.strftime('%Y-%m'); g['hi']=np.maximum(g.ma,g.mc)
g['pair']=g.loan_a.astype(str)+'|'+g.fa+'|'+g.fc; g['w']=1/g.groupby(['loan_a','q']).pair.transform('size')
base=g[g.q=='2025-09']; coh=set(base.loc[base.hi>=98,'pair'])
c6=g[g.pair.isin(coh)&g.q.isin(['2024-12','2025-03','2025-06']+Q)]
QQ=['2024-12','2025-03','2025-06']+Q
fig,axs=plt.subplots(1,2,figsize=(7.2,2.9))
ax=axs[0]
for s,c,nm in [(True,SW,'Software'),(False,OT,'Other')]:
    x=c6[c6.sw==s]; m=x.groupby('q').apply(lambda t: np.average(t.absd,weights=t.w)).reindex(QQ)
    ax.plot(range(len(QQ)),m.values,color=c,lw=2,marker='o',ms=3.5); ax.annotate(nm,(len(QQ)-1,m.values[-1]),xytext=(5,0),textcoords='offset points',va='center',fontsize=7.5,color=INK2)
    lines.append(f'Fig6a {nm}: pairs in cohort {x[x.q=="2025-09"].pair.nunique()}; weighted mean gap '+', '.join(f'{k} {v:.2f}' for k,v in m.items()))
ax.set_xticks(range(len(QQ))); ax.set_xticklabels(['Dec 24','Mar 25','Jun 25']+QL,fontsize=7)
ax.axvline(3,color=AXIS,lw=1,ls=':'); ax.text(3,ax.get_ylim()[1],' cohort fixed',fontsize=7,color=MUTED,va='top')
ax.set_ylabel('Mean |mark A - mark B| (points)'); ax.set_title('(a) Gap between two managers, same loan',fontsize=9,color=INK,loc='left')
ax=axs[1]; j=c6[c6.q=='2026-06'].copy()
bins=[-1,90,95,97,98,99,99.5,1e9]; labs=['<90','90-95','95-97','97-98','98-99','99-99.5','99.5+']
j['bin']=pd.cut(j.hi,bins,labels=labs)
xs=np.arange(len(labs))
for k,(s,c,nm) in enumerate([(True,SW,'Software'),(False,OT,'Other')]):
    t=j[j.sw==s].groupby('bin').apply(lambda u: pd.Series({'m':np.average(u.absd,weights=u.w) if len(u) else np.nan,'n':len(u)})).reindex(labs)
    ax.bar(xs+(k-0.5)*0.4,t.m.values,width=0.38,color=c,label=nm,lw=0)
    lines.append(f'Fig6b {nm} Jun 2026 gap by higher mark: '+', '.join(f'{b} {m:.2f} (n={int(n)})' for b,(m,n) in zip(labs,t[['m','n']].fillna(0).values)))
ax.set_xticks(xs); ax.set_xticklabels(labs,fontsize=7); ax.set_xlabel('Higher of the two marks, Jun 2026')
ax.legend(fontsize=7.5,loc='upper right'); ax.set_title('(b) Jun 2026 gap, by the higher of the two marks',fontsize=9,color=INK,loc='left')
fig.tight_layout(w_pad=3); save(fig,'fig6_h3_gap')
open(f'{OUT}/results_figures_5_6_2026-11-06.txt','w').write('\n'.join(lines)+'\n'); print('\n'.join(lines))
