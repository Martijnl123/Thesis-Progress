# Figures 1-3 for the thesis (6 Nov 2026, after the outside review). No September 2026 data, no scaled rank.
# Fig 1: mean mark and share below 99 by quarter, frozen Sep 2025 cohort, software vs other loans (LLM labels).
# Fig 2: event study (Equation 1 of the 6 Nov methodology): software x quarter dummies, X(Sep25) x quarter dummies,
#        position and fund x quarter FE, borrower-clustered 95% intervals. Base quarter Sep 2025.
# Fig 3: distribution of marks near par, Sep 2025 vs Jun 2026, software vs other (frozen cohort).
# Sample rules as v2_placebo_rank.py: positions clean at Sep 2025 with an LLM group other than UNKN/FUND; before Sep 2025
# the usual filter each quarter, after it 0<=mark<=110 and principal>=$250k.
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..')))  # repo root
import sys, warnings, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,(_ROOT+'/bdc/scripts'))
from ai_rev2_lib import fit, se
warnings.filterwarnings('ignore')
V=(_ROOT+'/bdc/v2'); F=(_ROOT+'/bdc/industry_llm/full')
OUT=(_ROOT+'/outputs/figures-2026-11-06')
import os; os.makedirs(OUT,exist_ok=True)

p=pd.read_parquet(f'{V}/panel.parquet')
p['fq']=p.cik.astype(str)+'_'+p.ddate.dt.strftime('%Y%m')
L=pd.read_parquet(f'{F}/labels.parquet')[['bid','group']]
pb=pd.read_parquet(f'{F}/position_borrower.parquet').merge(L,on='bid').merge(p[['pos_id','pkey']],on='pos_id').drop_duplicates('pkey').set_index('pkey')
p['grp']=p.pkey.map(pb.group); p['bid']=p.pkey.map(pb.bid)
lab=p.grp.notna()&~p.grp.isin(['UNKN','FUND'])
keys=set(p.loc[(p.ddate=='2025-09-30')&p.ok&lab,'pkey'])
fz=p[p.pkey.isin(keys)&lab]; post=fz.ddate>'2025-09-30'
d=fz[~post&fz.ok | post&fz.mark.between(0,110)&(fz.principal>=250e3)].copy()
d['sw']=(d.grp=='SOFT').astype(float); d['b99']=100*(d.mark<99)
qs=sorted(d.ddate.unique()); base=pd.Timestamp('2025-09-30')
lines=[f'frozen cohort: {len(keys)} positions, {d.bid.nunique()} borrowers, {d.cik.nunique()} funds, {d.family.nunique()} managers; rows {len(d)}']

# ---------- style (dataviz reference palette, light, print) ----------
SW,OT='#eb6834','#2a78d6'; INK,INK2,MUTED,GRID,AXIS='#0b0b0b','#52514e','#898781','#e1e0d9','#c3c2b7'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.edgecolor':AXIS,'axes.labelcolor':INK2,
    'xtick.color':MUTED,'ytick.color':MUTED,'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.color':GRID,'grid.linewidth':0.6,'axes.axisbelow':True,'legend.frameon':False})
def tz(ax):  # time zero: first post quarter is Mar 2026; shade from Dec 2025 to Mar 2026
    ax.axvspan(pd.Timestamp('2025-12-31'),pd.Timestamp('2026-03-31'),color='#f1f0ec',zorder=0,lw=0)
    ax.text(pd.Timestamp('2026-02-10'),ax.get_ylim()[1],'shock\nquarter',ha='center',va='top',fontsize=7,color=MUTED)
def yr(ax):
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator()); ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
    ax.tick_params(axis='x',labelsize=7.5)
def save(fig,name):
    fig.savefig(f'{OUT}/{name}.pdf',bbox_inches='tight'); fig.savefig(f'{OUT}/{name}.png',dpi=200,bbox_inches='tight'); plt.close(fig)

# ---------- Fig 1: raw means ----------
m=d.groupby(['ddate','sw']).agg(mark=('mark','mean'),b99=('b99','mean'),n=('mark','size')).reset_index()
fig,axs=plt.subplots(1,2,figsize=(7.2,2.8))
for ax,(y,ttl) in zip(axs,[('b99','Share marked below 99 (%)'),('mark','Mean mark (fair value per 100 of principal)')]):
    for s,c,nm in [(1,SW,'Software'),(0,OT,'Other loans')]:
        x=m[m.sw==s]; ax.plot(x.ddate,x[y],color=c,lw=2,marker='o',ms=3.5,label=nm)
        ax.annotate(nm,(x.ddate.iloc[-1],x[y].iloc[-1]),xytext=(4,0),textcoords='offset points',va='center',fontsize=7.5,color=INK2)
    ax.set_title(ttl,fontsize=9,color=INK,loc='left'); tz(ax)
    ax.set_xlim(qs[0]-pd.Timedelta(days=30),qs[-1]+pd.Timedelta(days=200))
    yr(ax)
axs[0].legend(loc='lower left',fontsize=7.5)
fig.tight_layout(); save(fig,'fig1_timeseries')
lines.append('\nFig 1 data (frozen cohort, LLM labels):\n'+m.pivot(index='ddate',columns='sw',values=['mark','b99','n']).round(2).to_string())

# ---------- Fig 2: event study ----------
ev=[q for q in qs if q!=base]
for q in ev:
    k=q.strftime('%Y%m'); dq=(d.ddate==q).astype(float)
    d[f'sw{k}']=d.sw*dq; d[f'sp{k}']=d.spread_pre.fillna(0)*100*dq; d[f'spm{k}']=d.spread_pre.isna()*dq; d[f'pk{k}']=d.pik_pre*dq
SWC=[f'sw{q.strftime("%Y%m")}' for q in ev]; XC=[f'{a}{q.strftime("%Y%m")}' for q in ev for a in ['sp','spm','pk']]
es={}
for y in ['b99','mark']:
    dd,b,sc,Ainv=fit(d,y,SWC+XC,['pkey','fq']); s=se(dd,sc,Ainv,'bid')[:len(SWC)]
    es[y]=pd.DataFrame({'q':ev,'b':b[SWC].values,'se':s})
    lines.append(f'\nevent study {y} (borrower-clustered SE, {dd.bid.nunique()} borrowers): '+'; '.join(f'{r.q:%Y-%m} {r.b:+.2f} ({r.se:.2f})' for r in es[y].itertuples()))
fig,axs=plt.subplots(1,2,figsize=(7.2,2.8))
for ax,(y,ttl) in zip(axs,[('b99','Below 99: software minus other (pp)'),('mark','Mark: software minus other (points)')]):
    e=pd.concat([es[y],pd.DataFrame({'q':[base],'b':[0.0],'se':[0.0]})]).sort_values('q')
    ax.axhline(0,color=AXIS,lw=1)
    ax.fill_between(e.q,e.b-1.96*e.se,e.b+1.96*e.se,color=SW,alpha=0.18,lw=0)
    ax.plot(e.q,e.b,color=SW,lw=2,marker='o',ms=3.5)
    ax.plot([base],[0],marker='o',ms=5,mfc='white',mec=SW)
    ax.annotate('base: Sep 2025',(base,0),xytext=(-6,-14 if y=='b99' else 10),textcoords='offset points',ha='right',fontsize=7,color=MUTED)
    ax.set_title(ttl,fontsize=9,color=INK,loc='left'); tz(ax)
    yr(ax)
fig.tight_layout(); save(fig,'fig2_event_study')

# ---------- Fig 3: distribution near par (grouped bars, marks below 80 pooled at 80, above 100 pooled at 100) ----------
edges=np.arange(80,102,1.0)
fig,axs=plt.subplots(1,2,figsize=(7.2,2.8),sharey=True)
for ax,(s_,nm) in zip(axs,[(1,'Software'),(0,'Other loans')]):
    for k,(q,c,lb) in enumerate([('2025-09-30','#b5b3ab','Sep 2025'),('2026-06-30',SW if s_ else OT,'Jun 2026')]):
        x=d[(d.sw==s_)&(d.ddate==q)].mark
        h=np.histogram(x.clip(80,100.5),bins=edges)[0]/len(x)*100
        ax.bar(edges[:-1]+0.5+(k-0.5)*0.42,h,width=0.4,color=c,label=lb,linewidth=0)
        lines.append(f'Fig 3 {nm} {q[:7]}: n {len(x)}, at or above 99: {(x>=99).mean()*100:.1f}%, below 80: {(x<80).mean()*100:.1f}%')
    ax.set_title(nm,fontsize=9,color=INK,loc='left'); ax.set_xlabel('Mark')
    ax.set_xticks([80.5,85.5,90.5,95.5,100.5]); ax.set_xticklabels(['80 or less','85','90','95','100+'])
    ax.grid(axis='x',visible=False); ax.legend(loc='upper left',fontsize=7.5)
axs[0].set_ylabel('Share of positions (%)')
fig.tight_layout(); save(fig,'fig3_distribution')

open(f'{OUT}/results_figures_2026-11-06.txt','w').write('\n'.join(lines)+'\n'); print('\n'.join(lines))
