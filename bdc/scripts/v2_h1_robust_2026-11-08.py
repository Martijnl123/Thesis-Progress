# Robustness checks (8 Nov 2026). Quarters to 30 June 2026 only (hold-out lock: no Sep 2026 data).
# Headline event study of PAP v5 section 4 (position FE, fund x quarter FE, vintage x quarter FE, pre-period mean spread
# (+missing dummy) and ever-PIK x quarter), borrower-clustered, on:
#   (a) the frozen cohort (headline, reproduces c07)
#   (b) $1m principal floor (cohort at Sep 2025 and after) (#37)
#   (c) revolvers and delayed-draw tranches dropped (#36)
#   (d) both
# and redraws Fig 2 with vintage x quarter effects (#46).
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..')))  # repo root
import sys, warnings, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,(_ROOT+'/bdc/scripts'))
from ai_rev2_lib import fit, se
warnings.filterwarnings('ignore')
V=(_ROOT+'/bdc/v2'); F=(_ROOT+'/bdc/industry_llm/full')
OUT=(_ROOT+'/outputs/figures-2026-11-08')
p=pd.read_parquet(f'{V}/panel.parquet'); assert p.ddate.max()<=pd.Timestamp('2026-06-30')
pos=pd.read_parquet(f'{V}/positions.parquet',columns=['pos_id','ident']); p=p.merge(pos,on='pos_id',how='left')
p['fq']=p.cik.astype(str)+'_'+p.ddate.dt.strftime('%Y%m')
p['vint']=p.pkey.map(p.groupby('pkey').ddate.min()).dt.strftime('%Y%m')
L=pd.read_parquet(f'{F}/labels.parquet')[['bid','group']]
pb=pd.read_parquet(f'{F}/position_borrower.parquet').merge(L,on='bid').merge(p[['pos_id','pkey']],on='pos_id').drop_duplicates('pkey').set_index('pkey')
p['grp']=p.pkey.map(pb.group); p['bid']=p.pkey.map(pb.bid)
lab=p.grp.notna()&~p.grp.isin(['UNKN','FUND'])
s9=p[(p.ddate=='2025-09-30')&p.ok&lab]
rev=s9.ident.fillna('').str.lower().str.contains(r'revolv|\brcf\b|delayed|\bddtl\b')
out=[]
def build(keys,floor):
    fz=p[p.pkey.isin(keys)&lab]; post=fz.ddate>'2025-09-30'
    d=fz[(~post&fz.ok&(fz.principal>=floor)) | (post&fz.mark.between(0,110)&(fz.principal>=floor))].copy()
    d['sw']=(d.grp=='SOFT').astype(float); d['b99']=100*(d.mark<99)
    d['vq']=d.vint+'|'+d.ddate.dt.strftime('%Y%m'); return d
def es(d,fe=('pkey','fq','vq')):
    qs=sorted(d.ddate.unique()); base=pd.Timestamp('2025-09-30'); ev=[q for q in qs if q!=base]
    for q in ev:
        k=q.strftime('%Y%m'); dq=(d.ddate==q).astype(float)
        d[f'sw{k}']=d.sw*dq; d[f'sp{k}']=d.spread_pre.fillna(0)*100*dq; d[f'spm{k}']=d.spread_pre.isna()*dq; d[f'pk{k}']=d.pik_pre.fillna(0)*dq
    SWC=[f'sw{q.strftime("%Y%m")}' for q in ev]; XC=[f'{a}{q.strftime("%Y%m")}' for q in ev for a in ['sp','spm','pk']]
    r={}
    for y in ['mark','b99']:
        dd,b,sc,Ainv=fit(d,y,SWC+XC,list(fe)); s=se(dd,sc,Ainv,'bid')[:len(SWC)]
        r[y]=pd.DataFrame({'q':ev,'b':b[SWC].values,'se':s}); r['n']=(len(dd),dd.pkey.nunique(),dd.bid.nunique())
    return r
sel=['2022-12','2024-12','2025-06','2025-12','2026-03','2026-06']
specs=[('(a) headline',set(s9.pkey),250e3),('(b) $1m floor',set(s9.loc[s9.principal>=1e6,'pkey']),1e6),
       ('(c) no revolvers/delayed draw',set(s9.loc[~rev,'pkey']),250e3),('(d) $1m floor and no revolvers/delayed draw',set(s9.loc[(s9.principal>=1e6)&~rev,'pkey']),1e6)]
res={}
for nm,keys,fl in specs:
    d=build(keys,fl); r=es(d); res[nm]=r
    out.append(f'{nm}: rows {r["n"][0]}, positions {r["n"][1]}, borrowers {r["n"][2]}')
    for y in ['mark','b99']:
        e=r[y]; e=e[e.q.dt.strftime('%Y-%m').isin(sel)]
        out.append(f'   {y}: '+'; '.join(f'{x.q:%Y-%m} {x.b:+.2f} ({x.se:.2f})' for x in e.itertuples()))
    print(out[-3:],flush=True)
open(f'{V}/results_h1_robust_2026-11-08.txt','w').write('\n'.join(out)+'\n')
# Fig 2 with vintage x quarter (headline)
SW,INK,MUTED,GRID,AXIS='#eb6834','#0b0b0b','#898781','#e1e0d9','#c3c2b7'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.edgecolor':AXIS,'xtick.color':MUTED,'ytick.color':MUTED,
    'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':GRID,'grid.linewidth':0.6,'axes.axisbelow':True})
base=pd.Timestamp('2025-09-30'); fig,axs=plt.subplots(1,2,figsize=(7.2,2.8))
for ax,(y,ttl) in zip(axs,[('b99','Below 99: software minus other (pp)'),('mark','Mark: software minus other (points)')]):
    e=pd.concat([res['(a) headline'][y],pd.DataFrame({'q':[base],'b':[0.0],'se':[0.0]})]).sort_values('q')
    ax.axhline(0,color=AXIS,lw=1); ax.fill_between(e.q,e.b-1.96*e.se,e.b+1.96*e.se,color=SW,alpha=0.18,lw=0)
    ax.plot(e.q,e.b,color=SW,lw=2,marker='o',ms=3.5); ax.plot([base],[0],marker='o',ms=5,mfc='white',mec=SW)
    ax.axvspan(pd.Timestamp('2025-12-31'),pd.Timestamp('2026-03-31'),color='#f1f0ec',zorder=0,lw=0)
    ax.set_title(ttl,fontsize=9,color=INK,loc='left')
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator()); ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
fig.tight_layout(); fig.savefig(f'{OUT}/fig2_event_study_vintage.pdf',bbox_inches='tight'); fig.savefig(f'{OUT}/fig2_event_study_vintage.png',dpi=200,bbox_inches='tight')
print('done')
