# Exits (PAP v5 section 4 'Exits'), 8 Nov 2026. Quarters to 30 Jun 2026 only.
# 1. Leaving the frozen cohort: share of cohort positions observed at Sep 2025 that are no longer in the sample, by quarter,
#    software vs other; and software x quarter on an exit indicator with fund x quarter FE (borrower-clustered).
# 2. Classification of exits from exits.parquet (v2_exits.py): similar position remains in the fund (refinanced/restructured),
#    gone with last mark 97+ (likely repaid or sold near par), 80-97, <80 (distressed), fund stopped reporting;
#    plus whether the same cross-fund loan is still held by another BDC the next quarter (sold or partially repaid, not a payoff).
# 3. Imputation bounds on the headline event study (Mar and Jun 2026): leavers get rows in every later quarter in which their
#    fund reports, imputed high (last mark) in one group and low (10th percentile of that quarter's marks, or a fixed 50) in the
#    other, then reversed; worst-case bounds use 0 and 110 for the mark and 0 and 100 for Below99.
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..')))  # repo root
import sys, warnings, numpy as np, pandas as pd
sys.path.insert(0,(_ROOT+'/bdc/scripts'))
from ai_rev2_lib import fit, se
warnings.filterwarnings('ignore')
V=(_ROOT+'/bdc/v2'); F=(_ROOT+'/bdc/industry_llm/full')
p=pd.read_parquet(f'{V}/panel.parquet'); assert p.ddate.max()<=pd.Timestamp('2026-06-30')
p['vint']=p.pkey.map(p.groupby('pkey').ddate.min()).dt.strftime('%Y%m')
L=pd.read_parquet(f'{F}/labels.parquet')[['bid','group']]
pb=pd.read_parquet(f'{F}/position_borrower.parquet').merge(L,on='bid').merge(p[['pos_id','pkey']],on='pos_id').drop_duplicates('pkey').set_index('pkey')
p['grp']=p.pkey.map(pb.group); p['bid']=p.pkey.map(pb.bid)
lab=p.grp.notna()&~p.grp.isin(['UNKN','FUND'])
keys=set(p.loc[(p.ddate=='2025-09-30')&p.ok&lab,'pkey'])
fz=p[p.pkey.isin(keys)&lab]; post=fz.ddate>'2025-09-30'
d=fz[~post&fz.ok | post&fz.mark.between(0,110)&(fz.principal>=250e3)].copy()
d['sw']=(d.grp=='SOFT').astype(float); d['b99']=100*(d.mark<99)
out=[]; Q=[pd.Timestamp(x) for x in ['2025-12-31','2026-03-31','2026-06-30']]
rep=p.groupby('ddate').cik.apply(set)
s9=d[d.ddate=='2025-09-30'].set_index('pkey')
# 1. leaving
rows=[]
for q in Q:
    inq=set(d.loc[d.ddate==q,'pkey']); 
    for k,r in s9.iterrows():
        rows.append((k,q,r.cik,r.sw,r.bid,int(k not in inq),int(r.cik in rep[q])))
X=pd.DataFrame(rows,columns=['pkey','q','cik','sw','bid','gone','fund_reports'])
t=X.groupby(['q','sw']).agg(gone=('gone','mean'),fund_reports=('fund_reports','mean'),n=('gone','size')).round(3)
out.append('1. Share of the Sep 2025 cohort not in the sample at each quarter (cumulative), and share whose fund reports that quarter:\n'+t.to_string())
Xr=X[X.fund_reports==1].copy(); Xr['fq']=Xr.cik.astype(str)+Xr.q.dt.strftime('%Y%m')
for q in Q:
    Xr[f'sw{q:%Y%m}']=Xr.sw*(Xr.q==q)
C=[f'sw{q:%Y%m}' for q in Q]
dd,b,sc,Ainv=fit(Xr.assign(gone=100*Xr.gone),'gone',C,['fq']); s=se(dd,sc,Ainv,'bid')
out.append('   software minus other, cumulative exit rate (pp), fund x quarter FE, borrower-clustered: '+'; '.join(f'{c[2:]} {b[c]:+.2f} ({s[i]:.2f})' for i,c in enumerate(C)))
# 2. classification of first exits (Sep 2025 -> Jun 2026) of cohort positions
ex=pd.read_parquet(f'{V}/exits.parquet')[['pos_id','status']]
pp=p[['pos_id','pkey','ddate','loan','cik','family']].merge(ex,on='pos_id',how='left')
seq=d[['pkey','ddate']].sort_values(['pkey','ddate'])
last=d[d.ddate>='2025-09-30'].sort_values('ddate').groupby('pkey').tail(1)
last=last[last.ddate<'2026-06-30']  # positions whose last sample quarter is before June 2026 = exited
last=last.merge(pp[['pos_id','status']],on='pos_id',how='left')
nxt={pd.Timestamp('2025-09-30'):Q[0],Q[0]:Q[1],Q[1]:Q[2]}
lp=last.copy()
nh={q:p[p.ddate==q].groupby('loan').family.apply(set) for q in Q}
lp['loan_elsewhere']=[ (r.loan in nh[nxt[r.ddate]].index) and len(nh[nxt[r.ddate]][r.loan]-{r.family})>0 for r in lp.itertuples()]
lp['status']=lp.status.fillna('not in exits file')
lp['sw']=lp.sw.map({1.0:'software',0.0:'other'})
c=pd.crosstab(lp.status,lp.sw,normalize='columns').round(3); c.loc['n']=lp.sw.value_counts()
out.append('\n2. Exits Sep 2025 -> Jun 2026 (last sample quarter before June 2026), by type (share of exits):\n'+c.to_string())
out.append('   same loan still held by another manager next quarter (sold or partly repaid rather than paid off): '+lp.groupby('sw').loan_elsewhere.mean().round(3).to_string().replace('\n','; '))
# 3. imputation bounds
d['fq']=d.cik.astype(str)+'_'+d.ddate.dt.strftime('%Y%m'); d['vq']=d.vint+'|'+d.ddate.dt.strftime('%Y%m')
obs=set(zip(d.pkey,d.ddate)); lastmk=d.sort_values('ddate').groupby('pkey').mark.last()
p10={q:d.loc[d.ddate==q,'mark'].quantile(.1) for q in Q}
imp=[]
for k,r in s9.iterrows():
    for q in Q:
        if (k,q) not in obs and r.cik in rep[q] and d[(d.pkey==k)&(d.ddate<q)].shape[0]>0:
            imp.append((k,q,r.cik,r.sw,r.bid,r.vint,r.spread_pre,r.pik_pre))
I=pd.DataFrame(imp,columns=['pkey','ddate','cik','sw','bid','vint','spread_pre','pik_pre'])
I['last']=I.pkey.map(lastmk)
I['fq']=I.cik.astype(str)+'_'+I.ddate.dt.strftime('%Y%m'); I['vq']=I.vint+'|'+I.ddate.dt.strftime('%Y%m')
out.append(f'\n3. Imputed leaver-quarters: {len(I)} (software {int(I.sw.sum())}); p10 of marks: '+', '.join(f'{q:%Y-%m} {v:.1f}' for q,v in p10.items()))
def es(dd_):
    qs=sorted(dd_.ddate.unique()); base=pd.Timestamp('2025-09-30'); ev=[q for q in qs if q!=base]
    for q in ev:
        k=q.strftime('%Y%m'); dq=(dd_.ddate==q).astype(float)
        dd_[f'sw{k}']=dd_.sw*dq; dd_[f'sp{k}']=dd_.spread_pre.fillna(0)*100*dq; dd_[f'spm{k}']=dd_.spread_pre.isna()*dq; dd_[f'pk{k}']=dd_.pik_pre.fillna(0)*dq
    SWC=[f'sw{q.strftime("%Y%m")}' for q in ev]; XC=[f'{a}{q.strftime("%Y%m")}' for q in ev for a in ['sp','spm','pk']]
    r={}
    for y in ['mark','b99']:
        z,b,sc,Ainv=fit(dd_,y,SWC+XC,['pkey','fq','vq']); s=se(z,sc,Ainv,'bid')[:len(SWC)]
        r[y]={q:(b[f'sw{q:%Y%m}'],s[ev.index(q)]) for q in Q[1:]}
    return r
base_cols=['pkey','ddate','cik','sw','bid','vint','spread_pre','pik_pre','fq','vq','mark','b99']
def scen(nm,sw_mark,ot_mark,sw_b=None,ot_b=None):
    J=I.copy()
    J['mark']=np.where(J.sw==1,sw_mark(J),ot_mark(J))
    J['b99']=100*(J.mark<99) if sw_b is None else np.where(J.sw==1,sw_b,ot_b)
    r=es(pd.concat([d[base_cols],J[base_cols]],ignore_index=True))
    out.append(f'   {nm}: '+' | '.join(f'{y} '+'; '.join(f'{q:%Y-%m} {v[0]:+.2f} ({v[1]:.2f})' for q,v in r[y].items()) for y in ['mark','b99'])); print(out[-1],flush=True)
lo10=lambda J: J.ddate.map(p10); last_=lambda J: J['last']; fifty=lambda J: 50.0
out.append('   observed only (no imputation): '); r0=es(d[base_cols].copy()); out[-1]+=' | '.join(f'{y} '+'; '.join(f'{q:%Y-%m} {v[0]:+.2f} ({v[1]:.2f})' for q,v in r0[y].items()) for y in ['mark','b99'])
scen('software leavers low (p10), others at last mark',lo10,last_)
scen('software leavers at last mark, others low (p10)',last_,lo10)
scen('software leavers at 50, others at last mark',fifty,last_)
scen('software leavers at last mark, others at 50',last_,fifty)
scen('worst case for H1 (software 110 / Below99 0; others 0 / 100)',lambda J:110.0,lambda J:0.0,0.0,100.0)
scen('worst case against H1 reversed (software 0 / 100; others 110 / 0)',lambda J:0.0,lambda J:110.0,100.0,0.0)
open(f'{V}/results_exit_bounds_2026-11-08.txt','w').write('\n'.join(out)+'\n'); print('\n'.join(out[:4]))
