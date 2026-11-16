# Clean the SEC schedule rows into debt positions (v2).
# Change from clean.py: for each fund and date keep the filing that reports the schedule most completely
# (most distinct usable debt rows), preferring the fund's own filing for that period, then the latest.
# clean.py kept the latest filing with any row at that date; that is often a later filing repeating a few
# roll-forward rows, so whole schedules were lost (Dec 2023 and Dec 2024 for 33 funds, 10-40% of rows before Sep 2025).
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..')))  # repo root
import os, sys, pandas as pd, numpy as np
sys.path.insert(0,(_ROOT+'/bdc/scripts'))
from families import classify
src,out=sys.argv[1],sys.argv[2]
d=pd.read_parquet(src)
d=d[d.ident.notna()].copy()
for c in ['ddate','filed','period']: d[c]=pd.to_datetime(d[c],errors='coerce')
d=d[d.ddate.notna()]
for c in ['spread','rate','pik_rate','floor']:
    x=d[c]; d[c]=np.where(x>1, x/100, x)
use=(d.spread.notna()|d.rate.notna())&(d.principal>0)&(d.fv>0)
u=d[use].drop_duplicates(['cik','ddate','adsh','ident','principal','fv'])
f=u.groupby(['cik','ddate','adsh']).agg(n=('ident','size'),filed=('filed','max'),period=('period','first')).reset_index()
f['own']=(f.period==f.ddate).astype(int)
f=f.sort_values(['cik','ddate','n','own','filed'],ascending=[True,True,False,False,False])
pick=f.drop_duplicates(['cik','ddate'])
print('fund-dates', len(pick), '| picked filing is the own-period filing:', round(pick.own.mean(),3))
d=d.merge(pick[['cik','ddate','adsh','own']],on=['cik','ddate','adsh'])
fam=d.bdc.map(lambda n: classify(n)); d['family']=fam.str[0]; d['ptype']=fam.str[1]
d=d[d.ptype!='exclude']
d['maturity']=pd.to_datetime(d.maturity,errors='coerce')
debt=d[(d.spread.notna()|d.rate.notna()) & (d.principal>0) & (d.fv>0)].copy()
debt=debt.drop_duplicates(subset=['cik','ddate','ident','principal','fv'])
debt['mark']=debt.fv/debt.principal*100
txt=debt.ident.str.lower()
debt['first_lien']=txt.str.contains(r'first lien|1st lien|senior secured|one stop|unitranche|first-lien').astype(int)
debt['second_lien']=txt.str.contains(r'second lien|2nd lien|second-lien|junior|subordinat|mezz').astype(int)
debt['software']=(debt.industry.fillna('')+' '+txt).str.contains(r'software|saas|information technology|it services',case=False).astype(int)
debt['pik']=((debt.pik_rate.fillna(0)>0)|txt.str.contains(r'\bpik\b|paid[- ]in[- ]kind')).astype(int)
debt['nonaccrual']=(debt.nonincome.fillna('').str.lower().eq('true')|txt.str.contains('non-accrual|nonaccrual')).astype(int)
print('debt positions', len(debt), 'BDCs', debt.cik.nunique())
debt.to_parquet(out,index=False)
OLD='/tmp/claude-0/-home-claude/dc0028bd-8cc6-5691-8e20-9ef3d2183413/scratchpad/bdc/debt.parquet'
if not os.path.exists(OLD): sys.exit(0)   # 7 Nov 2026: the comparison with the pre-audit file runs only where that file exists
old=pd.read_parquet(OLD,columns=['cik','ddate','principal'])
q=lambda x: x[x.ddate.dt.month.isin([3,6,9,12])&(x.ddate>='2022-12-31')]
a=q(old).groupby(old.ddate.dt.strftime('%Y-%m')).agg(n_old=('cik','size'),funds_old=('cik','nunique'),bn_old=('principal',lambda s:s.sum()/1e9))
b=q(debt).groupby(debt.ddate.dt.strftime('%Y-%m')).agg(n_new=('cik','size'),funds_new=('cik','nunique'),bn_new=('principal',lambda s:s.sum()/1e9))
print(a.join(b,how='outer').round(1).to_string())
