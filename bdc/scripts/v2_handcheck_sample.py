# Hand-check samples (6 Nov 2026): random draws of the links that drive results, for a false-match rate.
# A: cross-fund same-loan pairs used in H3 (pairs_0.5 with cos>=0.6, different families, Sep 2025-Jun 2026).
# B: N-PORT matches used in the same-loan benchmark (verified, Level 2, Sep 2025-Jun 2026).
# Each sample: 100 software + 100 other, seed 20261106. Verdicts are added by hand in *_verdicts.csv.
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..')))  # repo root
import sys, numpy as np, pandas as pd
sys.path.insert(0,(_ROOT+'/bdc/scripts'))
from matchlib import clean_text
V=(_ROOT+'/bdc/v2'); OUT=f'{V}/handcheck'
WIN=pd.to_datetime(['2025-09-30','2025-12-31','2026-03-31','2026-06-30'])
p=pd.read_parquet(f'{V}/panel.parquet').set_index('pos_id')
pos=pd.read_parquet(f'{V}/positions.parquet',columns=['pos_id','ident','maturity','rate','spread']).set_index('pos_id')
rng=np.random.default_rng(20261106)
# A
pr=pd.read_parquet(f'{V}/pairs_0.5.parquet'); pr=pr[(pr.cos>=0.6)&(pr.pa<pr.pc)]
ok=p.ok; pr=pr[ok.reindex(pr.pa).values & ok.reindex(pr.pc).values]
for s,c in [('a','pa'),('c','pc')]:
    for col in ['family','ddate','sector','mark']: pr[f'{col}_{s}']=p[col].reindex(pr[c]).values
    for col in ['ident','maturity','spread','rate']: pr[f'{col}_{s}']=pos[col].reindex(pr[c]).values
pr['sector']=pr.sector_a.fillna(pr.sector_c)
pr=pr[pr.sector.notna()&~pr.sector.isin(['it_services','tech_other'])&(pr.family_a!=pr.family_c)&pr.ddate_a.isin(WIN)]
print('A universe: pairs', len(pr), 'software share', round(pr.sector.eq('software').mean(),3))
print(pr.cos.describe().round(3).to_string())
A=pd.concat([g.sample(100,random_state=20261106) for _,g in pr.groupby(pr.sector.eq('software'))]).reset_index(drop=True)
A['sw']=A.sector.eq('software'); A['id']=['A%03d'%i for i in range(len(A))]
A[['id','sw','cos','ddate_a','family_a','ident_a','maturity_a','spread_a','mark_a','family_c','ident_c','maturity_c','spread_c','mark_c','pa','pc']].to_csv(f'{OUT}/A_crossfund_sample.csv',index=False)
# B
M=pd.read_parquet(f'{V}/nport_matches.parquet'); M=M[M.verified&(M.lvl=='2')&M.ddate.isin(WIN)]
M['sector']=p.sector.reindex(M.pos_id).values; M=M[M.sector.notna()]
n=pd.read_parquet(f'{V}/nport_qe_loans.parquet',columns=['report_date','issuer_name','issuer_title','maturity','rate','lvl','price_median','n_series']).rename(columns={'issuer_title':'title'})
n['nm_b']=n.issuer_name.fillna('').map(clean_text)
n=n.rename(columns={'report_date':'ddate','maturity':'mat_n','rate':'rate_n'}).drop_duplicates(['ddate','nm_b','mat_n','rate_n','lvl','price_median'])
print('B universe: matches', len(M), 'positions', M.pos_id.nunique(), 'software share', round(M.sector.eq('software').mean(),3))
B=pd.concat([g.sample(100,random_state=20261106) for _,g in M.groupby(M.sector.eq('software'))]).reset_index(drop=True)
B=B.merge(n[['ddate','nm_b','mat_n','rate_n','lvl','price_median','issuer_name','title']],on=['ddate','nm_b','mat_n','rate_n','lvl','price_median'],how='left').drop_duplicates(['pos_id','nm_b','mat_n','rate_n','ddate','price_median'])
B['sw']=B.sector.eq('software'); B['id']=['B%03d'%i for i in range(len(B))]
B['ident']=pos.ident.reindex(B.pos_id).values; B['mark']=p.mark.reindex(B.pos_id).values; B['family']=p.family.reindex(B.pos_id).values
B[['id','sw','cos','ddate','family','ident','maturity','rate','mark','issuer_name','title','mat_n','rate_n','price_median','pos_id']].to_csv(f'{OUT}/B_nport_sample.csv',index=False)
print('samples', len(A), len(B), 'B missing N-PORT title', int(B.title.isna().sum()))
