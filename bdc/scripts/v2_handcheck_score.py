# Scores the 6 Nov 2026 hand-check (verdicts by reading both descriptions).
# borrower: Y same borrower, U unclear, N different. instrument: S same or no conflict, R revolver vs non-revolver,
# L lien conflict, T other tranche conflict (e.g. first-out vs third-out). Then: how much do gross N-PORT mismatches move
# the same-loan benchmark?
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..')))  # repo root
import numpy as np, pandas as pd
from scipy.stats import beta
V=(_ROOT+'/bdc/v2'); H=f'{V}/handcheck'
VER={'A':{'U':['A030'],'R':['A002','A031','A042','A096','A097','A099','A124','A157','A162','A182'],'L':['A154']},
     'B':{'U':['B042'],'T':['B107','B160']}}
NOTE={'A030':'Gateway Services Inc (Golub) vs Gateway US Holdings Inc (OHA): may be different companies','A154':'first-lien term loan vs second lien, same spread',
      'A097':'revolver marked 50.0 vs drawn term loan 85.9','A124':'revolver 86.4 vs term loan 100','B042':'Bingo Group Buyer Inc vs Bingo Holdings I LLC: group link not confirmed',
      'B107':'BDC tranche marked 82.3 vs N-PORT third-out tranche at 20.6 (rates within 0.5pp)','B160':'BDC first lien S+5.75 (rate not parsed, matched on maturity only) vs N-PORT second lien at 43.1'}
def cp(k,n):
    lo=beta.ppf(.025,k,n-k+1) if k>0 else 0.0; hi=beta.ppf(.975,k+1,n-k) if k<n else 1.0
    return lo,hi
out=[]
for s,f in [('A','A_crossfund_sample.csv'),('B','B_nport_sample.csv')]:
    d=pd.read_csv(f'{H}/{f}')
    d['borrower']=np.where(d.id.isin(VER[s].get('U',[])),'U',np.where(d.id.isin(VER[s].get('N',[])),'N','Y'))
    d['instrument']='S'
    for k in ['R','L','T']: d.loc[d.id.isin(VER[s].get(k,[])),'instrument']=k
    d['note']=d.id.map(NOTE).fillna('')
    d.to_csv(f'{H}/{f.replace("_sample","_verdicts")}',index=False)
    for g,x in [('all',d),('software',d[d.sw]),('other',d[~d.sw])]:
        n=len(x); kb=(x.borrower!='Y').sum(); ki=(x.instrument!='S').sum(); kany=((x.borrower!='Y')|(x.instrument!='S')).sum()
        out.append(dict(sample=s,group=g,n=n,wrong_or_unclear_borrower=kb,borrower_ci=('%.1f-%.1f%%'%tuple(100*np.array(cp(kb,n)))),
                        tranche_conflict=ki,tranche_ci=('%.1f-%.1f%%'%tuple(100*np.array(cp(ki,n)))),any_problem=kany))
R=pd.DataFrame(out); print(R.to_string(index=False))
# N-PORT benchmark sensitivity: drop positions whose BDC mark and N-PORT Level 2 price differ by more than X points at Sep 2025
# (measured before the shock, so the filter does not condition on the outcome)
p=pd.read_parquet(f'{V}/panel.parquet',columns=['pos_id','ddate','mark','ok','sector','pkey'])
M=pd.read_parquet(f'{V}/nport_matches.parquet'); M=M[M.verified&(M.lvl=='2')]
px=M.groupby('pos_id').price_median.median().rename('px')
q=p.join(px,on='pos_id'); q=q[q.px.notna()&q.ok&q.sector.notna()&~q.sector.isin(['it_services','tech_other'])]
q['sw']=q.sector.eq('software'); q['gap']=(q.mark-q.px).abs()
print('\nShare of matched L2 position-dates with |BDC mark - N-PORT price| > 10 / > 20:',
      q.groupby('sw').gap.apply(lambda s: f'{(s>10).mean():.1%} / {(s>20).mean():.1%}').to_dict())
w=q.pivot_table(index=['pkey','sw'],columns=q.ddate.dt.strftime('%Y-%m'),values=['mark','px']).dropna()
gap0=(w[('mark','2025-09')]-w[('px','2025-09')]).abs()
for thr in [None,10,5]:
    x=w if thr is None else w[gap0<=thr]
    line=[]
    for s,lab in [(False,'other'),(True,'software')]:
        y=x.xs(s,level='sw')
        line.append(f'{lab} n={len(y)}: Jun 2026 change mark {(y[("mark","2026-06")]-y[("mark","2025-09")]).mean():+.2f} | price {(y[("px","2026-06")]-y[("px","2025-09")]).mean():+.2f}')
    print(('all positions' if thr is None else f'Sep 2025 gap <= {thr}')+': '+'; '.join(line))
