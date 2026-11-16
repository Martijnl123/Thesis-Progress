# Code audit, 16 Nov 2026: integrity checks on the panel. Data to 30 Jun 2026 only.
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..')))  # repo root
import warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
V=(_ROOT+'/bdc/v2'); F=(_ROOT+'/bdc/industry_llm/full')
O=(_ROOT+'/bdc/audit'); base=pd.Timestamp('2025-09-30'); out=[]
def say(s): out.append(s); print(s,flush=True)
p=pd.read_parquet(f'{V}/panel.parquet'); p['cik']=p.cik.astype('int64')
P=pd.read_parquet(f'{V}/positions.parquet',columns=['pos_id','ident','fv','form','filed','period','adsh'])
p=p.merge(P,on='pos_id',how='left')
PB=pd.read_parquet(f'{F}/position_borrower.parquet'); L=pd.read_parquet(f'{F}/labels.parquet')[['bid','group']]
pb=PB.merge(L,on='bid').merge(p[['pos_id','pkey']],on='pos_id')
say(f'1. One row per position and date: duplicates of (pkey, ddate) {int(p.duplicated(["pkey","ddate"]).sum())} of {len(p)}')
nb=pb.groupby('pkey').bid.nunique(); say(f'2. Position keys linked to more than one borrower id: {int((nb>1).sum())} of {len(nb)}')
pbu=pb.drop_duplicates('pkey').set_index('pkey'); p['grp']=p.pkey.map(pbu.group); p['bid']=p.pkey.map(pbu.bid)
lab=p.grp.notna()&~p.grp.isin(['UNKN','FUND']); c=p[(p.ddate==base)&p.ok&lab].copy(); c['sw']=(c.grp=='SOFT').astype(int)
say(f'3. Cohort {len(c)} positions. Position_borrower rows cover dates: {sorted(p[p.pos_id.isin(PB.pos_id)].ddate.dt.strftime("%Y-%m").unique())[:3]} ...')
# 4 same fund, same date, same principal and same fair value under two keys (possible double counting)
dup=c.duplicated(['cik','principal','fv'],keep=False); say(f'4. Cohort positions sharing fund, principal and fair value with another cohort position: {int(dup.sum())} ({dup.mean():.2%}); software share among them {c[dup].sw.mean():.2f} vs {c.sw.mean():.2f} overall')
# 5 currency
fx=c.ident.str.contains(r'€|£|\bEUR\b|\bGBP\b|\bCAD\b|\bAUD\b|\bSEK\b|\bNOK\b|\bCHF\b|EURIBOR|SONIA|\bBBSY\b|CORRA|\bCDOR\b|STIBOR|NIBOR|\bBBSW\b|\bSARON\b|\bE \+|\bSN \+',regex=True,na=False)
say(f'5. Cohort positions whose description names a non-USD currency or reference rate: {int(fx.sum())} ({fx.mean():.2%}); software {c[c.sw==1].index.isin(c[fx].index).mean():.2%}, other {c[c.sw==0].index.isin(c[fx].index).mean():.2%}')
keys=set(c.pkey); fz=p[p.pkey.isin(keys)]; d=fz[(fz.ddate==base)|((fz.ddate>base)&fz.mark.between(0,110)&(fz.principal>=250e3))]
w=d.pivot_table(index='pkey',columns=d.ddate.dt.strftime('%Y%m'),values='mark'); info=c.set_index('pkey')[['cik','bid','sw','spread_pre','pik_pre']]
def xs(mask,q='202603'):
    x=pd.DataFrame({'dy':w[q]-w['202509']}).join(info).dropna(subset=['dy']); x=x[mask.reindex(x.index).fillna(False)]
    x['sp']=x.spread_pre.fillna(0)*100; x['spm']=x.spread_pre.isna().astype(int); x['pk']=x.pik_pre.fillna(0)
    m=smf.ols('dy~sw+sp+spm+pk+C(cik)',x).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(x.bid)[0]}); return f'{m.params.sw:+.2f} ({m.bse.sw:.2f}), n {len(x)}'
allm=pd.Series(True,index=c.pkey.values)
say(f'   March 2026 software effect (cross-section route): all {xs(allm)}; without those positions {xs(pd.Series(~fx.values,index=c.pkey.values))}; without the possible duplicates of check 4 {xs(pd.Series(~dup.values,index=c.pkey.values))}')
# 6 principal jumps (a changed principal with an unchanged fair value moves the mark mechanically)
pr=d.pivot_table(index='pkey',columns=d.ddate.dt.strftime('%Y%m'),values='principal'); r=(pr['202603']/pr['202509']).dropna()
big=(r>1.25)|(r<0.75); s=info.sw.reindex(r.index)
say(f'6. Principal changed by more than 25% from Sep 2025 to Mar 2026: software {big[s==1].mean():.2%}, other {big[s==0].mean():.2%}; effect without them {xs(pd.Series(~big.reindex(c.pkey.values).fillna(False).astype(bool).values,index=c.pkey.values))}')
# 7 marks above 101 or exactly 100 at Sep 2025
say(f'7. Sep 2025 marks: exactly 100.00 software {((c.mark-100).abs()<0.005)[c.sw==1].mean():.2%}, other {((c.mark-100).abs()<0.005)[c.sw==0].mean():.2%}; above 101 software {(c.mark>101)[c.sw==1].mean():.2%}, other {(c.mark>101)[c.sw==0].mean():.2%}')
# 8 which filing the Mar 2026 marks come from
m3=p[(p.ddate=='2026-03-31')&p.pkey.isin(keys)]; own=(pd.to_datetime(m3.period)==m3.ddate)
say(f'8. March 2026 rows of cohort positions taken from the fund\'s own filing for that quarter: {own.mean():.2%}; forms {m3.form.value_counts().head(4).to_dict()}')
# 9 chain breaks: cohort positions not found in Dec 2025 whose fund still reports, by layout
g=fz[fz.ddate=='2025-12-31']; lost=c[~c.pkey.isin(set(g.pkey))&c.cik.isin(set(p[p.ddate=='2025-12-31'].cik))]
say(f'9. Cohort positions not found in Dec 2025 though the fund reports: {len(lost)} ({len(lost)/len(c):.2%}); software {lost.sw.mean():.2f} of them vs {c.sw.mean():.2f} of the cohort; their Sep 2025 mean mark {lost.mark.mean():.2f} vs {c.mark.mean():.2f}; below 99: {100*(lost.mark<99).mean():.1f}% vs {100*(c.mark<99).mean():.1f}%')
# 10 label coverage: who is outside the cohort because the borrower was unknown or not in the run
a=p[(p.ddate==base)&p.ok]; a['st']=np.where(a.grp.isna(),'not in run',np.where(a.grp=='UNKN','unknown',np.where(a.grp=='FUND','fund','labelled')))
m3k=p[(p.ddate=='2026-03-31')&p.mark.between(0,110)].set_index('pkey').mark; a['d']=a.pkey.map(m3k)-a.mark
t=a.groupby('st').agg(n=('pkey','size'),mark=('mark','mean'),d=('d','mean'),fundsw=('sector',lambda s:(s=='software').mean()))
say('10. Clean Sep 2025 positions by label status: '+'; '.join(f'{i}: n {int(r.n)}, change to Mar 2026 {r.d:+.2f}, fund-label software share {r.fundsw:.2f}' for i,r in t.iterrows()))
x=a[a.st.isin(['not in run','unknown'])&a.sector.notna()&~a.sector.isin(['it_services','tech_other'])].copy(); x['sw']=(x.sector=='software').astype(int); x=x.dropna(subset=['d'])
m=smf.ols('d~sw+C(cik)',x).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(x.loan)[0]})
say(f'    positions OUTSIDE the cohort (borrower unknown or not in the run) with a fund label: software effect in March 2026 {m.params.sw:+.2f} ({m.bse.sw:.2f}), n {len(x)}, software {int(x.sw.sum())}')
open(f'{O}/results_a02_integrity_2026-11-16.txt','w').write('\n'.join(out)+'\n')
