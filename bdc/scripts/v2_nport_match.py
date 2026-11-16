# N-PORT coverage check (26 Oct 2026): which BDC loans are also held by registered funds (mutual funds, ETFs, interval
# funds) on the same quarter-end, and at what price. Input: nport_qe_loans.csv.gz, built on a local PC from the SEC Form
# N-PORT data sets (asset category LON, report dates on the five quarter-ends, each fund's latest filing, USD principal
# amounts), aggregated to one row per date, fair value level, issuer, title, maturity and coupon.
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..')))  # repo root
import sys, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.insert(0,(_ROOT+'/bdc/scripts'))
from matchlib import clean_text
V=(_ROOT+'/bdc/v2')
QE=pd.to_datetime(['2025-06-30','2025-09-30','2025-12-31','2026-03-31','2026-06-30'])
def dates(s):
    d=pd.to_datetime(s,format='%d-%b-%Y',errors='coerce')
    return d.fillna(pd.to_datetime(s,errors='coerce'))
n=pd.read_csv(sys.argv[1],dtype=str)
n['report_date']=dates(n.report_date); n=n[n.report_date.isin(QE)].copy()
for c in ['annualized_rate','n_series','total_balance','total_value','price_median','price_min','price_max','n_default']: n[c]=pd.to_numeric(n[c],errors='coerce')
n['maturity']=dates(n.maturity_date)
n['rate']=np.where(n.annualized_rate>1,n.annualized_rate/100,n.annualized_rate)
n['lvl']=n.fair_value_level.fillna('NA').str.strip()
print('N-PORT quarter-end loan lines:', len(n)); print(n.groupby([n.report_date.dt.strftime('%Y-%m'),'lvl']).n_series.agg(['size','sum']).unstack().to_string(), flush=True)
n.to_parquet(f'{V}/nport_qe_loans.parquet',index=False)

p=pd.read_parquet(f'{V}/panel.parquet',columns=['pos_id','cik','family','ddate','mark','ok','sector','loan','pkey'])
p=p.merge(pd.read_parquet(f'{V}/positions.parquet',columns=['pos_id','ident','maturity','rate']),on='pos_id')
p=p[p.ddate.isin(QE)&p.ok&p.sector.notna()&~p.sector.isin(['it_services','tech_other'])].copy()
p['sw']=p.sector.eq('software')
p['nm']=p.ident.map(clean_text); n['nm']=n.issuer_name.fillna('').map(clean_text)
rows=[]
for d in QE:
    a=p[p.ddate==d]; b=n[n.report_date==d]
    ua=a.nm[a.nm.str.len()>0].drop_duplicates().reset_index(drop=True); ub=b.nm[b.nm.str.len()>0].drop_duplicates().reset_index(drop=True)
    # drop very common words (inc, llc, holdings...) so only pairs sharing a distinctive name word are scored
    vec=TfidfVectorizer(token_pattern=r'(?u)\b[a-z&]{2,}\b',sublinear_tf=True,max_df=0.02).fit(pd.concat([ua,ub]))
    XB=vec.transform(ub).T.tocsr(); cand=[]
    for i in range(0,len(ua),2000):
        S=(vec.transform(ua.iloc[i:i+2000])@XB).tocoo(); k=S.data>=0.6
        cand.append(pd.DataFrame({'nm':ua.values[i+S.row[k]],'nm_b':ub.values[S.col[k]],'cos':S.data[k]}))
    cand=pd.concat(cand,ignore_index=True)
    m=a[['pos_id','nm','maturity','rate']].merge(cand,on='nm').merge(
        b[['nm','maturity','rate','price_median','lvl','n_series']].rename(columns={'nm':'nm_b','maturity':'mat_n','rate':'rate_n'}),on='nm_b')
    gap=(m.maturity-m.mat_n).abs().dt.days; dr=(m.rate-m.rate_n).abs()
    # reject clear conflicts; a match is "verified" when maturity (within 62 days) or coupon (within 0.5 pp) agrees
    m=m[~(gap>62) & ~(dr>0.005)].copy()
    m['verified']=(gap.reindex(m.index)<=62)|(dr.reindex(m.index)<=0.005)
    m['ddate']=d; rows.append(m)
    print(d.date(), 'BDC positions', len(a), 'N-PORT lines', len(b), 'candidate name pairs', len(cand), 'kept', len(m), 'verified', int(m.verified.sum()), flush=True)
M=pd.concat(rows,ignore_index=True); M.to_parquet(f'{V}/nport_matches.parquet',index=False)
print('positions matched on name only (maturity and coupon missing on one side):', M[~M.verified].pos_id.nunique(), '| verified:', M[M.verified].pos_id.nunique(), flush=True)
M=M[M.verified]
g=M.groupby('pos_id').agg(holders=('n_series','sum'))
g['l2']=M[M.lvl=='2'].groupby('pos_id').n_series.sum()
g['px_l2']=M[M.lvl=='2'].groupby('pos_id').price_median.median()
g['px_l3']=M[M.lvl=='3'].groupby('pos_id').price_median.median()
p=p.join(g,on='pos_id'); p['matched']=p.holders.notna(); p['has_l2']=p.l2.notna(); p['has_l3']=p.px_l3.notna()
def cov(x):
    lo=x.groupby('loan')
    return pd.Series({'positions':len(x),'loans':x.loan.nunique(),'any fund %':100*x.matched.mean(),'L2 %':100*x.has_l2.mean(),'L3 %':100*x.has_l3.mean(),
                      'loans any %':100*lo.matched.any().mean(),'loans L2 %':100*lo.has_l2.any().mean()})
grp=np.where(p.sw,'software','other')
print('\nBDC clean, labelled positions held by at least one N-PORT fund on the same quarter-end'); print(p.groupby([p.ddate.dt.strftime('%Y-%m'),grp]).apply(cov).round(1).to_string(), flush=True)
for col,lab in [('px_l2','Level 2 (market) price'),('px_l3','Level 3 (fund mark)')]:
    q=p[p[col].notna()].copy(); q['diff']=q.mark-q[col]
    print(f'\nBDC mark minus N-PORT {lab}, same loan and date')
    print(q.groupby([q.ddate.dt.strftime('%Y-%m'),np.where(q.sw,'software','other')]).agg(n=('diff','size'),median_diff=('diff','median'),
          within_1=('diff',lambda s:(s.abs()<=1).mean()),over_5=('diff',lambda s:(s.abs()>5).mean()),bdc=('mark','median'),nport=(col,'median')).round(2).to_string(), flush=True)
q=p[p.px_l2.notna()]
w=q.pivot_table(index=['pkey','sw'],columns=q.ddate.dt.strftime('%Y-%m'),values=['mark','px_l2']).dropna()
print('\nPositions with a Level 2 price at every quarter-end: mean change since Sep 2025, BDC mark | market price')
for s,lab in [(False,'other'),(True,'software')]:
    if s not in w.index.get_level_values('sw'): continue
    x=w.xs(s,level='sw')
    print(f'  {lab} ({len(x)}): '+'; '.join(f'{c} {(x[("mark",c)]-x[("mark","2025-09")]).mean():+.2f} | {(x[("px_l2",c)]-x[("px_l2","2025-09")]).mean():+.2f}' for c in ['2025-06','2025-12','2026-03','2026-06']), flush=True)
