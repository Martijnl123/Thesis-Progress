# Code audit, 16 Nov 2026: who drives the traded / no-market-price split (March 2026, cross-section route). Data to 30 Jun 2026.
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..')))  # repo root
import warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
V=(_ROOT+'/bdc/v2'); O=(_ROOT+'/bdc/audit'); out=[]
def say(s): out.append(s); print(s,flush=True)
T=pd.read_parquet(f'{V}/traded_untraded_sep25_2026-11-13.parquet'); T['cik']=T.cik.astype('int64')
T['g']=np.where(T.cls=='traded','traded',np.where(T.cls.isin(['untraded','L3 only','not matched']),'noprice','x'))
def fit(x,y='dm_2026-03'):
    x=x.dropna(subset=[y]).rename(columns={y:'dy'}); m=smf.ols('dy~sw+spr_f+spr_m+pikp+C(cik)',x).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(x.bid)[0]}); return m.params.sw,m.bse.sw,len(x)
for g in ['traded','noprice']:
    x=T[T.g==g]; b,s,n=fit(x); say(f'{g}: software effect Mar 2026 {b:+.2f} ({s:.2f}), n {n}; software borrowers {x[x.sw==1].bid.nunique()}')
    sb=x[(x.sw==1)].groupby('bid')['dm_2026-03'].agg(['mean','size']).dropna().sort_values('mean')
    for k in [5,10,20]:
        b2,s2,n2=fit(x[~x.bid.isin(sb.index[:k])]); say(f'   without the {k} software borrowers with the largest falls: {b2:+.2f} ({s2:.2f})')
    xw=x.copy(); lo,hi=xw['dm_2026-03'].quantile([.01,.99]); xw['dm_2026-03']=xw['dm_2026-03'].clip(lo,hi); b3,s3,_=fit(xw); say(f'   changes winsorised at 1/99%: {b3:+.2f} ({s3:.2f})')
    xm=x.dropna(subset=['dm_2026-03']); say(f'   median change: software {xm[xm.sw==1]["dm_2026-03"].median():+.2f}, other {xm[xm.sw==0]["dm_2026-03"].median():+.2f}; share of software positions down more than 2 points: {(xm[xm.sw==1]["dm_2026-03"]<-2).mean():.2f}')
    per=[]
    for fam,gg in xm.groupby('family'):
        if gg.sw.sum()>=8 and (1-gg.sw).sum()>=8:
            dd=gg['dm_2026-03']-gg.groupby('cik')['dm_2026-03'].transform('mean'); ss=gg.sw-gg.groupby('cik').sw.transform('mean'); per.append((dd*ss).sum()/(ss**2).sum())
    per=pd.Series(per); say(f'   managers with 8+ software and 8+ other loans in this group: {len(per)}; negative in {int((per<0).sum())}; median {per.median():+.2f}')
# same borrower size: large private borrowers vs traded
big=T[(T.g=='noprice')&(T.ln_bsize>=T[T.g=='traded'].ln_bsize.median())]; b,s,n=fit(big); say(f'no market price, borrowers at least as large as the median traded borrower: {b:+.2f} ({s:.2f}), n {n}, software borrowers {big[big.sw==1].bid.nunique()}')
many=T[(T.g=='noprice')&(T.nfam>=4)]; b,s,n=fit(many); say(f'no market price, loans of borrowers held by 4+ managers: {b:+.2f} ({s:.2f}), n {n}')
open(f'{O}/results_a03_step3_drivers_2026-11-16.txt','w').write('\n'.join(out)+'\n')
