# Chain-linked event-time index and Rambachan-Roth (2023) relative-magnitude bounds (PAP v5 section 4 'Pre-trends';
#), 8 Nov 2026. Quarters to 30 Jun 2026 only.
# Index: for each quarter t, positions valid at t-1 and t (gap 80-100 days); c_t = software minus other change in y within
# fund (fund x quarter FE). theta_k = cumulated c from the base Sep 2025 (pre quarters: minus the changes up to Sep 2025).
# Two label versions: funds' own labels (all positions with a fund label, excl. it_services/tech_other; NOT restricted to
# borrowers held at Sep 2025) and model labels (positions of labelled borrowers, excl. ITSV/DATA/HARD/UNKN/FUND).
# Note: the panel's fund label is voted over all dates, not up to Sep 2025 as the PAP requires before the Sep data.
# Bootstrap: 300 block draws. Blocks = borrower (model labels) or cross-fund loan id (fund labels; no borrower id exists
# for positions outside the labelling run, so tranches of one borrower are separate blocks here).
# RR relative magnitudes: |delta_{s+1}-delta_s| <= Mbar * D for post s, D = max pre-period |change in theta|; with delta=0 at
# base, tau_k in [theta_k - k*Mbar*D, theta_k + k*Mbar*D], k = quarters after Sep 2025. Robust 90% interval: 5th percentile
# of the bootstrap lower bound, 95th of the upper bound (a simple percentile version, not the HonestDiD conditional test).
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..')))  # repo root
import numpy as np, pandas as pd
V=(_ROOT+'/bdc/v2'); F=(_ROOT+'/bdc/industry_llm/full')
p=pd.read_parquet(f'{V}/panel.parquet'); assert p.ddate.max()<=pd.Timestamp('2026-06-30')
L=pd.read_parquet(f'{F}/labels.parquet')[['bid','group']]
pb=pd.read_parquet(f'{F}/position_borrower.parquet').merge(L,on='bid').merge(p[['pos_id','pkey']],on='pos_id').drop_duplicates('pkey').set_index('pkey')
p['grp']=p.pkey.map(pb.group); p['bid']=p.pkey.map(pb.bid)
base=pd.Timestamp('2025-09-30')
ever_ok=set(p.loc[p.ok,'pkey'])
p['valid']=np.where(p.ddate<=base,p.ok,p.mark.between(0,110)&(p.principal>=250e3)&p.pkey.isin(ever_ok))
v=p[p.valid].sort_values(['pkey','ddate']).copy(); v['b99']=100*(v.mark<99)
g=v.groupby('pkey'); v['dmark']=g.mark.diff(); v['db99']=g.b99.diff(); v['gap']=g.ddate.diff().dt.days
ch=v[v.gap.between(80,100)].copy()
qs=sorted(ch.ddate.unique())
def c_t(x,y):
    # within-fund difference: OLS of dy on soft with fund FE in one quarter
    s=x.sw-x.groupby('cik').sw.transform('mean'); e=x[y]-x.groupby('cik')[y].transform('mean')
    den=(s*s).sum(); return (s*e).sum()/den if den>0 else np.nan
def index(x,y):
    c=pd.Series({q:c_t(x[x.ddate==q],y) for q in qs})
    th={}
    for q in qs:  # q is the END quarter of a change
        th[q]= c[(c.index>base)&(c.index<=q)].sum() if q>base else np.nan
    pre=[q for q in qs if q<=base]
    th_pre={}
    # theta at quarter k (k<base) = -sum of changes ending in (k, base]
    allq=sorted(set(qs)|{qs[0]-pd.offsets.QuarterEnd()})
    for k in allq:
        if k<base: th_pre[k]=-c[(c.index>k)&(c.index<=base)].sum()
    th={**th_pre,**{q:t for q,t in th.items() if q>base}}; th[base]=0.0
    return pd.Series(th).sort_index(), c
def rr(theta,Mbars):
    pre=theta[theta.index<=base]; D=pre.diff().abs().max()
    post=theta[theta.index>base]; res={}
    for k,(q,t) in enumerate(post.items(),start=1):
        res[q]={M:(t-k*M*D,t+k*M*D) for M in Mbars}
    return D,res
Mbars=[0,0.5,1,1.5,2]
out=[]
versions={'fund labels':(ch[ch.sector.notna()&~ch.sector.isin(['it_services','tech_other'])].assign(sw=lambda x:(x.sector=='software').astype(float)),'loan'),
          'model labels':(ch[ch.grp.notna()&~ch.grp.isin(['ITSV','DATA','HARD','UNKN','FUND'])].assign(sw=lambda x:(x.grp=='SOFT').astype(float)),'bid')}
rng=np.random.default_rng(20261108)
for nm,(x,blk) in versions.items():
    out.append(f'\n=== {nm}: {len(x)} position-quarter changes, {x.pkey.nunique()} positions, {x[blk].nunique()} blocks ({blk})')
    for y in ['dmark','db99']:
        th,c=index(x,y); D,res=rr(th,Mbars)
        B=[]; blocks=x[blk].dropna().unique(); idx=x.groupby(blk).indices
        for b in range(300):
            draw=rng.choice(blocks,len(blocks),replace=True)
            xb=x.iloc[np.concatenate([idx[k] for k in draw])]
            tb,_=index(xb,y); B.append(tb)
        B=pd.DataFrame(B); sd=B.std()
        out.append(f'{y}: theta (bootstrap SE) '+'; '.join(f'{q:%Y-%m} {th[q]:+.2f} ({sd[q]:.2f})' for q in th.index))
        out.append(f'   D = max pre-period |change| = {D:.2f}; post-period bounds [lower, upper] and robust 90% interval:')
        for q,bd in res.items():
            parts=[]
            for M in Mbars:
                lo_b=[]; hi_b=[]
                for _,tb in B.iterrows():
                    Db=tb[tb.index<=base].diff().abs().max(); k=list(th.index[th.index>base]).index(q)+1
                    lo_b.append(tb[q]-k*M*Db); hi_b.append(tb[q]+k*M*Db)
                parts.append(f'M={M}: [{bd[M][0]:+.2f}, {bd[M][1]:+.2f}] CI [{np.percentile(lo_b,5):+.2f}, {np.percentile(hi_b,95):+.2f}]')
            out.append(f'   {q:%Y-%m}: '+' | '.join(parts))
        # breakdown Mbar for Mar and Jun 2026: largest M with robust CI excluding 0
        for q in [pd.Timestamp('2026-03-31'),pd.Timestamp('2026-06-30')]:
            k=list(th.index[th.index>base]).index(q)+1; sgn=np.sign(th[q]); brk=None
            for M in np.arange(0,5.01,0.05):
                lo=[tb[q]-k*M*tb[tb.index<=base].diff().abs().max() for _,tb in B.iterrows()]
                hi=[tb[q]+k*M*tb[tb.index<=base].diff().abs().max() for _,tb in B.iterrows()]
                ok=(np.percentile(lo,5)>0) if sgn>0 else (np.percentile(hi,95)<0)
                if ok: brk=M
                else: break
            out.append(f'   breakdown Mbar {q:%Y-%m}: {brk if brk is None else round(brk,2)}')
        print('\n'.join(out[-8:]),flush=True)
open(f'{V}/results_chainlinked_rr_2026-11-08.txt','w').write('\n'.join(out)+'\n')
