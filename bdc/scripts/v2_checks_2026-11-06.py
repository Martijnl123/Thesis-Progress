# Quick checks run on 6 Nov 2026 (common shock, December audit timing, PIK anticipation, stale N-PORT prices).
# Saved from scratch scripts so every number in the chapters has a script and an output file: bdc/v2/results_checks_2026-11-06.txt.
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..')))  # repo root

print('\n===== #31 common shock: software quarterly change on other loans change, pre-period slope and prediction =====')
import numpy as np, pandas as pd
V=(_ROOT+'/bdc/v2'); F=(_ROOT+'/bdc/industry_llm/full')
p=pd.read_parquet(f'{V}/panel.parquet')
L=pd.read_parquet(f'{F}/labels.parquet')[['bid','group']]
pb=pd.read_parquet(f'{F}/position_borrower.parquet').merge(L,on='bid').merge(p[['pos_id','pkey']],on='pos_id').drop_duplicates('pkey').set_index('pkey')
p['grp']=p.pkey.map(pb.group)
p=p[p.grp.notna()&~p.grp.isin(['UNKN','FUND'])]
p=p[(p.mark.between(0,110))&(p.principal>=250e3)]
p['b99']=100*(p.mark<99); p['sw']=(p.grp=='SOFT')
p=p.sort_values(['pkey','ddate'])
qs=sorted(p.ddate.unique()); nxt=dict(zip(qs[:-1],qs[1:]))
a=p[['pkey','cik','ddate','b99','mark','sw']].copy(); a['dn']=a.ddate.map(nxt)
b=p[['pkey','ddate','b99','mark']].rename(columns={'ddate':'dn','b99':'b99n','mark':'markn'})
c=a.merge(b,on=['pkey','dn'])
c['db99']=c.b99n-c.b99; c['dmark']=c.markn-c.mark
# within-fund: demean by fund-quarter, then software minus other
rows=[]
for dn,g in c.groupby('dn'):
    for y in ['db99','dmark']:
        g[y+'_w']=g[y]-g.groupby('cik')[y].transform('mean')
    fq=g.groupby(['cik','sw'])[['db99','dmark']].mean().unstack()
    rows.append(dict(q=pd.Timestamp(dn).date(), sw_b99=g.loc[g.sw,'db99'].mean(), ot_b99=g.loc[~g.sw,'db99'].mean(),
                     sw_mark=g.loc[g.sw,'dmark'].mean(), ot_mark=g.loc[~g.sw,'dmark'].mean(), n=len(g)))
r=pd.DataFrame(rows); print(r.round(2).to_string())
pre=r[r.q<=pd.Timestamp('2025-09-30').date()]
for y in ['b99','mark']:
    x=pre['ot_'+y].values; z=pre['sw_'+y].values
    X=np.c_[np.ones(len(x)),x]; bta=np.linalg.lstsq(X,z,rcond=None)[0]
    res=z-X@bta; s2=res@res/(len(x)-2); se=np.sqrt(np.diag(s2*np.linalg.inv(X.T@X)))
    post=r[r.q>pd.Timestamp('2025-09-30').date()]
    print(y,'pre-period software-on-other slope', bta.round(2), 'se', se.round(2), 'n', len(x))
    for _,q in post.iterrows():
        pred=bta[0]+bta[1]*q['ot_'+y]
        print('   ',q.q,'other',round(q['ot_'+y],2),'software actual',round(q['sw_'+y],2),'predicted from slope',round(pred,2), ' slope+2se pred', round(bta[0]+(bta[1]+2*se[1])*q['ot_'+y],2))

print('\n===== #25 December 2025 audit timing: late vs early 10-K filers =====')
import numpy as np, pandas as pd
V=(_ROOT+'/bdc/v2'); F=(_ROOT+'/bdc/industry_llm/full')
p=pd.read_parquet(f'{V}/panel.parquet')
L=pd.read_parquet(f'{F}/labels.parquet')[['bid','group']]
pb=pd.read_parquet(f'{F}/position_borrower.parquet').merge(L,on='bid').merge(p[['pos_id','pkey']],on='pos_id').drop_duplicates('pkey').set_index('pkey')
p['grp']=p.pkey.map(pb.group)
lab=p.grp.notna()&~p.grp.isin(['UNKN','FUND'])
keys=set(p.loc[(p.ddate=='2025-09-30')&p.ok&lab,'pkey'])
a=p[p.pkey.isin(keys)&(p.ddate=='2025-09-30')][['pkey','cik','mark','grp']]
b=p[p.pkey.isin(keys)&(p.ddate=='2025-12-31')&p.mark.between(0,110)&(p.principal>=250e3)][['pkey','mark']]
c=a.merge(b,on='pkey',suffixes=('0','1')); c['sw']=c.grp=='SOFT'
c['db']=100*((c.mark1<99).astype(int)-(c.mark0<99).astype(int)); c['dm']=c.mark1-c.mark0
pos=pd.read_parquet(f'{V}/positions.parquet',columns=['cik','ddate','form','filed'])
f=pos[(pos.ddate=='2025-12-31')&pos.form.str.startswith('10-K',na=False)].drop_duplicates('cik').set_index('cik').filed
f=pd.to_datetime(f.astype(str))
c=c[c.cik.isin(f.index)]; c['filed']=c.cik.map(f)
c['late']=c.filed>=pd.Timestamp('2026-03-09')
c['dbw']=c.db-c.groupby('cik').db.transform('mean'); c['dmw']=c.dm-c.groupby('cik').dm.transform('mean')
for late,g in c.groupby('late'):
    print('filed on/after 9 Mar' if late else 'filed before 9 Mar', 'funds', g.cik.nunique(), 'n', len(g),
          '| within-fund sw-minus-other: Below99 change', round(g[g.sw].dbw.mean()-g[~g.sw].dbw.mean(),2), 'mark change', round(g[g.sw].dmw.mean()-g[~g.sw].dmw.mean(),3))
# OLS with interaction on fund-demeaned changes, SEs clustered by fund
import statsmodels.formula.api as smf
c['swl']=c.sw.astype(int)*c.late.astype(int)
for y in ['dbw','dmw']:
    r=smf.ols(f'{y}~sw+swl',data=c.assign(sw=c.sw.astype(int))).fit(cov_type='cluster',cov_kwds={'groups':c.cik})
    print(y, r.params.round(3).to_dict(), 'se', r.bse.round(3).to_dict())

print('\n===== #58 new PIK conversions, software vs other =====')
import numpy as np, pandas as pd
V=(_ROOT+'/bdc/v2'); F=(_ROOT+'/bdc/industry_llm/full')
p=pd.read_parquet(f'{V}/panel.parquet')
L=pd.read_parquet(f'{F}/labels.parquet')[['bid','group']]
pb=pd.read_parquet(f'{F}/position_borrower.parquet').merge(L,on='bid').merge(p[['pos_id','pkey']],on='pos_id').drop_duplicates('pkey').set_index('pkey')
p['grp']=p.pkey.map(pb.group)
p=p[p.grp.notna()&~p.grp.isin(['UNKN','FUND'])&p.pik.notna()]
p=p.sort_values(['pkey','ddate']); p['pik_prev']=p.groupby('pkey').pik.shift(); p['dprev']=p.groupby('pkey').ddate.shift()
x=p[p.pik_prev.notna()&(p.pik_prev==0)]
x=x.assign(sw=x.grp=='SOFT', new=x.pik.astype(float))
t=x.groupby(['ddate','sw']).new.mean().unstack().mul(100).round(2); t.columns=['other','software']; t['diff']=t.software-t.other
print('new PIK conversions (% of cash-pay positions last quarter):'); print(t.tail(10).to_string())

print('\n===== #61 N-PORT Level 2 price unchanged from previous quarter =====')
import pandas as pd, numpy as np
V=(_ROOT+'/bdc/v2'); F=(_ROOT+'/bdc/industry_llm/full')
n=pd.read_parquet(f'{V}/nport_matches.parquet')
n=n[(n.verified==True)&(n.lvl.astype(str)=='2')]
p=pd.read_parquet(f'{V}/panel.parquet',columns=['pos_id','pkey','ddate'])
L=pd.read_parquet(f'{F}/labels.parquet')[['bid','group']]
pb=pd.read_parquet(f'{F}/position_borrower.parquet').merge(L,on='bid').merge(p[['pos_id','pkey']],on='pos_id').drop_duplicates('pkey').set_index('pkey')
p['grp']=p.pkey.map(pb.group)
x=n.groupby(['pos_id','ddate']).price_median.median().reset_index().merge(p,on=['pos_id','ddate'])
x=x.dropna(subset=['grp'])
x=x.sort_values(['pkey','ddate']); x['prev']=x.groupby('pkey').price_median.shift(); x['pd']=x.groupby('pkey').ddate.shift()
y=x[x.prev.notna()]
y['same']=(y.price_median-y.prev).abs()<1e-6
y['sw']=y.grp=='SOFT'
print(y.groupby(['ddate','sw']).same.agg(['mean','size']).round(3).to_string())
