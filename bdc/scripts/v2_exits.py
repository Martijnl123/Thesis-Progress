# Exit table (v2): what happens to a position that has no successor at the fund's next quarter-end.
#   continues            linked by the stable key
#   similar remains      the fund still holds a position with a similar description (refinanced, restructured or split)
#   gone, last mark >=97 / 80-97 / <80   the borrower left the fund's schedule; last mark as a guide to repaid vs distressed
#   fund stopped         the fund has no schedule at the next quarter-end
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..')))  # repo root
import re, sys, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.insert(0,(_ROOT+'/bdc/scripts'))
from matchlib import norm_desc as norm
V=(_ROOT+'/bdc/v2')
p=pd.read_parquet(f'{V}/panel.parquet',columns=['pos_id','cik','ddate','mark','ok','sector'])
p=p.merge(pd.read_parquet(f'{V}/positions.parquet',columns=['pos_id','ident']),on='pos_id')
L=pd.read_parquet(f'{V}/links.parquet')
p['nk']=p.ident.map(norm)
dates=sorted(p.ddate.unique()); nxt=dict(zip(dates[:-1],dates[1:]))
p['nx']=p.ddate.map(nxt)
rep=set(zip(p.cik,p.ddate))
p['status']=np.where(p.pos_id.isin(L.pos_a),'continues',np.where([(c,n) in rep for c,n in zip(p.cik,p.nx)],'?','fund stopped'))
sim=pd.Series(0.0,index=p.index)
for cik,g in p.groupby('cik'):
    vec=TfidfVectorizer(token_pattern=r'(?u)\b[a-z0-9&]+\b',sublinear_tf=True).fit(g.nk)
    for d0,a in g[g.status=='?'].groupby('ddate'):
        b=g[g.ddate==nxt[d0]]
        sim[a.index]=(vec.transform(a.nk)@vec.transform(b.nk).T).max(axis=1).toarray().ravel()
q=p.status=='?'
p.loc[q,'status']=np.select([sim[q]>=0.5,p.mark[q]>=97,p.mark[q]>=80],['similar remains','gone, last mark 97+','gone, last mark 80-97'],'gone, last mark <80')
p.drop(columns=['ident','nk']).to_parquet(f'{V}/exits.parquet',index=False)
u=p[p.ok&p.sector.notna()&~p.sector.isin(['it_services','tech_other'])&p.nx.notna()].copy()
u['grp']=np.where(u.sector=='software','software','other')
u['per']=np.where(u.ddate<'2025-09-30','pre (Dec 2022-Jun 2025, avg per quarter)',u.ddate.dt.strftime('%b %Y')+' to next')
order=['continues','similar remains','gone, last mark 97+','gone, last mark 80-97','gone, last mark <80','fund stopped']
t=(u.groupby(['per','grp']).status.value_counts(normalize=True).unstack().reindex(columns=order).fillna(0)*100).round(1)
t['positions']=u.groupby(['per','grp']).size()
pd.set_option('display.width',250)
print('where clean, labelled positions are one quarter later (% of positions)'); print(t.to_string())
