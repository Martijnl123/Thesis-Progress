# Stable within-fund position key (v2), DETERMINISTIC TIE-BREAK version (draft 6, 26 Nov 2026; otherwise identical to sep2026/v2_stablekey.py).
# Links each position to the same fund's position at the fund's next reported quarter-end:
#  1. exact match on the normalised description (affiliation labels, punctuation, rates and dates removed; tranche numbers kept),
#     when that description is unique in the fund at both dates;
#  2. otherwise a one-to-one greedy match within the fund on description similarity, spread and principal.
# Chains of links give the stable id (sid). Output: v2/links.parquet and v2/sid.parquet.
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..','..')))  # repo root
import re, sys, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.insert(0,(_ROOT+'/bdc/scripts'))
from matchlib import norm_desc as norm
B=__import__('os').environ['OUTV']
p=pd.read_parquet(f'{B}/positions.parquet',columns=['pos_id','cik','ddate','ident','principal','spread','rate','maturity','mark'])
p['nk']=p.ident.map(norm)
dates=sorted(p.ddate.unique())
nxt={a:b for a,b in zip(dates[:-1],dates[1:])}
links=[]
for cik,g in p.groupby('cik'):
    ds=sorted(g.ddate.unique())
    if len(ds)<2: continue
    vec=TfidfVectorizer(token_pattern=r'(?u)\b[a-z0-9&]+\b',sublinear_tf=True).fit(g.nk)
    for d0,d1 in zip(ds[:-1],ds[1:]):
        if nxt.get(d0)!=d1: continue          # only consecutive quarter-ends; a skipped quarter breaks the chain
        a=g[g.ddate==d0]; b=g[g.ddate==d1]
        ua=a.nk.map(a.nk.value_counts())==1; ub=b.nk.map(b.nk.value_counts())==1
        m=a[ua].merge(b[ub],on='nk',suffixes=('_a','_b'))
        links+= [(x,y,'exact',1.0) for x,y in zip(m.pos_id_a,m.pos_id_b)]
        ra=a[~a.pos_id.isin(m.pos_id_a)]; rb=b[~b.pos_id.isin(m.pos_id_b)]
        if len(ra)==0 or len(rb)==0: continue
        S=(vec.transform(ra.nk)@vec.transform(rb.nk).T).toarray()
        i,j=np.nonzero(S>=0.5)
        if len(i)==0: continue
        sa,sb=ra.spread.values[i],rb.spread.values[j]
        rta,rtb=ra.rate.values[i],rb.rate.values[j]
        sp=np.where(~np.isnan(sa)&~np.isnan(sb),np.where(np.abs(sa-sb)<=1e-4,0.25,np.where(np.abs(sa-sb)>0.005,-0.25,0)),
           np.where(~np.isnan(rta)&~np.isnan(rtb)&(np.abs(rta-rtb)<=1e-4),0.15,0))
        lr=np.abs(np.log(rb.principal.values[j]/ra.principal.values[i]))
        pr=np.where(lr<=0.02,0.25,np.where(lr<=0.25,0.1,np.where(lr>np.log(3),-0.5,0)))
        sc=S[i,j]+sp+pr
        # 26 Nov 2026 (draft 6): deterministic tie-break. np.argsort(-sc) (quicksort) ordered equal scores differently across
        # machines/numpy builds (562 of 366,903 links on a local PC). Now: score rounded to 1e-9, highest first; ties by
        # the earlier position's pos_id, then the later position's pos_id. Only the order of exact ties changes.
        o=np.lexsort((rb.pos_id.values[j],ra.pos_id.values[i],-np.round(sc,9))); usedA=set(); usedB=set()
        for k in o:
            if sc[k]<0.75: break
            x,y=ra.pos_id.values[i[k]],rb.pos_id.values[j[k]]
            if x in usedA or y in usedB: continue
            usedA.add(x); usedB.add(y); links.append((x,y,'fuzzy',float(sc[k])))
L=pd.DataFrame(links,columns=['pos_a','pos_b','method','score'])
L.to_parquet(f'{B}/links.parquet',index=False)
# chains -> stable id (each position has at most one successor and one predecessor)
succ=dict(zip(L.pos_a,L.pos_b)); has_pred=set(L.pos_b)
sid={}
for s in p.pos_id[~p.pos_id.isin(has_pred)]:
    x=s
    while True:
        sid[x]=s
        if x not in succ: break
        x=succ[x]
out=pd.DataFrame({'pos_id':list(sid.keys()),'sid':list(sid.values())})
out.to_parquet(f'{B}/sid.parquet',index=False)
print('positions', len(p), '| with sid', len(out), '| links', len(L), L.method.value_counts().to_dict(), flush=True)
