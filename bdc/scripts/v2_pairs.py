# Direct pairwise matches between every position and positions of OTHER families (v2: as pairs.py, with paths as arguments).
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..')))  # repo root
import sys, pandas as pd, numpy as np, time
sys.path.insert(0,(_ROOT+'/bdc/scripts'))
from matchlib import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
debt,outdir=sys.argv[1],sys.argv[2]
THR=float(sys.argv[3]) if len(sys.argv)>3 else 0.5
d=pd.read_parquet(debt)
d=d[d.ddate.dt.month.isin([3,6,9,12]) & (d.ddate>='2022-12-31')].reset_index(drop=True)
d['pos_id']=np.arange(len(d))
rows=[]
for dt,g in d.groupby('ddate'):
    t=time.time(); g=g.reset_index(drop=True)
    docs=g.ident.map(clean_text)
    vec=TfidfVectorizer(token_pattern=r'(?u)\b[a-z&]{2,}\b',sublinear_tf=True); X=vec.fit_transform(docs)
    key=np.where(g.spread.notna(),'s'+(g.spread*10000).round().astype('Int64').astype(str),'r'+(g.rate*10000).round().astype('Int64').astype(str))
    fam=g.family.values; mat=g.maturity.values; pid=g.pos_id.values
    for b,idx in pd.Series(range(len(g))).groupby(key).indices.items():
        if len(idx)<2 or len(set(fam[idx]))<2: continue
        S=(X[idx]@X[idx].T).tocoo()
        m=(S.row<S.col)&(S.data>=THR)
        for i,j,v in zip(S.row[m],S.col[m],S.data[m]):
            a,c=idx[i],idx[j]
            if fam[a]==fam[c]: continue
            ma,mc=mat[a],mat[c]
            if not(pd.isna(ma) or pd.isna(mc)) and abs((ma-mc)/np.timedelta64(1,'D'))>62: continue
            rows.append((pid[a],pid[c],v))
    print(dt.date(), len(g), f'{time.time()-t:.0f}s', flush=True)
p=pd.DataFrame(rows,columns=['pa','pc','cos'])
p=pd.concat([p,p.rename(columns={'pa':'pc','pc':'pa'})],ignore_index=True)
p.to_parquet(f'{outdir}/pairs_{THR}.parquet',index=False); d.to_parquet(f'{outdir}/positions.parquet',index=False)
print('pairs (directed)', len(p))
