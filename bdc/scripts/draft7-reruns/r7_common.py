# Shared helpers for the draft 7 reruns (26 Nov 2026). Reuses the draft 2-6 loaders and estimators unchanged.
# Data through 30 June 2026 only (the loader asserts it). The September 2026 builds are read, never rebuilt here.
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..','..')))  # repo root
import sys, os, numpy as np, pandas as pd
sys.path.insert(0,(_ROOT+'/bdc/scripts/draft2')); sys.path.insert(0,(_ROOT+'/bdc/scripts'))
import d2_common as CM
from d2_common import load, cohort_keys, frozen, model_lab, BASE, JUN, MAR, DEC, es, get, MAC, put, putse
from ai_rev2_lib import fit, se, vcov
from scipy.stats import chi2
CM.PCT_SIGN=True
OUT=(_ROOT+'/bdc/v2/draft7-reruns')
os.makedirs(OUT,exist_ok=True)
FUNDLAB=lambda d: d.fsec.notna()&~d.fsec.isin(['it_services','tech_other'])

def es_v(d,fe=('pkey','fq','vq'),clu='clu',ys=('mark','b99'),sw='sw'):
    """Same regressors as d2_common.es (Eq. 1), but also returns the clustered vcov of the software-by-quarter terms
    so the pre-period betas can be tested jointly."""
    d=d.copy(); qs=sorted(d.ddate.unique()); ev=[q for q in qs if q!=BASE]; S=[]; X=[]
    for q in ev:
        k=q.strftime('%Y%m'); dq=(d.ddate==q).astype(float)
        d[f'sw_{k}']=d[sw]*dq; S.append(f'sw_{k}')
        d[f'sp{k}']=d.spread_pre.fillna(0)*100*dq; d[f'spm{k}']=d.spread_pre.isna()*dq; d[f'pk{k}']=d.pik_pre.fillna(0)*dq
        X+=[f'sp{k}',f'spm{k}',f'pk{k}']
    r={}
    for y in ys:
        dd,b,sc,Ainv=fit(d,y,S+X,list(fe)); V=vcov(dd,sc,Ainv,clu)[:len(S),:len(S)]
        bb=b[S].values; idx=[q.strftime('%Y-%m') for q in ev]
        pre=[i for i,q in enumerate(ev) if q<BASE]
        W=float(bb[pre]@np.linalg.pinv(V[np.ix_(pre,pre)])@bb[pre]); 
        u=dd.drop_duplicates('pkey')
        r[y]=dict(path=pd.DataFrame({'b':bb,'se':np.sqrt(np.diag(V))},index=idx),wald=W,df=len(pre),p=float(chi2.sf(W,len(pre))),
                  n=dict(pos=dd.pkey.nunique(),sw=int(u[sw].sum()),funds=dd.cik.nunique(),obs=len(dd),clusters=dd[clu].nunique()))
    return r

class Log:
    def __init__(s,path): s.path=path; s.L=[]
    def __call__(s,x): s.L.append(str(x)); print(x,flush=True)
    def save(s): open(s.path,'w').write('\n'.join(s.L)+'\n')

def write_macros(path,names,header):
    with open(path,'w') as f:
        f.write(f'% {header}\n')
        for n in names:
            tex,raw,note=MAC[n]; f.write(f'\\newcommand{{\\{n}}}{{{tex}}} % {note}\n')
