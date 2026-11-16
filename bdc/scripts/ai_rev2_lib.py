# Helpers for the checks the second (deep-research) review asked for, 23 Oct 2026:
# weighted OLS with absorbed fixed effects, multiway clustering by inclusion-exclusion,
# and dyadic-robust variance for manager pairs (Aronow, Samii and Assenova 2015) combined with loan clustering.
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..')))  # repo root
import sys, itertools, numpy as np, pandas as pd
sys.path.insert(0,(_ROOT+'/bdc/scripts'))
from ai_prelim_fe import drop_singletons, _codes

def fit(d, y, xcols, fe, w=None):
    d=drop_singletons(d.dropna(subset=[y]+xcols), fe).copy()
    wt=np.ones(len(d)) if w is None else d[w].to_numpy(float)
    groups=[_codes(d[f]) for f in fe]; cw=[np.bincount(g,weights=wt) for g in groups]
    A=d[[y]+xcols].to_numpy(float).copy()
    for it in range(5000):
        old=A.copy()
        for g,c in zip(groups,cw):
            for j in range(A.shape[1]): A[:,j]-=(np.bincount(g,weights=wt*A[:,j],minlength=len(c))/c)[g]
        if np.abs(A-old).max()<1e-9: break
    Y=A[:,0]; X=A[:,1:]; XW=X*wt[:,None]
    Ainv=np.linalg.pinv(X.T@XW); b=Ainv@XW.T@Y; u=Y-X@b
    return d, pd.Series(b,index=xcols), XW*u[:,None], Ainv

def _ss(sc, codes, corr=True):
    G=codes.max()+1
    S=np.column_stack([np.bincount(codes,weights=sc[:,j],minlength=G) for j in range(sc.shape[1])])
    return S.T@S*(G/(G-1) if corr else 1.0)

def vcov(d, sc, Ainv, dims):
    """Cluster-robust V; dims is one column or several (multiway, inclusion-exclusion)."""
    dims=[dims] if isinstance(dims,str) else list(dims)
    M=np.zeros((sc.shape[1],)*2)
    for k in range(1,len(dims)+1):
        for sub in itertools.combinations(dims,k):
            key=d[sub[0]].astype(str)
            for s in sub[1:]: key=key+'|'+d[s].astype(str)
            M+=(-1)**(k+1)*_ss(sc,_codes(key))
    return Ainv@M@Ainv

def se(d, sc, Ainv, dims): return np.sqrt(np.clip(np.diag(vcov(d,sc,Ainv,dims)),0,None))

def se_dyad_loan(d, sc, Ainv, loan='loan_a', a='fa', b='fc'):
    """Observations are dependent if they share the loan or share a manager."""
    two=np.vstack([sc,sc]); L=d[loan].astype(str).values; A=d[a].values; Bm=d[b].values
    M_loan=_ss(sc,_codes(d[loan]),False)
    M_dyad=_ss(two,_codes(pd.Series(np.concatenate([A,Bm]))),False)-_ss(sc,_codes(d[a]+'|'+d[b]),False)
    M_int=_ss(two,_codes(pd.Series(np.concatenate([L+'|'+A,L+'|'+Bm]))),False)-_ss(sc,_codes(pd.Series(L+'|'+A+'|'+Bm)),False)
    V=Ainv@(M_loan+M_dyad-M_int)@Ainv
    return np.sqrt(np.clip(np.diag(V),0,None))
