# Minimal OLS with several absorbed fixed effects (alternating projections) and cluster-robust SEs.
import numpy as np, pandas as pd
def _codes(s): return pd.factorize(s)[0]
def demean(A, groups, tol=1e-9, maxit=2000):
    A=A.astype(float).copy()
    cnts=[np.bincount(g) for g in groups]
    for it in range(maxit):
        old=A.copy()
        for g,c in zip(groups,cnts):
            for j in range(A.shape[1]):
                A[:,j]-=(np.bincount(g,weights=A[:,j],minlength=len(c))/c)[g]
        if np.abs(A-old).max()<tol: break
    return A
def drop_singletons(d, fe):
    while True:
        m=np.ones(len(d),bool)
        for f in fe: m&=d.groupby(f)[f].transform('size').values>1
        if m.all(): return d
        d=d[m]
def feols(d, y, xcols, fe, cluster):
    d=drop_singletons(d.dropna(subset=[y]+xcols), fe)
    groups=[_codes(d[f]) for f in fe]
    A=demean(d[[y]+xcols].to_numpy(), groups)
    Y=A[:,0]; X=A[:,1:]
    XtX_inv=np.linalg.pinv(X.T@X); b=XtX_inv@X.T@Y; u=Y-X@b
    cl=_codes(d[cluster]); G=cl.max()+1
    S=np.zeros((G,X.shape[1]))
    for j in range(X.shape[1]): S[:,j]=np.bincount(cl,weights=X[:,j]*u,minlength=G)
    V=XtX_inv@(S.T@S)@XtX_inv*G/(G-1)
    return pd.DataFrame({'coef':b,'se':np.sqrt(np.diag(V))},index=xcols), len(d), G
