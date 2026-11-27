# #172: export the HonestDiD (Rambachan-Roth 2023) input for the registered chain-linked indices (Appendix B):
# betahat = chain-linked index at every event quarter except the base (Sep 2025 = 0), sigma = covariance of the same
# vector over 300 block-bootstrap draws (seed 20261108, same blocks as the registered breakdown: borrower id for the
# Model index, loan for the Fund index). Outcomes: mark and Below99. R runs on a local PC (CRAN is blocked in the cloud).
from r7_common import *
from d2_rank import changes, c_t
H=f'{OUT}/honestdid'; os.makedirs(H,exist_ok=True)
L=Log(f'{H}/log_07_honestdid_input.txt'); p=load(); ch=changes(p)
def theta(x,y):
    q2=sorted(x.ddate.unique()); c=pd.Series({q:c_t(x[x.ddate==q],y,x[x.ddate==q].sw) for q in q2}); th={}
    for k in sorted(set(q2)|{q2[0]-pd.offsets.QuarterEnd()}):
        if k<BASE: th[k]=-c[(c.index>k)&(c.index<=BASE)].sum()
    th[BASE]=0.0
    for q in q2:
        if q>BASE: th[q]=c[(c.index>BASE)&(c.index<=q)].sum()
    return pd.Series(th).sort_index()
cm=ch[model_lab(ch)&~ch.grp.isin(['ITSV','DATA','HARD'])].assign(sw=lambda x:(x.grp=='SOFT').astype(float)).dropna(subset=['bid'])
cf=ch[FUNDLAB(ch)].assign(sw=lambda x:(x.fsec=='software').astype(float)).dropna(subset=['loan'])
for nm,x,blk in [('Model',cm,'bid'),('Fund',cf,'loan')]:
    rng=np.random.default_rng(20261108); idx=x.groupby(blk).indices; blocks=np.array(list(idx.keys()))
    for y,Y in [('dmark','mark'),('db99','below99')]:
        th=theta(x,y); B=[]
        for _ in range(300):
            dr=rng.choice(len(blocks),len(blocks),replace=True); B.append(theta(x.iloc[np.concatenate([idx[blocks[j]] for j in dr])],y))
        B=pd.DataFrame(B)[th.index]; keep=[q for q in th.index if q!=BASE]
        S=B[keep].cov()
        pre=[q for q in keep if q<BASE]; post=[q for q in keep if q>BASE]
        pd.DataFrame({'quarter':[f'{q:%Y-%m}' for q in keep],'event_time':[i-len(pre) if i<len(pre) else i-len(pre)+1 for i in range(len(keep))],
                      'betahat':th[keep].values,'se':np.sqrt(np.diag(S))}).to_csv(f'{H}/honestdid_{nm}_{Y}_betahat.csv',index=False)
        S.to_csv(f'{H}/honestdid_{nm}_{Y}_sigma.csv',index=False,header=False)
        L(f'{nm} {y}: {len(x)} changes, {len(blocks)} {blk} blocks | numPre {len(pre)} numPost {len(post)} | June 2026 {th[JUN]:+.2f} (bootstrap SE {np.sqrt(S.loc[JUN,JUN]):.2f})')
L.save()
