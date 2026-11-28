# Item 1 (draft 7): is the Step 1 pre-trend (mark coefficient -1.3 in Dec 2022 rising to 0 in Sep 2025, Fig. 3b) survivorship?
# (a) balanced cohort: cohort positions (held, valid and model-labelled on 30 Sep 2025) that are also valid at EVERY quarter
#     from Dec 2022 to Sep 2025; and a looser version: valid on 31 Dec 2022.
# (b) rolling cohort: no survival to 30 Sep 2025 required. Model labels exist only for borrowers held on 30 Sep 2025, so
#     (b1) model labels: every position of a labelled borrower, valid at its own dates (loan survival not required,
#          borrower survival still is); (b2) funds' labels (pre-Sep-2025 vote): every labelled position, no survival at all.
#     Comparison for (b2): the same funds' labels on the Sep 2025 cohort (Table 4 column 3).
#     Same Eq. 1 (position + fund x quarter + vintage x quarter FE, spread/PIK x quarter controls, borrower clusters,
#     loan clusters where no borrower id). Post-period rows as in the registered frozen rule (mark 0-110, principal >= 250k).
# (c) chain-linked index (Appendix B) and breakdown Mbar on the same samples, same seed 20261108, 300 block draws.
# (d) joint Wald test that all 11 pre-period betas are zero (examiner #88), for every version.
from r7_common import *
# Usage: python3 r7_01_survivorship.py es            -> event studies + Wald tests (numbers-r7-01-survivorship.tex)
#        python3 r7_01_survivorship.py chain <Name>  -> chain-linked index + breakdown for one sample (numbers-r7-01-chain-<Name>.tex)
# Model labels exist only for positions held on 30 Sep 2025, so the 'Model' index equals the cohort-only model index; it is not rerun.
PART=sys.argv[1]; CHV=sys.argv[2] if len(sys.argv)>2 else None
from d2_rank import changes, c_t
L=Log(f'{OUT}/log_01_survivorship.txt' if PART=='es' else f'{OUT}/log_01_chain_{CHV}.txt'); p=load(); L(f'panel {len(p)} rows, last date {p.ddate.max().date()}')
qs=sorted(p.ddate.unique()); pre_q=[q for q in qs if q<=BASE]
keys=cohort_keys(p); d0=frozen(p,keys); d0['sw']=(d0.grp=='SOFT').astype(float)
okq=p[p.ok&(p.ddate<=BASE)].groupby('pkey').ddate.nunique()
bal=set(okq[okq==len(pre_q)].index)&keys
dec22=set(p.loc[p.ok&(p.ddate==qs[0]),'pkey'])&keys
L(f'cohort positions {len(keys)}; valid at all {len(pre_q)} pre quarters {len(bal)}; valid on 31 Dec 2022 {len(dec22)}')
ever_ok=set(p.loc[p.ok,'pkey'])
valid=np.where(p.ddate<=BASE,p.ok,p.mark.between(0,110)&(p.principal>=250e3)&p.pkey.isin(ever_ok))
roll=p[valid].copy()
r_m=roll[model_lab(roll)].copy(); r_m['sw']=(r_m.grp=='SOFT').astype(float)
r_f=roll[FUNDLAB(roll)].copy(); r_f['sw']=(r_f.fsec=='software').astype(float)
r_f['clu']=np.where(r_f.bid.notna(),'b'+r_f.bid.astype(str),'l'+r_f.loan.astype(str))
c_f=d0[FUNDLAB(d0)].copy(); c_f=c_f[c_f.pkey.isin(set(c_f.loc[c_f.ddate==BASE,'pkey']))]; c_f['sw']=(c_f.fsec=='software').astype(float)
V={'Head':('model labels, Sep 2025 cohort (headline, Table 4 col 1)',d0),
   'Bal':('model labels, cohort valid at every quarter Dec 2022-Sep 2025 (balanced)',d0[d0.pkey.isin(bal)]),
   'Dec':('model labels, cohort valid on 31 Dec 2022',d0[d0.pkey.isin(dec22)]),
   'RollM':('model labels, rolling: all positions of labelled borrowers, no loan survival',r_m),
   'CohF':("funds' labels, Sep 2025 cohort (Table 4 col 3)",c_f),
   'RollF':("funds' labels, rolling: all labelled positions, no survival",r_f)}
R={}
for nm,(lab,d) in (V.items() if PART=='es' else []):
    r=es_v(d); R[nm]=r
    L(f'\n=== {nm}: {lab}: {r["mark"]["n"]}')
    for y,Y,dg in [('mark','Mark',2),('b99','Below',1)]:
        P=r[y]['path']; L(f'  {y}: '+'; '.join(f'{q} {P.b[q]:+.{dg}f} ({P.se[q]:.{dg}f})' for q in P.index))
        L(f'  {y}: joint Wald all pre betas = 0: chi2({r[y]["df"]}) = {r[y]["wald"]:.1f}, p = {r[y]["p"]:.4f}')
        putse(f'Sv{nm}{Y}First',P.b['2022-12'],P.se['2022-12'],f'{lab}: {y} beta Dec 2022',d=dg)
        putse(f'Sv{nm}{Y}Jun',P.b['2026-06'],P.se['2026-06'],f'{lab}: {y} beta Jun 2026',d=dg)
        put(f'Sv{nm}{Y}Wald',r[y]['wald'],'f1',f'{lab}: joint Wald chi2({r[y]["df"]}) all pre-period betas zero, {y}')
        put(f'Sv{nm}{Y}WaldP',r[y]['p'],'f3',f'{lab}: p of joint pre-trend Wald, {y}')
    n=r['mark']['n']; put(f'NSv{nm}Pos',n['pos'],'int',f'{lab}: positions'); put(f'NSv{nm}Sw',n['sw'],'int',f'{lab}: software positions')

# ---------------- (c) chain-linked index and breakdown Mbar on the same samples
ch=changes(p) if PART=='chain' else None
def theta(x,y):
    q2=sorted(x.ddate.unique())
    c=pd.Series({q:c_t(x[x.ddate==q],y,x[x.ddate==q].sw) for q in q2}); th={}
    allq=sorted(set(q2)|{q2[0]-pd.offsets.QuarterEnd()})
    for k in allq:
        if k<BASE: th[k]=-c[(c.index>k)&(c.index<=BASE)].sum()
    th[BASE]=0.0
    for q in q2:
        if q>BASE: th[q]=c[(c.index>BASE)&(c.index<=q)].sum()
    return pd.Series(th).sort_index()
def bnds(th,M,q):
    pre=th[th.index<=BASE]; D=pre.diff().abs().max(); post=list(th.index[th.index>BASE]); k=post.index(q)+1
    return th[q]-k*M*D, th[q]+k*M*D
cm=None if ch is None else ch[model_lab(ch)&~ch.grp.isin(['ITSV','DATA','HARD'])].assign(sw=lambda x:(x.grp=='SOFT').astype(float))
cf=None if ch is None else ch[FUNDLAB(ch)].assign(sw=lambda x:(x.fsec=='software').astype(float))
CV={} if ch is None else {'Model':('model labels, all labelled-borrower positions (registered Model index)',cm,'bid'),
    'ModelCoh':('model labels, Sep 2025 cohort positions only',cm[cm.pkey.isin(keys)],'bid'),
    'ModelBal':('model labels, balanced cohort only',cm[cm.pkey.isin(bal)],'bid'),
    'Fund':("funds' labels, all labelled positions (registered Fund index, rolling)",cf,'loan'),
    'FundCoh':("funds' labels, Sep 2025 cohort positions only",cf[cf.pkey.isin(set(c_f.pkey))],'loan'),
    'FundBal':("funds' labels, balanced cohort positions only",cf[cf.pkey.isin(bal)],'loan')}
Mgrid=np.round(np.arange(0,3.01,0.05),2)
for nm,(lab,x,blk) in ([(CHV,CV[CHV])] if PART=='chain' else []):
    x=x.dropna(subset=[blk]); rng=np.random.default_rng(20261108)
    for y,Y in [('dmark','Mark'),('db99','Below')]:
        th=theta(x,y); idx=x.groupby(blk).indices; blocks=np.array(list(idx.keys())); B=[]
        for _ in range(300):
            dr=rng.choice(len(blocks),len(blocks),replace=True); B.append(theta(x.iloc[np.concatenate([idx[blocks[j]] for j in dr])],y))
        B=pd.DataFrame(B); D=th[th.index<=BASE].diff().abs().max()
        out={}
        for q,Q in [(MAR,'Mar'),(JUN,'Jun')]:
            bd=None
            for M in Mgrid:
                lo=np.percentile([bnds(B.iloc[i],M,q)[0] for i in range(len(B))],5); hi=np.percentile([bnds(B.iloc[i],M,q)[1] for i in range(len(B))],95)
                if lo<=0<=hi: break
                bd=M
            out[Q]=0.0 if bd is None else bd
            put(f'Bd{nm}{Y}{Q}Sv',out[Q],'f2','breakdown Mbar (largest, step 0.05, robust 90% interval excludes zero), '+lab)
        dg=2 if Y=='Mark' else 1
        put(f'Th{nm}{Y}FirstSv',th.iloc[0],f'f{dg}','chain-linked index at Dec 2022 (cumulated pre-period drift to Sep 2025, sign flipped), '+lab,sign=True)
        put(f'Th{nm}{Y}DSv',D,f'f{dg}','largest pre-period quarterly change D, '+lab)
        put(f'Th{nm}{Y}JunSv',th[JUN],f'f{dg}','chain-linked index June 2026, '+lab,sign=True)
        L(f'[chain] {nm} ({lab}) {y}: {len(x)} changes | path '+'; '.join(f'{q:%Y-%m} {v:+.2f}' for q,v in th.items())+f' | D {D:.2f} | breakdown Mar {out["Mar"]} Jun {out["Jun"]}')
names=[n for n in MAC]
write_macros(f'{OUT}/numbers-r7-01-survivorship.tex' if PART=='es' else f'{OUT}/numbers-r7-01-chain-{CHV}.tex',names,'Draft 7 rerun item 1: survivorship of the Step 1 pre-trend, joint pre-trend Wald tests (r7_01_survivorship.py, 26 Nov 2026)')
L.save()
