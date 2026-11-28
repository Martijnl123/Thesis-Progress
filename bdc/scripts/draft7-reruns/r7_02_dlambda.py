# Item 2 (draft 7): the Jun-Sep 2025 Delta-lambda (draft 6: "driven by a few large gaps").
# Registered estimator (d2_step2.pa): dm = m_t - m_{t-1} on G = p_t - m_{t-1}, G x Soft; |g0| = |p_{t-1} - m_{t-1}| <= 10; borrower clusters.
# Because m_{t-1} sits in both dm and G, any transitory error in the June 2025 mark biases lambda towards one and can create a
# software difference. (a) 2SLS: instrument G with Z = p_t - m_{t-2} (the March 2025 mark replaces the June 2025 mark), and
# G x Soft with Z x Soft. (b) quoted-both-quarters subsample: the registered sample already needs a price at t-1 and t (g0 and G);
# here also a price at t-2 (March 2025), on the same pairs as the 2SLS, so OLS and 2SLS compare like with like.
# (c) the same for the three later quarters, so the Jun-Sep 2025 change can be compared with the quarters of the text.
from r7_common import *
import d2_step2 as S2
L=Log(f'{OUT}/log_02_dlambda.txt'); p=load()
def build(p,matchfile='nport_matches_coupon.parquet'):
    M=pd.read_parquet(f'{CM.V}/{matchfile}'); assert M.ddate.max()<=CM.CAP; M=M[M.verified&(M.lvl=='2')]
    px=M.groupby('pos_id').price_median.median().rename('px')
    lab=p.grp.notna()&~p.grp.isin(['UNKN','FUND','ITSV','DATA','HARD']); keys=set(p.loc[(p.ddate==BASE)&p.ok&lab,'pkey'])
    q=p[p.pkey.isin(keys)].join(px,on='pos_id'); q=q[q.mark.between(0,110)]; q['sw']=(q.grp=='SOFT').astype(float)
    bs=set(q.loc[(q.ddate==BASE)&q.px.notna(),'pkey']); q=q[q.pkey.isin(bs)&(q.ddate>='2025-03-31')].copy()
    return q
q=build(p); w=S2.wide(q)
assert set(S2.wide(q[q.ddate>='2025-06-30']).index)==set(w.index)
def tsls(x,clu='clu'):
    """2SLS of dm on [1, soft, G, G*soft] with instruments [1, soft, Z, Z*soft]; cluster-robust sandwich, small-sample factor G/(G-1)."""
    X=np.column_stack([np.ones(len(x)),x.soft,x.gap,x.gap*x.soft]); Zm=np.column_stack([np.ones(len(x)),x.soft,x.z,x.z*x.soft]); y=x.dm.to_numpy()
    Pz=Zm@np.linalg.pinv(Zm.T@Zm)@Zm.T; Xh=Pz@X; b=np.linalg.solve(Xh.T@X,Xh.T@y); u=y-X@b
    A=np.linalg.inv(Xh.T@Xh); codes=pd.factorize(x[clu])[0]; G=codes.max()+1
    S=np.column_stack([np.bincount(codes,weights=Xh[:,j]*u,minlength=G) for j in range(4)]); V=A@(S.T@S)@A*G/(G-1)
    # first stage F for G (Kleibergen-Paap not needed for a check; report the cluster-robust Wald F of Z and Z*soft on G)
    import statsmodels.api as sm
    fs=sm.OLS(x.gap.to_numpy(),Zm).fit(cov_type='cluster',cov_kwds={'groups':codes}); F=float(fs.f_test(np.eye(4)[2:]).fvalue)
    return b[2],b[3],float(np.sqrt(V[3,3])),F
rows=[]
QQ3=[('202503','202506','202509'),('202506','202509','202512'),('202509','202512','202603'),('202512','202603','202606')]
for t2,t0,t1 in QQ3:
    x=S2.pa_frame(w,t0,t1); reg=x[x.g0.abs()<=10]
    lam,dl,dls,ns,n=S2.pa(reg); rows.append((t1,'registered (price at t-1 and t, |g0|<=10)',lam,dl,dls,ns,n,np.nan))
    x['z']=w[f'px_{t1}'].reindex(x.index)-w[f'mark_{t2}'].reindex(x.index)
    hasq=w[f'px_{t2}'].reindex(x.index).notna() if f'px_{t2}' in w else pd.Series(False,index=x.index)   # no N-PORT price file before June 2025
    b3=x[(x.g0.abs()<=10)&x.z.notna()]; lam,dl,dls,ns,n=S2.pa(b3); rows.append((t1,'OLS, also priced and held at t-2',lam,dl,dls,ns,n,np.nan))
    b4=x[(x.g0.abs()<=10)&x.z.notna()&hasq]
    if len(b4)>50: lam,dl,dls,ns,n=S2.pa(b4); rows.append((t1,'OLS, quoted at t-2, t-1 and t',lam,dl,dls,ns,n,np.nan))
    lam,dl,dls,F=tsls(b3); rows.append((t1,'2SLS, Z = p_t - m_{t-2}',lam,dl,dls,int(b3.soft.sum()),len(b3),F))
    if len(b4)>50: lam,dl,dls,F=tsls(b4); rows.append((t1,'2SLS, quoted at t-2, t-1 and t',lam,dl,dls,int(b4.soft.sum()),len(b4),F))
    # no identical pairs at t-1 (mark copied from the quote): if the June mark equals the June price, the mark has no own error
    b6=reg[reg.gap.abs()<=20]; lam,dl,dls,ns,n=S2.pa(b6); rows.append((t1,'OLS, registered sample with |G| <= 20',lam,dl,dls,ns,n,np.nan))
    b5=b3[(b3.g0.abs()>=0.01)]; lam,dl,dls,ns,n=S2.pa(b5); rows.append((t1,'OLS, mark differs from price at t-1',lam,dl,dls,ns,n,np.nan))
R=pd.DataFrame(rows,columns=['t','version','lam_other','dlambda','se','n_sw','n','firstF'])
L('[dlambda] by quarter and version (t = end quarter):\n'+R.round(3).to_string())
tag={'registered (price at t-1 and t, |g0|<=10)':'Reg','OLS, also priced and held at t-2':'Ols','OLS, quoted at t-2, t-1 and t':'OlsQ','2SLS, Z = p_t - m_{t-2}':'Iv','2SLS, quoted at t-2, t-1 and t':'IvQ','OLS, mark differs from price at t-1':'NoSame','OLS, registered sample with |G| <= 20':'Gtw'}
QN={'202509':'Sep','202512':'Dec','202603':'Mar','202606':'Jun'}
for _,r in R.iterrows():
    nm=f'Dlr{tag[r.version]}{QN[r.t]}'; putse(nm,r.dlambda,r.se,f'Delta-lambda ending {r.t}, {r.version}'); put(f'N{nm}',r.n_sw,'int','software pairs')
    put(f'{nm}Lam',r.lam_other,'f2','lambda for other loans')
    if not np.isnan(r.firstF): put(f'{nm}F',r.firstF,'f0','first-stage cluster-robust F on the two excluded instruments')
# Jun-Sep 2025: who drives it (names behind the large gaps, to-do #105)
x=S2.pa_frame(w,'202506','202509'); x=x[x.g0.abs()<=10]
x['name']=p.drop_duplicates('pkey').set_index('pkey').ident.reindex(x.index).str.slice(0,70)
sw=x[x.soft==1].assign(a=lambda z:z.gap.abs()).sort_values('a',ascending=False).head(15)
L('[dlambda] Jun-Sep 2025: 15 software pairs with the largest |G| (manager, G, dm, g0, description):\n'+sw[['family','gap','dm','g0','name']].round(2).to_string())
write_macros(f'{OUT}/numbers-r7-02-dlambda.tex',list(MAC),'Draft 7 rerun item 2: Jun-Sep 2025 Delta-lambda with the lagged-mark instrument (r7_02_dlambda.py, 26 Nov 2026)')
L.save()
