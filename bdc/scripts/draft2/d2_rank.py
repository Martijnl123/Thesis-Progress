# Sector rank (scaled, divided by sqrt(3), #47), fake cohorts (#50), smoothness bounds (#51), exits (#52),
# outside-cohort markdown by size (#31) and the late-December-audit check (#56). Data to 30 June 2026.
from d2_common import *
NAMES={'SOFT':'Software','TELE':'Telecom','MEDI':'Media','MKTG':'Marketing','HCPS':'Health care services','HCEQ':'Health care equipment','PHAR':'Pharma and biotech','INSU':'Insurance','FINS':'Financial services','PROF':'Professional services','COMM':'Commercial services','EDUC':'Education','CONS':'Consumer services','CPRD':'Consumer products','FOOD':'Food and beverage','CAPG':'Capital goods','CONB':'Construction services','TRAN':'Transport','AUTO':'Automotive','CHEM':'Chemicals and materials','ENER':'Energy','DIST':'Distribution','REAL':'Real estate'}

def changes(p):
    ever_ok=set(p.loc[p.ok,'pkey'])
    valid=np.where(p.ddate<=BASE,p.ok,p.mark.between(0,110)&(p.principal>=250e3)&p.pkey.isin(ever_ok))
    v=p[valid].sort_values(['pkey','ddate']).copy(); g=v.groupby('pkey'); v['dmark']=g.mark.diff(); v['db99']=g.b99.diff(); v['gapd']=g.ddate.diff().dt.days
    return v[v.gapd.between(80,100)].copy()
def c_t(x,y,ind):
    s=ind-ind.groupby(x.cik).transform('mean'); e=x[y]-x.groupby('cik')[y].transform('mean'); den=(s*s).sum(); return (s*e).sum()/den if den>0 else np.nan

def rank(p,T,log):
    ch=changes(p); ch=ch[ch.grp.notna()&~ch.grp.isin(['UNKN','FUND','ITSV','DATA','HARD'])]
    qs=sorted(ch.ddate.unique()); pre=[q for q in qs if q<=BASE]; post=[q for q in qs if q>BASE]; assert len(pre)==11 and len(post)==3
    rows=[]
    for gname in sorted(ch.grp.unique()):
        x=ch if gname=='SOFT' else ch[ch.grp!='SOFT']
        for y in ['dmark','db99']:
            c=pd.Series({q:c_t(x[x.ddate==q],y,(x[x.ddate==q].grp==gname).astype(float)) for q in qs}); sd=c[pre].std()
            rows.append(dict(grp=gname,y=y,sd=sd,post=c[post].sum(),T=c[post].sum()/sd/np.sqrt(3),T8=c[post].sum()/c[pre[-8:]].std()/np.sqrt(3),Tmar=c[post[1:]].sum()/sd/np.sqrt(2)))
    S=pd.DataFrame(rows); T['rank']=S; G=S.grp.nunique()
    put('NGroups',G,'int','groups in the sector rank'); put('RankP',1/G,'f3','p-value of rank 1 among the groups')
    for y,asc,Y in [('db99',False,'Below'),('dmark',True,'Mark')]:
        t=S[S.y==y].sort_values('T',ascending=asc).reset_index(drop=True); rk=int(t.index[t.grp=='SOFT'][0])+1
        put(f'Rank{Y}',rk,'int','software rank on the scaled statistic'); put(f'Scaled{Y}',float(t.loc[t.grp=='SOFT','T'].iloc[0]),'f1','software scaled statistic / sqrt(3), Dec25-Jun26 sum',sign=True)
        nx=t[t.grp!='SOFT'].iloc[0]; put(f'Next{Y}Name',NAMES[nx.grp].lower(),'str','next group'); put(f'Next{Y}',float(nx['T']),'f1','next group statistic',sign=True)
        t8=S[S.y==y].sort_values('T8',ascending=asc).reset_index(drop=True); put(f'Rank{Y}Eight',int(t8.index[t8.grp=='SOFT'][0])+1,'int','rank with last 8 pre changes')
        a=S[S.y==y].assign(a=lambda z:z['T'].abs()).sort_values('a',ascending=False).reset_index(drop=True); put(f'Rank{Y}Abs',int(a.index[a.grp=='SOFT'][0])+1,'int','rank on |T|')
        log(f'[rank] {y}: software rank {rk} of {G}; T {float(t.loc[t.grp=="SOFT","T"].iloc[0]):+.1f}; next {nx.grp} {nx["T"]:+.1f}; top5 '+'; '.join(f'{r.grp} {r["T"]:+.1f}' for _,r in t.head(5).iterrows()))
    return S

def fake_cohorts(p,T,log):
    """Placebo cohorts fixed at Sep 2023 and Sep 2024 (funds' labels voted on filings to Sep 2025), fake shock two quarters later."""
    lab=p.fsec.notna()&~p.fsec.isin(['it_services','tech_other']); qs=sorted(p.ddate.unique()); out={}
    for bq,nm in [('2023-09-30','Three'),('2024-09-30','Four'),('2025-09-30','Real')]:
        bq=pd.Timestamp(bq); i=qs.index(bq); after=qs[i+1:i+4]; shock=after[1]
        base=p[(p.ddate==bq)&p.ok&lab]; keys=set(base.pkey); sp=base.set_index('pkey').spread; pk=base.set_index('pkey').pik
        fz=p[p.pkey.isin(keys)&(p.ddate<=after[-1])]; post=fz.ddate>bq
        d=fz[~post&fz.ok | post&fz.mark.between(0,110)&(fz.principal>=250e3)].copy(); d['sw']=(d.fsec=='software').astype(float)
        d['post']=(d.ddate>=shock).astype(float); d['sp_b']=d.pkey.map(sp); d['pk_b']=d.pkey.map(pk).fillna(0)
        d['sp_post']=d.sp_b.fillna(0)*100*d.post; d['spm_post']=d.sp_b.isna()*d.post; d['pik_post']=d.pk_b*d.post; d['sw_post']=d.sw*d.post
        for y,Y,dd_ in [('mark','Mark',2),('b99','Below',1)]:
            dd,b,sc,Ainv=fit(d,y,['sw_post','sp_post','spm_post','pik_post'],['pkey','fq']); s=se(dd,sc,Ainv,'clu')[0]
            putse(f'Fake{nm}{Y}',float(b['sw_post']),float(s),f'cohort fixed {bq:%b %Y}, funds labels (pre-Sep-2025 vote), fake shock {shock:%b %Y}, post = 2 quarters, position + fund x quarter FE',d=dd_)
            out[(nm,y)]=(float(b['sw_post']),float(s))
        put(f'NFake{nm}',len(keys),'int',f'positions in the {bq:%b %Y} cohort')
        log(f'[fake] {nm}: '+'; '.join(f'{y} {out[(nm,y)][0]:+.2f} ({out[(nm,y)][1]:.2f})' for y in ['mark','b99']))
    T['fake']=out

def bounds(p,T,log,draws=300):
    """Chain-linked index and relative-magnitude bounds in the spirit of Rambachan and Roth (2023), percentile bootstrap."""
    ch=changes(p); base=BASE; qs=sorted(ch.ddate.unique())
    vers={'Model':(ch[ch.grp.notna()&~ch.grp.isin(['ITSV','DATA','HARD','UNKN','FUND'])].assign(sw=lambda x:(x.grp=='SOFT').astype(float)),'bid'),
          'Fund':(ch[ch.fsec.notna()&~ch.fsec.isin(['it_services','tech_other'])].assign(sw=lambda x:(x.fsec=='software').astype(float)),'loan')}
    rng=np.random.default_rng(20261108); Mgrid=np.round(np.arange(0,3.01,0.05),2); res={}
    def theta(x,y):
        c=pd.Series({q:c_t(x[x.ddate==q],y,x[x.ddate==q].sw) for q in qs}); th={}
        allq=sorted(set(qs)|{qs[0]-pd.offsets.QuarterEnd()})
        for k in allq:
            if k<base: th[k]=-c[(c.index>k)&(c.index<=base)].sum()
        th[base]=0.0
        for q in qs:
            if q>base: th[q]=c[(c.index>base)&(c.index<=q)].sum()
        return pd.Series(th).sort_index()
    def bnds(th,M):
        pre=th[th.index<=base]; D=pre.diff().abs().max(); post=th[th.index>base]
        return {q:(t-k*M*D,t+k*M*D) for k,(q,t) in enumerate(post.items(),start=1)}
    for nm,(x,blk) in vers.items():
        x=x.dropna(subset=[blk]) if blk=='bid' else x
        for y,Y in [('dmark','Mark'),('db99','Below')]:
            th=theta(x,y); idx=x.groupby(blk).indices; blocks=np.array(list(idx.keys())); B=[]
            for _ in range(draws):
                dr=rng.choice(len(blocks),len(blocks),replace=True); ii=np.concatenate([idx[blocks[j]] for j in dr]); B.append(theta(x.iloc[ii],y))
            B=pd.DataFrame(B)
            for q,Q in [(MAR,'Mar'),(JUN,'Jun')]:
                bd=None
                for M in Mgrid:
                    lo=np.percentile([bnds(B.iloc[i],M)[q][0] for i in range(len(B))],5); hi=np.percentile([bnds(B.iloc[i],M)[q][1] for i in range(len(B))],95)
                    if (lo<=0<=hi): break
                    bd=M
                bd=0.0 if bd is None else bd
                put(f'Bd{nm}{Y}{Q}',bd,'f2' if int(round(bd*20))%2 else 'f1',f'largest Mbar (step 0.05) at which the robust 90% interval excludes zero, {nm} labels, chain-linked index, {draws} block draws')
                res[(nm,Y,Q)]=bd
            log(f'[bounds] {nm} {y}: theta Mar {th[MAR]:+.2f} Jun {th[JUN]:+.2f}; breakdown Mar {res[(nm,Y,"Mar")]} Jun {res[(nm,Y,"Jun")]}; first pre theta {th.iloc[0]:+.2f}')
    T['bounds']=res

def outside(p,T,log):
    """#31: software markdown outside the cohort (positions with a fund label but no model label), by borrower-size tercile, June."""
    import statsmodels.formula.api as smf
    s9=p[(p.ddate==BASE)&p.ok].copy(); inc=s9.pkey.isin(cohort_keys(p))
    s9=s9[s9.fsec.notna()&~s9.fsec.isin(['it_services','tech_other'])].copy(); s9['sw']=(s9.fsec=='software').astype(int)
    lsz=p[(p.ddate==BASE)&p.ok].groupby('loan').principal.sum(); s9['lsize']=s9.loan.map(lsz)
    mj=p[(p.ddate==JUN)&p.mark.between(0,110)&(p.principal>=250e3)].set_index('pkey').mark
    s9['dy']=s9.pkey.map(mj)-s9.mark; s9['sp']=s9.spread_pre.fillna(0)*100; s9['spm']=s9.spread_pre.isna().astype(int); s9['pk']=s9.pik_pre.fillna(0)
    s9['in']=inc.reindex(s9.index).values
    cuts=s9.lsize.quantile([1/3,2/3]).values; s9['terc']=np.digitize(s9.lsize,cuts)
    rows=[]
    for grp,gn in [(False,'Out'),(True,'In')]:
        for t_,tn in [(None,'All'),(0,'Small'),(1,'Mid'),(2,'Large')]:
            x=s9[(s9['in']==grp)&(True if t_ is None else s9.terc==t_)].dropna(subset=['dy'])
            x=x.assign(cikS=x.cik.astype(str)); m=smf.ols('dy~sw+sp+spm+pk+C(cikS)',x).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(x.loan)[0]})
            rows.append((gn,tn,m.params.sw,m.bse.sw,len(x),int(x.sw.sum())))
            putse(f'Side{gn}{tn}',m.params.sw,m.bse.sw,f'cross-section, June 2026 change, funds labels, {"inside" if grp else "outside"} cohort, loan-size tercile {tn}, fund FE, loan-clustered')
            if tn=='All': put(f'NSide{gn}',len(x),'int','positions'); put(f'NSide{gn}Sw',int(x.sw.sum()),'int','software positions')
    T['outside']=pd.DataFrame(rows,columns=['grp','terc','b','se','n','nsw']); log('[outside] '+T['outside'].round(2).to_string())

def late_audit(d0,T,log):
    """#56: December 2025 software coefficient for funds whose 10-K for Dec 2025 was filed after 3 Feb 2026 vs before."""
    dec=d0[(d0.ddate==DEC)&d0.form.astype(str).str.startswith('10-K')]
    f=dec.groupby('cik').filed.min(); f=pd.to_datetime(f,errors='coerce')
    f=f.dropna(); md=f.median().normalize()   # every Dec 2025 10-K came after 3 Feb 2026, so split at the median filing date
    late=set(f[f>md].index); early=set(f[f<=md].index); put('AuditMedFiled',md.strftime('%-d %B'),'str','median Dec 2025 10-K filing date among cohort funds')
    put('AuditAfterShock',(f>pd.Timestamp('2026-02-03')).mean(),'pct0','share of cohort funds whose Dec 2025 10-K was filed after 3 Feb 2026')
    for nm,S in [('Late',late),('Early',early)]:
        r=es(d0[d0.cik.isin(S)],ys=('mark',)); b,s=get(r,'mark','2025-12'); putse(f'Audit{nm}Dec',b,s,f'H1 Dec 2025 mark coefficient, funds with Dec 2025 10-K filed {"after" if nm=="Late" else "on or before"} the median filing date')
        put(f'NAudit{nm}',len(S),'int','funds'); log(f'[audit] {nm} filers: {len(S)} funds, Dec mark {b:+.2f} ({s:.2f})')
    put('AuditFirstFiled',f.min().strftime('%-d %B'),'str','first Dec 2025 10-K filing date among cohort funds'); put('AuditLastFiled',f.max().strftime('%-d %B'),'str','last')
