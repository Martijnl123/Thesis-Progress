# Step 3 numbers for draft 2: traded against private loans, classes rebuilt on 17 Nov (v2_step3_classes_2026-11-17.py):
# all positions searched in N-PORT (#68), traded set at tranche level (#25). Exploratory.
from d2_common import *
import statsmodels.formula.api as smf

def frame(p,cls_file='step3_classes_2026-11-17.parquet',labmode='model'):
    C=pd.read_parquet(f'{V}/{cls_file}').set_index('pkey')
    keys=set(C.index); d=frozen(p,keys)
    w=d.pivot_table(index='pkey',columns=d.ddate.dt.strftime('%Y%m'),values='mark')
    s=C.copy()
    if labmode=='fund':
        fl=p[(p.ddate==BASE)].drop_duplicates('pkey').set_index('pkey').fsec.reindex(s.index)
        s=s[fl.notna()&~fl.isin(['it_services','tech_other'])].copy(); s['sw']=(fl.reindex(s.index)=='software').astype(int)
    for q in ['202506','202512','202603','202606']:
        s[f'dm{q}']=(w[q]-w['202509']).reindex(s.index) if q>'202509' else (w['202509']-w[q]).reindex(s.index)
    s['unch']=((w['202603']-w['202512']).abs()<0.05).astype(float).where(w['202603'].notna()&w['202512'].notna()).reindex(s.index)
    s['ttm']=(pd.to_datetime(s.maturity)-BASE).dt.days/365.25
    bsz=s.groupby('bid').principal.sum(); s['ln_bsize']=np.log(s.bid.map(bsz))
    a=p[(p.ddate==BASE)&p.ok]; nf=a.groupby('loan').family.nunique(); s['nfam']=s.loan.map(nf)
    s['spr']=s.spread_pre*100; s['pikp']=s.pik_pre.fillna(0); s['cik']=s.cik.astype(str); s['T_']=(s.cls=='traded').astype(int)
    s['clu']='b'+s.bid.astype(str); s['fs']=s.cik+'_'+s.sw.astype(str)
    return s

def ols(f,x,clu='clu'):
    return smf.ols(f,x).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(x[clu])[0]})

def run(p,T,log):
    s=frame(p); S='Step 3 classes (17 Nov): cohort, all positions searched, traded at tranche level, held at each date'
    for sw,SW in [(1,'Sw'),(0,'Ot')]:
        for c,Cn in [('traded','Tr'),('private','Pr'),('L3 only','Lt'),('traded borrower','Tb')]:
            x=s[(s.sw==sw)&(s.cls==c)]; put(f'NCls{SW}{Cn}',len(x),'int',S); put(f'NCls{SW}{Cn}Borr',x.bid.nunique(),'int',S)
            put(f'Chg{SW}{Cn}Jun',x.dm202606.mean(),'f1',S+', mean change Sep25-Jun26',sign=True); put(f'Chg{SW}{Cn}Mar',x.dm202603.mean(),'f1',S+', to Mar26',sign=True)
            put(f'Med{SW}{Cn}Jun',x.dm202606.median(),'f1',S+', median',sign=True)
            put(f'Five{SW}{Cn}',(x.dm202606.dropna()<-5).mean(),'pct0',S+', share down more than 5 by June')
            put(f'Unch{SW}{Cn}',x.unch.mean(),'pct0',S+', share unchanged Dec25-Mar26')
            put(f'Mark{SW}{Cn}',x.mark.mean(),'f1',S+', Sep 2025 mean mark'); put(f'Spr{SW}{Cn}',x.spr.mean()*100,'f0',S+', spread bp')
            put(f'Pik{SW}{Cn}',x.pikp.mean(),'pct0',S+', PIK share'); put(f'Hold{SW}{Cn}',x.nfam.mean(),'f1',S+', managers holding the loan')
            put(f'NJun{SW}{Cn}',x.dm202606.notna().sum(),'int','positions held in June')
    # Step 1 within groups (event study on the frozen cohort split by class)
    keys0=cohort_keys(p); d0=frozen(p,keys0); d0['sw']=(d0.grp=='SOFT').astype(float); d0['cls']=d0.pkey.map(s.cls)
    for nm,m in [('Tr',d0.cls=='traded'),('Pr',d0.cls=='private'),('Np',d0.cls.isin(['private','L3 only','traded borrower']))]:
        r=es(d0[m]);
        for q,Q in [('2026-03','Mar'),('2026-06','Jun')]:
            putse(f'Within{nm}Mark{Q}',*get(r,'mark',q),S+f', H1 event study within {nm}'); putse(f'Within{nm}Below{Q}',*get(r,'b99',q),S+f', H1 within {nm}',d=1)
        put(f'NWithin{nm}',r['n']['pos'],'int','positions'); log(f'[step3] within {nm}: '+sample_note(r,nm)+' Jun mark %+.2f (%.2f)'%get(r,'mark','2026-06'))
    # traded vs private, fund x sector FE (#40): dm = a_fs + b T + g T x Soft
    u=s[s.cls.isin(['traded','private'])].copy(); u['Tsw']=u.T_*u.sw
    res={}
    for q,Q in [('202603','Mar'),('202606','Jun')]:
        x=u.dropna(subset=[f'dm{q}']); r=ols(f'dm{q} ~ T_ + Tsw + C(fs)',x); b,se_=r.params['Tsw'],r.bse['Tsw']
        putse(f'Gap{Q}',b,se_,S+', traded vs private only, fund x sector FE, borrower-clustered'); put(f'Gap{Q}Lo',b-1.96*se_,'f1','95% CI',sign=True); put(f'Gap{Q}Hi',b+1.96*se_,'f1','95% CI',sign=True)
        putse(f'GapT{Q}',r.params['T_'],r.bse['T_'],'traded effect for other loans'); put(f'NGap{Q}',len(x),'int','positions'); res[Q]=(b,se_)
        # private-only Step-1-style comparison: software vs other private loans, fund FE (#69)
        y=x[x.cls=='private']; rp=ols(f'dm{q} ~ sw + C(cik)',y); putse(f'PrivOnly{Q}',rp.params['sw'],rp.bse['sw'],S+', private loans only, software vs other, fund FE')
        y=x[x.cls=='traded']; rt=ols(f'dm{q} ~ sw + C(cik)',y); putse(f'TrOnly{Q}',rt.params['sw'],rt.bse['sw'],S+', traded loans only, software vs other, fund FE')
    x=u.dropna(subset=['dm202606']).copy()
    # equal-share trim (#64): drop the 5% largest falls in each class x sector cell
    keep=x.groupby(['cls','sw']).dm202606.transform(lambda v:v>v.quantile(0.05))
    r=ols('dm202606 ~ T_ + Tsw + C(fs)',x[keep]); putse('GapTrim',r.params['Tsw'],r.bse['Tsw'],S+', June, 5% largest falls dropped in each class x sector cell')
    nb=x[(~keep)&(x.sw==1)&(x.cls=='traded')].bid.nunique(); put('NTrimTrBorr',nb,'int','traded software borrowers among the dropped 5%')
    lo,hi=x.dm202606.quantile([.05,.95]); x['yw']=x.dm202606.clip(lo,hi); r=ols('yw ~ T_ + Tsw + C(fs)',x); putse('GapWins',r.params['Tsw'],r.bse['Tsw'],S+', winsorised 5/95')
    # fall > 5 points
    x['f5']=100*(x.dm202606<-5); r=ols('f5 ~ T_ + Tsw + C(fs)',x); putse('GapFive',r.params['Tsw'],r.bse['Tsw'],S+', share down >5 (pp)',d=1)
    # overlap of holder counts (#63), 7+ bin
    x['hb']=pd.cut(x.nfam,[0,1,3,6,1000],labels=['1','2-3','4-6','7+'])
    ov=x[x.sw==1].groupby(['hb','cls']).dm202606.agg(['mean','size']).unstack('cls'); T['overlap']=ov; log('[step3] overlap by holder bin, software:\n'+ov.round(2).to_string())
    sw=x[x.sw==1]
    put('TrSevenShare',(sw[sw.cls=='traded'].nfam>=7).mean(),'pct0','traded software positions in loans held by 7+ managers')
    put('PrSevenShare',(sw[sw.cls=='private'].nfam>=7).mean(),'pct1','private software positions in loans held by 7+ managers')
    put('PrOneShare',(sw[sw.cls=='private'].nfam<=1).mean(),'pct0','private software positions held by one manager')
    put('TrOneShare',(sw[sw.cls=='traded'].nfam<=1).mean(),'pct0','traded software positions held by one manager')
    # specification count (T5): software traded minus private, minus the same gap for other loans, in each cut
    cuts=[]
    def gapcut(z,lab):
        g=lambda sv,c: z[(z.sw==sv)&(z.cls==c)].dm202606
        if min(len(g(1,'traded')),len(g(1,'private')),len(g(0,'traded')),len(g(0,'private')))<10: return
        cuts.append((lab,(g(1,'traded').mean()-g(1,'private').mean())-(g(0,'traded').mean()-g(0,'private').mean())))
    gapcut(x,'all')
    x['sb']=pd.qcut(x.ln_bsize,3,labels=['small','mid','large']); x['spb']=pd.cut(x.spr,[0,4,5,6,30],labels=['<4','4-5','5-6','6+'])
    x['mb']=pd.cut(x.mark,[0,97,99,101,200],labels=['<97','97-99','99-101','>101'])
    for col in ['hb','sb','spb','mb','vtype','lien']:
        for lv in x[col].dropna().unique(): gapcut(x[x[col]==lv],f'{col}={lv}')
    C=pd.DataFrame(cuts,columns=['cut','dd']); T['cuts']=C; put('NCuts',len(C),'int','cuts with 10+ positions in each of the four cells'); put('NCutsNeg',int((C.dd<0).sum()),'int','cuts where the software traded-private gap exceeds the other-loan gap')
    log('[step3] cuts:\n'+C.round(2).to_string())
    # catch-up (Mar->Jun) and pre-trend (Jun->Sep 2025)
    z=s[(s.sw==1)&s.cls.isin(['traded','private'])].dropna(subset=['dm202603','dm202606'])
    for c,Cn in [('traded','Tr'),('private','Pr')]: put(f'Catch{Cn}',(z[z.cls==c].dm202606-z[z.cls==c].dm202603).mean(),'f1','software, change Mar->Jun 2026, held at both',sign=True)
    z=s[(s.sw==1)&s.cls.isin(['traded','private'])].dropna(subset=['dm202506'])
    r=ols('dm202506 ~ T_ + C(cik)',z); putse('PreTr',r.params['T_'],r.bse['T_'],'software, change Jun->Sep 2025, traded minus private, fund FE')
    # funds' labels (#29)
    sf=frame(p,labmode='fund'); uf=sf[sf.cls.isin(['traded','private'])].copy(); uf['Tsw']=uf.T_*uf.sw; xf=uf.dropna(subset=['dm202606'])
    r=ols('dm202606 ~ T_ + Tsw + C(fs)',xf); putse('GapFund',r.params['Tsw'],r.bse['Tsw'],'Step 3 with funds labels, June'); put('NGapFund',len(xf),'int','positions')
    rp=ols('dm202606 ~ sw + C(cik)',xf[xf.cls=='private']); putse('PrivOnlyFund',rp.params['sw'],rp.bse['sw'],'private only, funds labels, June')
    # concentration: without the 20 software borrowers with the largest falls (traded side)
    tb=x[(x.sw==1)&(x.cls=='traded')].groupby('bid').dm202606.mean().sort_values().index[:20]
    r=ols('dm202606 ~ T_ + Tsw + C(fs)',x[~x.bid.isin(tb)]); putse('GapNoTwenty',r.params['Tsw'],r.bse['Tsw'],'June gap without the 20 traded software borrowers with the largest falls')
    T['s3']=s
    log('[step3] gap Mar %+.2f (%.2f), Jun %+.2f (%.2f)'%(*res['Mar'],*res['Jun']))
    return s
