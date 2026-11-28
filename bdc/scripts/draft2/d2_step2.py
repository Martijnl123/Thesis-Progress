# Step 2 numbers for draft 2: BDC marks against the N-PORT Level 2 price of the same loan.
# Main sample: the registered 8 Nov coupon-fixed match (nport_matches_coupon.parquet), positions with a verified Level 2 match at
# 30 Sep 2025, cohort positions with a model label other than IT services / data / hardware, held at each date.
from d2_common import *
import statsmodels.formula.api as smf
QQ=[('202506','202509'),('202509','202512'),('202512','202603'),('202603','202606')]
QN={'202509':'Sep','202512':'Dec','202603':'Mar','202606':'Jun'}

def build(p,matchfile,labmode='model'):
    M=pd.read_parquet(f'{V}/{matchfile}'); assert M.ddate.max()<=CAP; M=M[M.verified&(M.lvl=='2')]
    px=M.groupby('pos_id').price_median.median().rename('px')
    if labmode=='model':
        lab=p.grp.notna()&~p.grp.isin(['UNKN','FUND','ITSV','DATA','HARD']); swv=(p.grp=='SOFT')
        keys=set(p.loc[(p.ddate==BASE)&p.ok&lab,'pkey'])
    else:   # funds' labels on the cohort
        lab=p.fsec.notna()&~p.fsec.isin(['it_services','tech_other']); swv=(p.fsec=='software')
        keys=set(p.loc[(p.ddate==BASE)&p.ok&lab,'pkey'])&cohort_keys(p)
    q=p[p.pkey.isin(keys)].join(px,on='pos_id'); q=q[q.mark.between(0,110)]
    q['sw']=swv.reindex(q.index).astype(float)
    bs=set(q.loc[(q.ddate==BASE)&q.px.notna(),'pkey']); q=q[q.pkey.isin(bs)&(q.ddate>='2025-06-30')].copy(); q['gap']=q.mark-q.px
    return q

def es2(d,y,fe=('pkey','fq')):
    d=d.dropna(subset=[y]).copy(); ev=[x for x in sorted(d.ddate.unique()) if x!=BASE]; C=[]
    for x in ev: c='s'+x.strftime('%Y%m'); d[c]=d.sw*(d.ddate==x); C.append(c)
    dd,b,sc,Ainv=fit(d,y,C,list(fe)); s=se(dd,sc,Ainv,'clu')
    return {c[1:]:(float(b[c]),float(s[i])) for i,c in enumerate(C)}, dd

def wide(q):
    w=q.pivot_table(index='pkey',columns=q.ddate.dt.strftime('%Y%m'),values=['mark','px']); w.columns=[f'{a}_{b}' for a,b in w.columns]
    return w.join(q.drop_duplicates('pkey').set_index('pkey')[['sw','bid','family','cik','clu']])
def pa_frame(w,t0,t1):
    return pd.DataFrame({'dm':w[f'mark_{t1}']-w[f'mark_{t0}'],'g0':w[f'px_{t0}']-w[f'mark_{t0}'],'gap':w[f'px_{t1}']-w[f'mark_{t0}'],
                         'dp':w[f'px_{t1}']-w[f'px_{t0}'],'soft':w.sw,'bid':w.bid,'clu':w.clu,'family':w.family,'cik':w.cik,'t':t1,
                         'same':(w[f'px_{t1}']-w[f'mark_{t1}']).abs()<0.01}).dropna(subset=['dm','g0','gap'])
def pa(x):
    r=smf.ols('dm~gap*soft',x).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(x.clu)[0]})
    return r.params['gap'],r.params['gap:soft'],r.bse['gap:soft'],int(x.soft.sum()),len(x)

def run(p,T,log):
    q=build(p,'nport_matches_coupon.parquet'); S='Step 2 sample: 8 Nov coupon-fixed N-PORT match, verified Level 2 price at Sep 2025, held at each date'
    s9=q[q.ddate==BASE]; put('NStepTwoPos',len(s9),'int',S); put('NStepTwoSw',int(s9.sw.sum()),'int',S); put('NStepTwoSwBorr',s9[s9.sw==1].bid.nunique(),'int',S)
    put('NStepTwoBorr',s9.bid.nunique(),'int',S); put('NStepTwoFunds',s9.cik.nunique(),'int',S); put('NStepTwoMgr',s9.family.nunique(),'int',S)
    # raw changes, held and priced at both dates (held-at-each-date sample, #8)
    w=wide(q)
    for sw,nm in [(1,'Sw'),(0,'Ot')]:
        x=w[(w.sw==sw)&w.px_202606.notna()&w.mark_202606.notna()]
        put(f'RawMark{nm}Jun',(x.mark_202606-x.mark_202509).mean(),'f1',S+', priced at Sep 2025 and Jun 2026',sign=True)
        put(f'RawPx{nm}Jun',(x.px_202606-x.px_202509).mean(),'f1',S+', priced at Sep 2025 and Jun 2026',sign=True)
        put(f'NRaw{nm}Jun',len(x),'int','positions priced at both dates')
    # theta_k
    th,dd=es2(q,'gap'); mk,_=es2(q[q.px.notna()],'mark'); pr,_=es2(q,'px')
    T['theta']={'gap':th,'mark':mk,'px':pr}
    put('NThetaObs',len(dd),'int',S+', position-quarters in the gap regression')
    for k in ['202506','202512','202603','202606']:
        Q={'202506':'JunPrev','202512':'Dec','202603':'Mar','202606':'Jun'}[k]
        putse(f'Theta{Q}',*th[k],S); putse(f'ThMark{Q}',*mk[k],S); putse(f'ThPx{Q}',*pr[k],S)
    b,s=th['202606']; put('ThetaJunLo',b-1.96*s,'f1',S+', 95% CI',sign=True); put('ThetaJunHi',b+1.96*s,'f1',S+', 95% CI',sign=True)
    # partial adjustment
    rows=[]
    for t0,t1 in QQ:
        x=pa_frame(w,t0,t1)
        for nm,yy in [('trim',x[x.g0.abs()<=10]),('nosame',x[(x.g0.abs()<=10)&~x.same]),('notrim',x)]:
            lam,dl,dls,ns,n=pa(yy); rows.append((t1,nm,lam,dl,dls,ns,n))
    P=pd.DataFrame(rows,columns=['t','v','lam','dl','se','nsw','n']); T['pa']=P
    for _,r in P.iterrows():
        tag={'trim':'','nosame':'Nosame','notrim':'Notrim'}[r.v]; Q=QN[r.t]
        putse(f'Dl{tag}{Q}',r.dl,r.se,S+f', partial adjustment {r.t}, {r.v}'); put(f'Lam{tag}{Q}',r.lam,'f2',S+' lambda other')
        put(f'NDl{tag}{Q}',r.nsw,'int','software pairs')
    xs=pd.concat([pa_frame(w,a,b_) for a,b_ in QQ[1:]]); xt=xs[xs.g0.abs()<=10]
    r=smf.ols('dm~gap*soft+C(t)',xt).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(xt.clu)[0]})
    putse('DlPool',r.params['gap:soft'],r.bse['gap:soft'],S+', pooled Sep25-Jun26, quarter dummies, 10-point trim'); put('LamPool',r.params['gap'],'f2','lambda other pooled')
    put('NDlPool',len(xt),'int','pairs pooled'); put('NDlPoolSw',int(xt.soft.sum()),'int','software pairs pooled')
    b,s=r.params['gap:soft'],r.bse['gap:soft']; put('DlPoolLo',b-1.96*s,'f2','95% CI',sign=True); put('DlPoolHi',b+1.96*s,'f2','95% CI',sign=True)
    # pooled with fund x quarter FE (#38)
    xt=xt.copy(); xt['fq']=xt.cik.astype(str)+'_'+xt.t; xt['gs']=xt.gap*xt.soft
    dd_,bb,sc,Ainv=fit(xt,'dm',['gap','gs'],['fq']); ss=se(dd_,sc,Ainv,'clu'); putse('DlPoolFq',float(bb['gs']),float(ss[1]),S+', pooled with fund x quarter FE')
    # untrimmed pooled
    rn=smf.ols('dm~gap*soft+C(t)',xs).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(xs.clu)[0]}); putse('DlPoolNotrim',rn.params['gap:soft'],rn.bse['gap:soft'],S+', pooled, no trim')
    # G split (#38): dm on dP and g0 separately; equal under partial adjustment
    xg=xt.dropna(subset=['dp']).copy()
    xg['dps']=xg.dp*xg.soft; xg['g0s']=xg.g0*xg.soft
    rg=smf.ols('dm~dp+g0+soft+dps+g0s+C(t)',xg).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(xg.clu)[0]})
    putse('SplitDp',rg.params['dp'],rg.bse['dp'],'G split: coefficient on this quarter price change, other loans'); putse('SplitGz',rg.params['g0'],rg.bse['g0'],'G split: coefficient on last quarter gap, other loans')
    putse('SplitDpSw',rg.params['dps'],rg.bse['dps'],'software difference on dP'); putse('SplitGzSw',rg.params['g0s'],rg.bse['g0s'],'software difference on g0')
    wt=rg.wald_test('dp = g0',scalar=True); put('SplitPOther',float(wt.pvalue),'f3','p-value equal coefficients, other loans')
    wt2=rg.wald_test('dps = g0s',scalar=True); put('SplitPSw',float(wt2.pvalue),'f3','p-value equal software differences')
    # var(G) by group and quarter (#39)
    vg=xt.groupby(['t','soft']).gap.var().unstack(); T['varG']=vg; log('[step2] var(G) by quarter (rows) and software (cols):\n'+vg.round(2).to_string())
    put('VarGRatioMax',float((vg[1.0]/vg[0.0]).max()),'f1','max ratio of var(G) software/other across shock quarters')
    put('VarGRatioMin',float((vg[1.0]/vg[0.0]).min()),'f1','min ratio')
    # second-price instrument (#39): only useful if two N-PORT series price the loan differently
    M=pd.read_parquet(f'{V}/nport_matches_coupon.parquet'); M=M[M.verified&(M.lvl=='2')&M.pos_id.isin(set(q.pos_id))]
    Q_=pd.read_parquet(f'{V}/nport_qe_loans.parquet'); assert Q_.report_date.max()<=CAP; Q_=Q_[Q_.lvl=='2'].rename(columns={'report_date':'ddate','maturity':'mat_n','rate':'rate_n'})
    k_=['ddate','mat_n','rate_n','price_median','n_series']; J=M.merge(Q_[k_+['price_min','price_max']].drop_duplicates(k_),on=k_,how='left')
    J=J[(J.ddate>BASE)&(J.n_series>=2)]
    put('SecondPxShare',(M[M.ddate>BASE].n_series>=2).mean(),'pct0','Step 2 matches Dec25-Jun26 priced by two or more N-PORT series')
    put('SecondPxSame',((J.price_max-J.price_min)<0.01).mean(),'pct0','of those, share where all series report the same price (within 0.01)')
    # identical pairs
    z=q[q.px.notna()&(q.ddate>BASE)]; put('ShareSame',(z.gap.abs()<0.01).mean(),'pct0',S+', share of position-quarters with mark = price within 0.01, Dec25-Jun26')
    # lost quotes (#57)
    held=w[w.mark_202606.notna()]
    for sw,nm in [(1,'Sw'),(0,'Ot')]:
        put(f'Keep{nm}',held[held.sw==sw].px_202606.notna().mean(),'pct0','share of positions priced at Sep 2025 and still held in June that still have a price')
    qq=q.sort_values(['pkey','ddate']).copy(); qq['pxl']=qq.groupby('pkey').px.ffill()
    chg=qq.groupby('pkey').px.diff(); qq['dpx']=chg
    lowq=qq[qq.sw==1].groupby('ddate').dpx.quantile(0.10)
    # low price: last observed price minus the 90th-percentile software price fall of that quarter (cumulated since the last quote)
    qq['lowstep']=qq.ddate.map(lowq).clip(upper=0)
    qq['miss']=qq.px.isna()&qq.pxl.notna()
    qq['cumlow']=qq.groupby('pkey').apply(lambda g:(g.lowstep*g.miss).cumsum()).reset_index(level=0,drop=True)
    for nm,pxv in [('Stale',qq.px.fillna(qq.pxl)),('Low',qq.px.fillna(qq.pxl+qq.cumlow))]:
        x=qq.copy(); x['px']=pxv; x['gap']=x.mark-x.px; t_,_=es2(x,'gap'); putse(f'ThetaJun{nm}',*t_['202606'],S+f', lost quotes imputed ({nm})')
    # Jun -> Sep 2025 decomposition (#60)
    x=pa_frame(w,'202506','202509'); x=x[x.g0.abs()<=10]; lam,dl,dls,ns,n=pa(x)
    mg=[]
    for fam in x.family.dropna().unique():
        y=x[x.family!=fam]
        if y.soft.sum()>=10: mg.append((fam,pa(y)[1]))
    mg=pd.DataFrame(mg,columns=['fam','dl']).sort_values('dl'); T['jun25_loo']=mg
    put('DlJunLooLo',mg.dl.min(),'f2','Jun-Sep 2025 dlambda, leave one manager out, min',sign=True); put('DlJunLooHi',mg.dl.max(),'f2','max',sign=True)
    top=mg.iloc[-1]; put('DlJunLooTopName',str(top.fam).title(),'str','manager whose removal moves dlambda most towards zero')
    xs_=x[x.soft==1].assign(a=lambda z:z.gap.abs()).sort_values('a',ascending=False)
    for k in [10,20]:
        put(f'DlJunDropTop{["","Ten","Twenty"][k//10]}',pa(x.drop(xs_.index[:k]))[1],'f2',f'Jun-Sep 2025 dlambda without the {k} software pairs with the largest |G|',sign=True)
    sh=(x[x.soft==1].family.value_counts(normalize=True)); put('DlJunTopMgrShare',float(sh.iloc[0]),'pct0','largest manager share of software pairs Jun-Sep 2025'); put('DlJunTopMgrName',str(sh.index[0]).title(),'str','that manager')
    put('DlJunGapZero',(x[x.soft==1].gap.abs()<0.01).mean(),'pct0','share of software pairs Jun-Sep 2025 with zero gap to the new price')
    log(f'[step2] Jun-Sep 2025 dlambda {dl:+.2f} ({dls:.2f}); LOO range {mg.dl.min():+.2f} to {mg.dl.max():+.2f}; manager shares {sh.head(5).round(2).to_dict()}')
    # alternative samples: funds' labels (#29) and all-positions match
    for nm,qq2 in [('Fund',build(p,'nport_matches_coupon.parquet','fund')),('All',build(p,'nport_matches_all_2026-11-17.parquet'))]:
        t_,_=es2(qq2,'gap'); putse(f'ThetaJun{nm}',*t_['202606'],f'Step 2, {nm} version')
        w2=wide(qq2); xs2=pd.concat([pa_frame(w2,a,b_) for a,b_ in QQ[1:]]); xs2=xs2[xs2.g0.abs()<=10]
        r2=smf.ols('dm~gap*soft+C(t)',xs2).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(xs2.clu)[0]}); putse(f'DlPool{nm}',r2.params['gap:soft'],r2.bse['gap:soft'],f'Step 2 pooled dlambda, {nm} version')
        s9_=qq2[qq2.ddate==BASE]; put(f'NStepTwo{nm}',len(s9_),'int',f'positions, {nm} version'); put(f'NStepTwo{nm}Sw',int(s9_.sw.sum()),'int','software')
    T['q2']=q
    log('[step2] theta gap: '+'; '.join(f'{k} {v[0]:+.2f} ({v[1]:.2f})' for k,v in th.items()))
    log('[step2] partial adjustment:\n'+P.round(3).to_string())
    return q
