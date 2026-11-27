# Analyses added for draft 3 (18 Nov 2026), from the draft 3 revision prompt, section C (C1 to C17).
# Called from v3_numbers.py after the draft 2 stages. Data through 30 June 2026 only (the loaders assert it).
# Nothing registered is changed: the Step 3 classes file is read as it is, and C10/C11 are robustness rows next to it.
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..','..')))  # repo root
import sys, re
sys.path.insert(0,(_ROOT+'/bdc/scripts/draft2'))
from d2_common import *
import d2_common as CM
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import norm
from ai_rev2_lib import vcov

def ols(f,x,clu='clu',w=None):
    g=pd.factorize(x[clu])[0]
    if w is None: return smf.ols(f,x).fit(cov_type='cluster',cov_kwds={'groups':g})
    return smf.wls(f,x,weights=x[w]).fit(cov_type='cluster',cov_kwds={'groups':g})

# ---------------------------------------------------------------- C1 common shock (PAP v5 line 68 (a))
def common_shock(p,T,log):
    keys=cohort_keys(p); d0=frozen(p,keys); d0['sw']=(d0.grp=='SOFT').astype(float)
    v=d0.sort_values(['pkey','ddate']).copy(); g=v.groupby('pkey')
    v['dmark']=g.mark.diff(); v['db99']=g.b99.diff(); v['gapd']=g.ddate.diff().dt.days; v=v[v.gapd.between(80,100)]
    S='frozen cohort (model labels), within-position quarterly changes, mean by quarter for software and other loans'
    qs=sorted(v.ddate.unique()); pre=[q for q in qs if q<=BASE]; assert len(pre)==11, len(pre)
    out={}
    for y,Y,dec in [('db99','Below',1),('dmark','Mark',2)]:
        m=v.groupby(['ddate','sw'])[y].mean().unstack(); m.columns=['ot','sw']
        x=m.loc[pre]; X=sm.add_constant(x.ot); r=sm.OLS(x.sw,X).fit()
        put(f'CsSlope{Y}',r.params.ot,'f2',S+', pre-period slope (11 quarters)'); put(f'CsSlope{Y}SE',r.bse.ot,'f2',S)
        for q,Q in [(MAR,'Mar'),(JUN,'Jun')]:
            xo=m.loc[q,'ot']; pr=r.get_prediction(np.array([[1.0,xo]])); f=pr.summary_frame(alpha=0.05)
            pred=float(f['mean'].iloc[0]); sef=float(np.sqrt(pr.var_pred_mean[0]+r.scale))
            act=float(m.loc[q,'sw'])
            put(f'CsPred{Y}{Q}',pred,f'f{dec}',S+f', predicted software change {Q} at the pre-period slope',sign=True)
            put(f'CsPred{Y}{Q}SE',sef,f'f{dec}',S+', forecast SE (parameter + residual)')
            put(f'CsAct{Y}{Q}',act,f'f{dec}',S+f', actual software change {Q}',sign=True)
            put(f'CsOt{Y}{Q}',xo,f'f{dec}',S+f', other-loan change {Q}',sign=True)
            rej=(act>pred+2*sef) if y=='db99' else (act<pred-2*sef)
            put(f'CsRej{Y}{Q}','rejected' if rej else 'not rejected','str','registered rule: high beta rejected if actual beyond prediction plus two SE')
            out[(Y,Q)]=(pred,sef,act,rej)
        lo,hi=x.ot.min(),x.ot.max()
        put(f'CsPreLo{Y}',lo,f'f{dec}','min pre-period other-loan change',sign=True); put(f'CsPreHi{Y}',hi,f'f{dec}','max',sign=True)
        put(f'CsOutside{Y}','outside' if not (lo<=m.loc[MAR,'ot']<=hi) else 'inside','str','March other-loan change vs pre-period range')
        log(f'[C1] {y}: slope {r.params.ot:.2f} ({r.bse.ot:.2f}); pre range {lo:+.2f}..{hi:+.2f}; '+'; '.join(f'{Q}: pred {out[(Y,Q)][0]:+.2f} ({out[(Y,Q)][1]:.2f}) act {out[(Y,Q)][2]:+.2f} rej {out[(Y,Q)][3]}' for Q in ['Mar','Jun'])+'\n'+m.round(3).to_string())
    T['cs']=out
    # market-wide loan move: median N-PORT Level 2 price, all loans, and other traded BDC marks (Step 2 sample)
    N=pd.read_parquet(f'{V}/nport_qe_loans.parquet',columns=['report_date','lvl','price_median']); assert N.report_date.max()<=CAP
    N=N[N.lvl=='2']
    for q,Q in [(BASE,'Sep'),(DEC,'Dec'),(MAR,'Mar'),(JUN,'Jun')]:
        put(f'NpMed{Q}',N[N.report_date==q].price_median.median(),'f2','median N-PORT Level 2 loan price, all loan rows, quarter-end')
    q2=T['q2']; w=q2.pivot_table(index='pkey',columns=q2.ddate.dt.strftime('%Y%m'),values='mark').join(q2.drop_duplicates('pkey').set_index('pkey').sw)
    o=w[w.sw==0]; put('OtTrMarkMar',(o['202603']-o['202509']).mean(),'f1','Step 2 sample, other loans, mean BDC mark change Sep25-Mar26, held at both',sign=True)
    put('OtTrMarkJun',(o['202606']-o['202509']).mean(),'f1','same to Jun26',sign=True)
    log('[C1] N-PORT medians: '+', '.join(f'{Q} {MAC["NpMed"+Q][0]}' for Q in ['Sep','Dec','Mar','Jun']))

# ---------------------------------------------------------------- Step 3 additions (C2, C3, C4, C5, C8, C9, C10, C11)
def _gap(x,clu='clu',w=None):
    r=ols('dm202606 ~ T_ + Tsw + C(fs)',x,clu,w); return float(r.params['Tsw']),float(r.bse['Tsw'])

def step3_extra(p,T,log):
    s=T['s3'].copy(); s=s.reset_index() if 'pkey' not in s.columns else s
    S='Step 3 classes (17 Nov, registered file), traded vs private, June 2026, fund x sector FE, borrower-clustered'
    u=s[s.cls.isin(['traded','private'])].copy(); u['Tsw']=u.T_*u.sw; x=u.dropna(subset=['dm202606']).copy()
    # ---- C2 overlap and within-bin gaps
    x['hb']=pd.cut(x.nfam,[0,1,6,1000],labels=['One','Mid','Seven'])
    x['sb']=pd.qcut(x.ln_bsize,3,labels=['Small','Middle','Large'])
    for col,levels in [('hb',['One','Mid','Seven']),('sb',['Small','Middle','Large'])]:
        for lv in levels:
            y=x[x[col]==lv]
            for sw_,SW in [(1,'Sw'),(0,'Ot')]:
                for c,Cn in [('traded','Tr'),('private','Pr')]:
                    put(f'Bin{lv}{SW}{Cn}',int(((y.sw==sw_)&(y.cls==c)).sum()),'int',f'Step 3 positions held in June, bin {col}={lv}')
            nT=((y.sw==1)&(y.cls=='traded')).sum(); nP=((y.sw==1)&(y.cls=='private')).sum()
            if min(nT,nP)>=20:
                b,se_=_gap(y); putse(f'GapBin{lv}',b,se_,S+f', within bin {col}={lv}')
            else: put(f'GapBin{lv}','not estimated (fewer than 20 software positions in a class)','str','bin rule')
            log(f'[C2] bin {col}={lv}: sw traded {nT}, sw private {nP}')
    y=x[x.nfam<=6]; b,se_=_gap(y); putse('GapBinOneSix',b,se_,S+', loans held by 1 to 6 managers')
    put('BinOneSixSwTr',int(((y.sw==1)&(y.cls=='traded')).sum()),'int','software traded positions, 1-6 holders'); put('BinOneSixSwPr',int(((y.sw==1)&(y.cls=='private')).sum()),'int','software private, 1-6 holders')
    x['wb']=1/x.groupby(['bid','cls']).pkey.transform('count'); b,se_=_gap(x,w='wb'); putse('GapBorrW',b,se_,S+', each borrower weight 1 within class')
    x['fam']=x.family.astype(str); b,se_=_gap(x,clu='fam'); putse('GapMgrClu',b,se_,S+', clustered by manager'); put('NGapMgr',x.fam.nunique(),'int','managers')
    bs=s.groupby('bid').principal.sum()
    for c,Cn in [('traded','Tr'),('private','Pr')]:
        bb=x[(x.sw==1)&(x.cls==c)].bid.unique(); put(f'MedBorrPrin{Cn}',bs.reindex(bb).median()/1e6,'f0',f'median BDC principal per software borrower in the June estimation sample, {c}, $m, Sep 2025 Step 3 positions')
    # ---- C3 breadth and symmetric trim
    for c,Cn in [('traded','Tr'),('private','Pr')]:
        bm=x[(x.sw==1)&(x.cls==c)].groupby('bid').dm202606.mean()
        put(f'BreadthSw{Cn}',(bm<-5).mean(),'pct0',f'share of {c} software borrowers whose mean June change is below -5'); put(f'BorrMedSw{Cn}',bm.median(),'f1',f'median borrower mean change, {c} software',sign=True)
        put(f'NBorrSw{Cn}',len(bm),'int','borrowers held in June')
    def top(c,sv,k=20): return set(x[(x.sw==sv)&(x.cls==c)].groupby('bid').dm202606.mean().sort_values().index[:k])
    drop=set().union(*[top(c,sv) for c in ['traded','private'] for sv in [0,1]])
    b,se_=_gap(x[~x.bid.isin(drop)]); putse('GapSymTrim',b,se_,S+', 20 worst borrowers dropped in each of the four class x sector cells')
    # ---- C4 exits
    fz=frozen(p,set(u.pkey)); last=fz.sort_values('ddate').groupby('pkey').agg(lastd=('ddate','max'),lastm=('mark','last'))
    u2=u.set_index('pkey').join(last); u2['exit']=u2.lastd<JUN; u2['chg_last']=u2.lastm-u2.mark
    for sw_,SW in [(1,'Sw'),(0,'Ot')]:
        for c,Cn in [('traded','Tr'),('private','Pr')]:
            put(f'ExitS{SW}{Cn}',u2[(u2.sw==sw_)&(u2.cls==c)].exit.mean(),'pct0','share of Step 3 positions gone by June 2026')
    u2['dm_cf']=u2.dm202606.fillna(u2.chg_last); xb=u2.dropna(subset=['dm_cf']).reset_index()
    r=ols('dm_cf ~ T_ + Tsw + C(fs)',xb); putse('GapExitCarry',r.params['Tsw'],r.bse['Tsw'],S+', exits carried at their last observed mark change'); put('NGapExitCarry',len(xb),'int','positions')
    p10=fz[fz.ddate==JUN].mark.quantile(0.10); put('SThreePTen',p10,'f1','10th percentile of June 2026 marks, Step 3 positions held')
    for tag,cond in [('Pr',(u2.cls=='private')&(u2.sw==1)),('Tr',(u2.cls=='traded')&(u2.sw==1))]:
        z=u2.copy(); low=z.exit&cond; z['dm_b']=np.where(low,p10-z.mark,z.dm202606.fillna(z.chg_last)); zb=z.dropna(subset=['dm_b']).reset_index()
        r=ols('dm_b ~ T_ + Tsw + C(fs)',zb); putse(f'GapExitLow{tag}',r.params['Tsw'],r.bse['Tsw'],S+f', {tag} software leavers at the June 10th-percentile mark, other leavers at last mark change')
    # ---- C5 sector rank of kappa
    uu=u[u.grp.notna()&~u.grp.isin(['ITSV','DATA','HARD','UNKN','FUND'])].dropna(subset=['dm202606']).copy(); rows=[]
    for g in sorted(uu.grp.unique()):
        z=uu if g=='SOFT' else uu[uu.grp!='SOFT']
        z=z.copy(); z['g']=(z.grp==g).astype(int); z['Tg']=z.T_*z.g; z['fs2']=z.cik.astype(str)+'_'+z.g.astype(str)
        n=z.groupby(['g','cls']).size()
        if min(n.get((1,'traded'),0),n.get((1,'private'),0))<20: continue
        r=ols('dm202606 ~ T_ + Tg + C(fs2)',z); rows.append((g,float(r.params['Tg']),float(r.bse['Tg'])))
    R=pd.DataFrame(rows,columns=['grp','k','se']); R['t']=R.k/R.se; T['s3rank']=R
    from d2_rank import NAMES
    for col,Y in [('k','K'),('t','T')]:
        t=R.sort_values(col).reset_index(drop=True); put(f'SThreeRank{Y}',int(t.index[t.grp=='SOFT'][0])+1,'int',f'software rank on {col} (most negative first)')
        nx=t[t.grp!='SOFT'].iloc[0]; put(f'SThreeNext{Y}Name',NAMES[nx.grp].lower(),'str','next group'); put(f'SThreeNext{Y}',float(nx[col]),'f2','next group value',sign=True)
        if Y=='K':
            nx2=t[t.grp!='SOFT'].iloc[1]; put('SThreeNextTwoKName',NAMES[nx2.grp].lower(),'str','third group'); put('SThreeNextTwoK',float(nx2[col]),'f2','third',sign=True)
    put('SThreeNGroups',len(R),'int','groups with at least 20 positions in both classes'); put('SThreeRankP',1/len(R),'f3','1/groups')
    log('[C5] Step 3 rank:\n'+R.sort_values('k').round(2).to_string())
    # ---- C8 registered statistic rehearsed, one-quarter windows
    w=fz.pivot_table(index='pkey',columns=fz.ddate.dt.strftime('%Y%m'),values='mark')
    ss=s.set_index('pkey')
    win=[('SepDec','202509','202512'),('DecMar','202512','202603'),('MarJun','202603','202606')]
    conf=T.get('conf_ciks')
    for W,a,b_ in win:
        ss['dq']=(w[b_]-w[a]).reindex(ss.index)
        for c,Cn in [('private','P'),('traded','Q')]:
            z=ss[ss.cls==c].dropna(subset=['dq']).reset_index()
            r=ols('dq ~ sw + C(cik)',z); putse(f'Reh{Cn}{W}',r.params['sw'],r.bse['sw'],f'registered Step 3 statistic, {c}, {a}-{b_}, all funds, fund FE, borrower-clustered, held at both')
            put(f'NReh{Cn}{W}',len(z),'int','positions')
            if W=='MarJun' and conf is not None:
                zc=z[z.cik.astype(int).isin(conf)]; rc=ols('dq ~ sw + C(cik)',zc); putse(f'Reh{Cn}{W}Conf',rc.params['sw'],rc.bse['sw'],f'same, 76 confirmatory funds'); put(f'NReh{Cn}{W}Conf',len(zc),'int','positions')
        P_,Q_=MAC[f'RehP{W}'][1],MAC[f'RehQ{W}'][1]
        v_=('R1' if P_<=-0.5 and Q_<=1 else 'R2' if P_>-0.5 and Q_<=1 else 'R3' if P_>-0.5 and Q_>1 else 'none')
        put(f'RehVerdict{W}',v_,'str','registered rule applied to the rehearsal window')
    # ---- C9 power at Mar-Jun SEs (all funds)
    seP,seQ=MAC['RehPMarJunSE'][1],MAC['RehQMarJunSE'][1]; PW=[]
    for (Pt,Qt),nm in [((0,0),'Zero'),((-0.5,0),'Half'),((-1,0),'One'),((0,2),'Over')]:
        pP=norm.cdf((-0.5-Pt)/seP); pQ=1-norm.cdf((1-Qt)/seQ)
        pr={'R1':pP*(1-pQ),'R2':(1-pP)*(1-pQ),'R3':(1-pP)*pQ,'none':pP*pQ}
        for k,vv in pr.items(): put(f'PowS{nm}{k.replace("none","None").replace("R1","Ra").replace("R2","Rb").replace("R3","Rc")}',vv,'pct0',f'probability of verdict {k} when true (P,Q)=({Pt},{Qt}), normal approximation at the Mar-Jun SEs')
        PW.append((Pt,Qt,pr))
    T['s3power']=PW
    # ---- C10 private-class false negatives (robustness): hand-checked list below
    hits=fn_hits()
    z=x.copy(); z.loc[z.bid.isin(hits)&(z.cls=='private'),'cls']='traded borrower'; z=z[z.cls.isin(['traded','private'])].copy(); z['T_']=(z.cls=='traded').astype(int); z['Tsw']=z.T_*z.sw
    b,se_=_gap(z); putse('GapFn',b,se_,S+', hand-confirmed private-class false negatives moved to traded borrower')
    rp=ols('dm202606 ~ sw + C(cik)',z[z.cls=='private']); putse('PrivOnlyFn',rp.params['sw'],rp.bse['sw'],'private only, false negatives moved')
    put('NFnCand',len(fn_search()),'int','candidates of the token search for the hand check'); put('NFnHits',len(hits),'int','confirmed false-negative borrowers'); put('NFnPos',int(x[x.bid.isin(hits)&(x.cls=='private')].shape[0]),'int','private positions moved')
    # ---- C11 JV duplicates of CIK 1803498: collapse to fund x tranche
    z=x.copy(); z['cik']=z.cik.astype(str); jv=z.cik=='1803498'
    put('NJvCells',int(jv.sum()),'int','Step 3 positions of CIK 1803498 held in June')
    zc=pd.concat([z[~jv],z[jv].groupby(['cik','bid','tr','cls','sw'],as_index=False).agg(dm202606=('dm202606','mean'),T_=('T_','first'),Tsw=('Tsw','first'),fs=('fs','first'),clu=('clu','first'))])
    put('NJvCellsAfter',int((zc.cik=='1803498').sum()),'int','after collapsing to fund x borrower x tranche')
    b,se_=_gap(zc); putse('GapJv',b,se_,S+', CIK 1803498 collapsed to fund x borrower x tranche')
    log(f'[step3 extra] symmetric trim {MAC["GapSymTrim"][0]}; exit carry {MAC["GapExitCarry"][0]}; fn {MAC["GapFn"][0]}; jv {MAC["GapJv"][0]}')

def fn_hits():
    """C10: private software borrowers with a Level 2 issuer in N-PORT at 30 Sep 2025, found by fn_search() and confirmed by
    reading both names (18 Nov 2026). Only names judged the same company are listed; see draft3/decisions.md for every candidate."""
    return set(FN_CONFIRMED)
FN_CONFIRMED=[1757.0,1941.0]   # Cloud Software Group (Cloud Software Group Inc. term loans), Consilio (Skopima Consilio Parent LLC); hand check 18 Nov 2026

def fn_search(min_len=4):
    """Token search that keeps common words (no max_df cut). Returns candidates for the hand check."""
    C=pd.read_parquet(f'{V}/step3_classes_2026-11-17.parquet')
    B=pd.read_parquet(f'{F}/borrowers.parquet').set_index('bid')
    n=pd.read_parquet(f'{V}/nport_qe_loans.parquet'); n=n[(n.report_date==BASE)&(n.lvl=='2')]
    GEN=set('inc llc lp ltd limited corp corporation co company holdings holding group parent intermediate topco midco bidco buyer purchaser borrower acquisition acquisitions finance holdco the and of us usa dba fka i ii iii iv llp sarl plc gmbh bv first lien senior secured loan term revolver delayed draw new'.split())
    def toks(s): return [t for t in re.sub(r'[^a-z0-9 ]',' ',str(s).lower()).split() if t not in GEN and len(t)>=min_len and not t.isdigit()]
    nn=n.issuer_name.fillna('').str.lower().str.replace(r'[^a-z0-9 ]',' ',regex=True).drop_duplicates()
    idx={}
    for nm in nn:
        for t in set(nm.split()): idx.setdefault(t,set()).add(nm)
    rows=[]; x=C[(C.sw==1)&(C.cls=='private')].groupby('bid').principal.sum().sort_values(ascending=False)
    for bid,pr in x.items():
        names=str(B.descs.get(bid,'')).split(' || ')[:3]+[str(B.trade_names.get(bid,''))]
        for nm in names:
            t=toks(nm)[:2]
            if not t: continue
            cand=set.intersection(*[idx.get(tt,set()) for tt in t])
            if cand: rows.append((bid,round(pr/1e6),nm[:60],' '.join(t),sorted(cand)[:3])); break
    return pd.DataFrame(rows,columns=['bid','prin_m','bdc_name','tokens','nport_L2_names'])

# ---------------------------------------------------------------- C6 Step 2 sector rank, C7 quotes
def step2_extra(p,T,log):
    q=T['q2']; s9=q[q.ddate==BASE]
    gq=s9.groupby(s9.pkey.map(p.drop_duplicates('pkey').set_index('pkey').grp)).size()
    gq=gq[~gq.index.isin(['SOFT','ITSV','DATA','HARD','UNKN','FUND'])]
    nq=int((gq>=30).sum()); put('NStepTwoGroups',nq,'int','non-software groups with at least 30 positions in the registered Step 2 sample'); T['s2groups']=gq
    put('StepTwoRankRun','run' if nq>=9 else 'not run','str','C6 rule: at least 10 groups including software')
    log(f'[C6] groups with 30+ matched positions (besides software): {nq}\n'+gq.sort_values(ascending=False).to_string())
    if nq>=9:
        from d2_step2 import es2
        from d2_rank import NAMES
        rows=[]
        for g in ['SOFT']+list(gq[gq>=30].index):
            z=q if g=='SOFT' else q[q.grp!='SOFT']
            z=z.copy(); z['sw']=(z.grp==g).astype(float); th,_=es2(z,'gap'); rows.append((g,*th['202606']))
        R=pd.DataFrame(rows,columns=['grp','th','se']); R['t']=R.th/R.se; T['s2rank']=R
        for col,Y in [('th','Th'),('t','T')]:
            t=R.sort_values(col,ascending=False).reset_index(drop=True); put(f'StepTwoRank{Y}',int(t.index[t.grp=='SOFT'][0])+1,'int',f'software rank on June {col}, largest (lag direction) first')
            a=R.assign(a=R[col].abs()).sort_values('a',ascending=False).reset_index(drop=True); put(f'StepTwoRankAbs{Y}',int(a.index[a.grp=='SOFT'][0])+1,'int',f'software rank on |{col}|')
        put('StepTwoNRank',len(R),'int','groups in the Step 2 rank incl. software'); put('StepTwoMaxTh',float(R[R.grp!='SOFT'].th.max()),'f2','largest theta_Jun among other groups',sign=True)
        put('StepTwoMinTh',float(R[R.grp!='SOFT'].th.min()),'f2','smallest',sign=True)
        log('[C6] Step 2 rank:\n'+R.sort_values('th').round(2).to_string())
    # C7 stale quotes: share of matched loans whose N-PORT price is unchanged from the previous quarter
    qq=q.sort_values(['pkey','ddate']).copy(); qq['dpx']=qq.groupby('pkey').px.diff()
    for d_,Q in [(DEC,'Dec'),(MAR,'Mar'),(JUN,'Jun')]:
        z=qq[(qq.ddate==d_)&qq.dpx.notna()]
        for sw_,SW in [(1,'Sw'),(0,'Ot')]:
            put(f'Stale{SW}{Q}',(z[z.sw==sw_].dpx.abs()<0.005).mean(),'pct0',f'Step 2 sample, share of priced positions whose N-PORT price is unchanged from the previous quarter, {Q}')
    b,s_=MAC['ThetaJunLow'][1],MAC['ThetaJunLowSE'][1]; put('ThetaJunLowHi',b+1.96*s_,'f1','upper end of the 95% interval, lost quotes at a low price',sign=True); put('ThetaJunLowLo',b-1.96*s_,'f1','lower end',sign=True)
    put('LostSw',1-MAC['KeepSw'][1],'pct0','software positions priced Sep 2025, held June, without a price in June'); put('LostOt',1-MAC['KeepOt'][1],'pct0','other')

# ---------------------------------------------------------------- C13 leakage difference test
def leak_diff(p,T,log):
    keys=cohort_keys(p); c=frozen(p,keys); c['sw']=(c.grp=='SOFT').astype(float)
    fl=c.fsec.notna()&~c.fsec.isin(['it_services','tech_other'])
    dz=c[c.pkey.isin(set(c.loc[fl&(c.ddate==BASE),'pkey']))].copy()
    dz['myfn']=((dz.grp=='SOFT')&(dz.fsec!='software')).astype(float); dz['mnfy']=((dz.grp!='SOFT')&(dz.fsec=='software')).astype(float)
    dz['both']=((dz.grp=='SOFT')&(dz.fsec=='software')).astype(float)
    qs=sorted(dz.ddate.unique()); ev=[q for q in qs if q!=BASE]; X=[]; SW={k:[] for k in ['both','myfn','mnfy']}
    for q in ev:
        k=q.strftime('%Y%m'); dq=(dz.ddate==q).astype(float)
        for cc in SW: dz[f'{cc}_{k}']=dz[cc]*dq; SW[cc].append(f'{cc}_{k}')
        dz[f'sp{k}']=dz.spread_pre.fillna(0)*100*dq; dz[f'spm{k}']=dz.spread_pre.isna()*dq; dz[f'pk{k}']=dz.pik_pre.fillna(0)*dq; X+=[f'sp{k}',f'spm{k}',f'pk{k}']
    allsw=[x for cc in SW for x in SW[cc]]
    for y,Y,dec in [('mark','Mark',2),('b99','Below',1)]:
        dd,b,sc,Ainv=fit(dz,y,allsw+X,['pkey','fq','vq']); V_=vcov(dd,sc,Ainv,'clu')
        i=allsw.index('myfn_202606'); j=allsw.index('mnfy_202606')
        diff=b.iloc[i]-b.iloc[j]; sed=np.sqrt(V_[i,i]+V_[j,j]-2*V_[i,j]); z=diff/sed
        putse(f'LeakDiff{Y}',diff,sed,'June 2026: model-yes/fund-no minus fund-yes/model-no, one regression (both groups and both-agree against neither), borrower-clustered with covariance',d=dec)
        put(f'LeakDiff{Y}P',2*(1-norm.cdf(abs(z))),'f2','two-sided p')
        log(f'[C13] {y}: myfn {b.iloc[i]:+.2f}, mnfy {b.iloc[j]:+.2f}, diff {diff:+.2f} ({sed:.2f}), p {2*(1-norm.cdf(abs(z))):.2f}')

# ---------------------------------------------------------------- C15 specialisation (30 Oct), ported from v2_specialisation.py
def specialisation(p,T,log):
    keys=cohort_keys(p); b=p[(p.ddate==BASE)&p.pkey.isin(keys)].copy(); b['swb']=b.grp.eq('SOFT')
    smv=b.groupby('family').apply(lambda x: x.principal[x.swb].sum()/x.principal.sum()).rename('s_m')
    put('SpecMgrs',len(smv),'int','managers in the cohort'); put('SpecMedShare',smv.median(),'pct0','median manager software share of cohort principal'); put('SpecPNinety',smv.quantile(.9),'pct0','90th percentile')
    d=frozen(p,keys).merge(smv,left_on='family',right_index=True); d['sw']=(d.grp=='SOFT').astype(float)
    d['post']=(d.ddate>=MAR).astype(float); zz=(d.s_m-smv.mean())/smv.std()
    d['sp_post']=d.spread_pre.fillna(0)*100*d.post; d['spm_post']=d.spread_pre.isna()*d.post; d['pik_post']=d.pik_pre.fillna(0)*d.post
    d['sw_post']=d.sw*d.post; d['sw_post_s']=d.sw_post*zz
    for y,Y,dec in [('mark','Mark',2),('b99','Below',1)]:
        dd,bb,sc,Ainv=fit(d,y,['sw_post','sw_post_s','sp_post','spm_post','pik_post'],['pkey','fq']); e=se(dd,sc,Ainv,('clu','family'))
        putse(f'SpecCohort{Y}',float(bb['sw_post_s']),float(e[1]),'frozen cohort, software x post x manager software share (1 SD), position + fund x quarter FE, borrower and manager clusters',d=dec)
    q=p[p.ok&(p.ddate>='2024-12-31')&p.grp.notna()&~p.grp.isin(['UNKN','FUND'])].copy()
    q=q.groupby(['loan','ddate','family']).apply(lambda x: pd.Series({'mark':np.average(x.mark,weights=x.principal),'grp':x.grp.iloc[0]})).reset_index()
    q=q.merge(smv,left_on='family',right_index=True); q['n']=q.groupby(['loan','ddate']).family.transform('size'); q=q[q.n>=2].copy()
    q['sw']=q.grp.eq('SOFT').astype(float); q['post']=(q.ddate>=MAR).astype(float); q['b99']=100*(q.mark<99)
    q['ld']=q.loan.astype(str)+'_'+q.ddate.dt.strftime('%Y%m'); q['s']=(q.s_m-smv.mean())/smv.std()
    q['s_post']=q.s*q.post; q['s_sw']=q.s*q.sw; q['s_post_sw']=q.s*q.post*q.sw
    for y,Y,dec in [('mark','Mark',2),('b99','Below',1)]:
        X=['s','s_post','s_sw','s_post_sw']; dd,bb,sc,Ainv=fit(q,y,X,['ld']); e=se(dd,sc,Ainv,('loan','family'))
        putse(f'SpecLoan{Y}',float(bb['s_post_sw']),float(e[3]),'same loan and date, manager-level marks, loan x date FE, manager software share x post x software (1 SD), loan and manager clusters',d=dec)
    put('NSpecLoans',q.loan.nunique(),'int','co-held loans Dec 2024-Jun 2026'); put('NSpecObs',len(q),'int','manager-loan-dates')
    log(f'[C15] cohort mark {MAC["SpecCohortMark"][0]} ({MAC["SpecCohortMarkSE"][0]}); same loan mark {MAC["SpecLoanMark"][0]} ({MAC["SpecLoanMarkSE"][0]})')

# ---------------------------------------------------------------- C16 sample edge cases
def edge_cases(p,T,log):
    keys=cohort_keys(p); s9=p[(p.ddate==BASE)&p.pkey.isin(keys)]
    idn=s9.ident.fillna('').str.lower()
    rev=idn.str.contains(r'revolv|\brcf\b|\babl\b',regex=True); ddl=idn.str.contains(r'delayed|\bddtl\b|\bdd term',regex=True)
    put('NEdgeRev',int(rev.sum()),'int','cohort positions on 30 Sep 2025 whose description names a revolving credit line (funded part only: principal >= $250k passes the rules)')
    put('NEdgeDdtl',int((ddl&~rev).sum()),'int','cohort positions naming a delayed-draw tranche')
    a=p[(p.ddate==BASE)]; notok=a[~a.ok]; idn2=notok.ident.fillna('').str.lower()
    unf=idn2.str.contains(r'unfunded|undrawn|commitment',regex=True)|(notok.principal<=0)|(notok.mark<=0)
    put('NEdgeUnfunded',int(unf.sum()),'int','Sep 2025 positions that fail the mark, cost or size rules and are unfunded commitments (description names unfunded/undrawn/commitment, or principal or fair value at or below zero)')
    put('NEdgeFail',len(notok),'int','Sep 2025 positions that fail the rules')
    log(f'[C16] revolvers {int(rev.sum())}, ddtl {int((ddl&~rev).sum())}, unfunded among failures {int(unf.sum())} of {len(notok)}')

# ---------------------------------------------------------------- C17 how much
def how_much(T,log):
    a=MAC['HoneMarkJun'][1]; b=MAC['ThPxJun'][1]; c=MAC['ThMarkJun'][1]
    put('HowMuchShare',a/b,'pct0','Step 1 June mark coefficient divided by the relative traded software price change (Step 2, same fund)')
    put('HowMuchTraded',c/b,'pct0','Step 2 relative mark change divided by relative price change, traded loans')
    put('HowMuchPrivate',MAC['PrivOnlyJun'][1]/MAC['TrOnlyJun'][1],'pct0','private software relative markdown as a share of the traded one')

# ---------------------------------------------------------------- C12 bounds with the full sensitivity curve (replaces d2_rank.bounds; same seed, same order)
def bounds_curve(p,T,log,draws=300):
    from d2_rank import changes, c_t
    ch=changes(p); base=BASE; qs=sorted(ch.ddate.unique())
    vers={'Model':(ch[ch.grp.notna()&~ch.grp.isin(['ITSV','DATA','HARD','UNKN','FUND'])].assign(sw=lambda x:(x.grp=='SOFT').astype(float)),'bid'),
          'Fund':(ch[ch.fsec.notna()&~ch.fsec.isin(['it_services','tech_other'])].assign(sw=lambda x:(x.fsec=='software').astype(float)),'loan')}
    rng=np.random.default_rng(20261108); Mgrid=np.round(np.arange(0,3.01,0.05),2); res={}; curve=[]
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
                bd=None; found=False
                for M in Mgrid:
                    lo=np.percentile([bnds(B.iloc[i],M)[q][0] for i in range(len(B))],5); hi=np.percentile([bnds(B.iloc[i],M)[q][1] for i in range(len(B))],95)
                    if Q=='Jun' and abs(M*4-round(M*4))<1e-9 and M<=2: curve.append((nm,Y,M,lo,hi))
                    if not found:
                        if lo<=0<=hi: found=True
                        else: bd=M
                    if found and (Q!='Jun' or M>=2): break
                bd=0.0 if bd is None else bd
                put(f'Bd{nm}{Y}{Q}',bd,'f2' if int(round(bd*20))%2 else 'f1',f'largest Mbar (step 0.05) at which the robust 90% interval excludes zero, {nm} labels, chain-linked index, {draws} block draws')
                res[(nm,Y,Q)]=bd
            put(f'ThIdx{nm}{Y}Jun',th[JUN],'f1' if Y=='Below' else 'f2','chain-linked index, June 2026',sign=True)
            D=th[th.index<=base].diff().abs().max(); put(f'ThIdx{nm}{Y}D',D,'f1' if Y=='Below' else 'f2','largest pre-period quarterly change of the index')
            log(f'[bounds] {nm} {y}: theta Mar {th[MAR]:+.2f} Jun {th[JUN]:+.2f}; breakdown Mar {res[(nm,Y,"Mar")]} Jun {res[(nm,Y,"Jun")]}')
    T['bounds']=res; T['curve']=pd.DataFrame(curve,columns=['lab','y','M','lo','hi'])

# ---------------------------------------------------------------- unsigned copies for sentences like "fell by 2.43 points" (E28)
ABS=['HoneMarkJun','HoneMarkMar','PrivOnlyJun','PrivOnlyMar','TrOnlyJun','ThPxJun','ThMarkJun','ThetaJunLowHi','GapJun','RawMarkSwJun','RawPxSwJun','CsActMarkMar','CsPredMarkMar',
     'OtTrMarkMar','ChgSwTbJun','ChgSwPrJun','ChgSwTrJun','RehQMarJun','GapSymTrim','GapExitCarry','BorrMedSwTr','BorrMedSwPr','GapBinOne','GapBinMid','GapBinOneSix','GapBinSmall','GapBinMiddle','GapBinLarge',
     'SThreeNextK','SThreeNextTwoK','RehPDecMar','RehQDecMar','GapExitLowPr','GapFn','GapJv','GapBorrW','GapTrim','LoanPxSwDecMar','EqSwDecMar','EqSwSepMar']
def abs_macros(T,log):
    for n in ABS:
        tex=MAC[n][0]; raw=MAC[n][1]
        v=tex.replace('$','').replace('+','').replace('-','')
        put(n+'Abs',v,'str',f'unsigned copy of {n} for text that says the direction in words')
