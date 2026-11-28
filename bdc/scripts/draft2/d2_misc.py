# Data chapter numbers, Table 1, reconciliation table, labels, link checks, exits, registered rehearsal, power table,
# extensions H3/H4. Draft 2, 17 Nov 2026. Data to 30 June 2026.
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..','..')))  # repo root
import subprocess, re
from d2_common import *
from scipy.stats import norm, beta
SC=(_ROOT+'/bdc/scripts')

def sh(cmd,env=None):
    import os; e=dict(os.environ); e.update(env or {})
    r=subprocess.run(cmd,shell=True,capture_output=True,text=True,env=e,cwd=SC)
    if r.returncode: raise RuntimeError(cmd+'\n'+r.stderr[-2000:])
    return r.stdout
def grab(pat,txt,cast=float,group=1):
    m=re.search(pat,txt,re.S); assert m, f'pattern not found: {pat}'; return cast(m.group(group))

def data_numbers(p,T,log):
    pos=pd.read_parquet(f'{V}/positions.parquet',columns=['pos_id','adsh','cik','ddate','principal','cost','mark','family','srcfile'])
    put('NRawPos',len(pos),'int','SEC BDC data sets, debt rows Dec 2022-Jun 2026'); put('NRawFilings',pos.adsh.nunique(),'int','filings'); put('NRawFunds',pos.cik.nunique(),'int','BDCs')
    put('NRawMgr',pos.family.nunique(),'int','managers'); put('NRawFiles',pos.srcfile.nunique(),'int','SEC data set files with rows')
    r=pos.cost/pos.principal; s1=r.between(0.90,1.05); s2=s1&pos.mark.between(20,110); s3=s2&(pos.principal>=250e3)
    put('NFiltCost',int(s1.sum()),'int','cost 90-105% of principal'); put('NFiltMark',int(s2.sum()),'int','and mark 20-110'); put('NFiltPrin',int(s3.sum()),'int','and principal >= 250k')
    put('ShareClean',s3.mean(),'pct0','share of debt rows passing the rules')
    s9=p[(p.ddate==BASE)&p.ok]; put('NCleanSep',len(s9),'int','clean positions at Sep 2025'); put('NCleanSepMgr',s9.family.nunique(),'int','managers'); put('NCleanSepFunds',s9.cik.nunique(),'int','funds')
    put('NNoLabelRun',int(s9.grp.isna().sum()),'int','borrower outside the labelling run'); put('NUnknown',int((s9.grp=='UNKN').sum()),'int','labelled unknown'); put('NFundLab',int((s9.grp=='FUND').sum()),'int','labelled FUND')
    keys=cohort_keys(p); c=s9[s9.pkey.isin(keys)].copy(); c['sw']=c.grp.eq('SOFT')
    put('NCohort',len(c),'int','frozen cohort positions'); put('NCohortSw',int(c.sw.sum()),'int','software positions'); put('NCohortBorr',c.bid.nunique(),'int','borrowers')
    put('NCohortSwBorr',c[c.sw].bid.nunique(),'int','software borrowers'); put('NCohortFunds',c.cik.nunique(),'int','funds'); put('NCohortMgr',c.family.nunique(),'int','managers')
    put('CohortPrin',c.principal.sum()/1e9,'f0','principal $bn'); put('CohortShare',len(c)/len(s9),'pct0','share of clean Sep 2025 positions in the cohort')
    put('NoFundLabShare',c.fsec.isna().mean(),'pct0','cohort positions without a fund label (pre-Sep-2025 vote)'); put('SwSharePos',c.sw.mean(),'pct0','software share of positions'); put('SwSharePrin',c[c.sw].principal.sum()/c.principal.sum(),'pct0','software share of principal')
    # Table 1 (#26, #34, #67): 8 Nov match for the Level 2 share (Step 2 sample), Step 3 traded share separately
    M=pd.read_parquet(f'{V}/nport_matches_coupon.parquet'); M=M[M.verified&(M.lvl=='2')&(M.ddate==BASE)]; l2=set(M.pos_id)
    cls=pd.read_parquet(f'{V}/step3_classes_2026-11-17.parquet').set_index('pkey').cls
    lm=s9.groupby('loan').family.nunique(); c['nman']=c.loan.map(lm); c['cls']=c.pkey.map(cls)
    rows=[]
    for nm,x in [('Software',c[c.sw]),('Other',c[~c.sw]),('All',c)]:
        rows.append(pd.Series({'Positions':f'{len(x):,}','Borrowers':f'{x.bid.nunique():,}','Funds':f'{x.cik.nunique()}','Principal (\\$bn)':f'{x.principal.sum()/1e9:.1f}',
            'Median principal (\\$m)':f'{x.principal.median()/1e6:.1f}','Mark, mean':f'{x.mark.mean():.1f}','Marked below 99 (\\%)':f'{100*(x.mark<99).mean():.1f}',
            'Spread (bp), mean':f'{100*100*x.spread_pre.mean():.0f}','Paying interest in kind (\\%)':f'{100*x.pik_pre.fillna(0).mean():.1f}',
            'Maturity reported (\\%)':f'{100*x.ttm.notna().mean():.1f}','Held by 2+ managers (\\%)':f'{100*(x.nman>=2).mean():.1f}',
            'Level 2 price, Step 2 match (\\%)':f'{100*x.pos_id.isin(l2).mean():.1f}','Traded tranche, Step 3 (\\%)':f'{100*(x.cls=="traded").mean():.1f}',
            'Private borrower, Step 3 (\\%)':f'{100*(x.cls=="private").mean():.1f}'},name=nm))
    T1=pd.DataFrame(rows).T; T['table1']=T1
    for nm,x in [('Sw',c[c.sw]),('Ot',c[~c.sw])]:
        put(f'Tab{nm}Mark',x.mark.mean(),'f1','Sep 2025 mean mark'); put(f'Tab{nm}Spr',1e4*x.spread_pre.mean(),'f0','spread bp'); put(f'Tab{nm}Pik',x.pik_pre.fillna(0).mean(),'pct0','PIK share')
        put(f'Tab{nm}Below',(x.mark<99).mean(),'pct0','share below 99 at Sep 2025'); put(f'Tab{nm}Ltwo',x.pos_id.isin(l2).mean(),'pct1','Level 2 share, 8 Nov match')
    put('LtwoAll',c.pos_id.isin(l2).mean(),'pct1','cohort share with Level 2 price (8 Nov match, filtered universe)')
    MU=c.sector.notna()&~c.sector.isin(['it_services','tech_other']); put('MatchUnivShare',MU.mean(),'pct1','share of cohort positions in the registered match universe (fund label not IT services / other tech)')
    put('NCohortSwUnsearched',int((c.sw&~MU).sum()),'int','software cohort positions outside the registered match universe')
    # position key and holding
    nxt=set(p.loc[p.ddate==DEC,'pkey']); fd=set(p.loc[p.ddate==DEC,'cik']); cc=c[c.cik.isin(fd)]; put('KeyFound',cc.pkey.isin(nxt).mean(),'pct1','Sep 2025 positions found again in Dec 2025')
    d0=frozen(p,keys); d0['sw']=(d0.grp=='SOFT')
    for sw,nm in [(True,'Sw'),(False,'Ot')]:
        k=set(d0.loc[(d0.ddate==BASE)&(d0.sw==sw),'pkey']); put(f'Held{nm}Jun',len(set(d0.loc[d0.ddate==JUN,'pkey'])&k)/len(k),'pct0','share still in sample in June')
    # raw means (Figure 1)
    for sw,nm in [(True,'Sw'),(False,'Ot')]:
        x=d0[d0.sw==sw]
        for q,Q in [(BASE,'Sep'),(MAR,'Mar'),(JUN,'Jun')]:
            y=x[x.ddate==q]; put(f'Raw{nm}Below{Q}',(y.mark<99).mean(),'pct0','cohort held at date, share below 99'); put(f'Raw{nm}Mean{Q}',y.mark.mean(),'f1','cohort held at date, mean mark')
            put(f'Raw{nm}Par{Q}',(y.mark>=99).mean(),'pct0','share at 99 or above')
        y=x[x.ddate==JUN]; put(f'Raw{nm}EightyJun',(y.mark<80).mean(),'pct0','share below 80 in June')
    w=d0.pivot_table(index='pkey',columns=d0.ddate.dt.strftime('%Y%m'),values='mark'); sw0=d0.drop_duplicates('pkey').set_index('pkey').sw
    ch=(w['202606']-w['202509']).dropna(); s_=sw0.reindex(ch.index)
    put('WithinSwMeanJun',ch[s_].mean(),'f1','software positions held Sep25 and Jun26, mean change',sign=True); put('WithinSwMedJun',ch[s_].median(),'f1','median',sign=True)
    put('WithinOtMeanJun',ch[~s_].mean(),'f1','other, mean change',sign=True); put('WithinOtMedJun',ch[~s_].median(),'f1','other, median',sign=True)
    # Dec 2025 10-K dates
    pz=pd.read_parquet(f'{V}/positions.parquet',columns=['cik','ddate','form','filed']); dec=pz[(pz.ddate==DEC)&pz.form.str.startswith('10-K',na=False)&pz.cik.isin(c.cik.unique())].drop_duplicates('cik')
    f=pd.to_datetime(dec.filed); put('DecFiledFirst',f.min().strftime('%-d %B'),'str','first Dec 2025 10-K filing among cohort funds'); put('DecFiledLast',f.max().strftime('%-d %B'),'str','last')
    put('DecFiledAfter',(f>pd.Timestamp('2026-02-03')).mean(),'pct0','share of Dec 2025 10-Ks filed after 3 Feb 2026')
    T['cohort_c']=c

def recon(p,T,log):
    """Appendix A reconciliation: every sample in the text as one row (positions, software, borrowers, funds, managers)."""
    s9=p[(p.ddate==BASE)&p.ok]; keys=cohort_keys(p); rows=[]
    def add(nm,x,sw=None,unit='positions'):
        rows.append(dict(sample=nm,unit=unit,positions=len(x) if unit=='positions' else x,software=sw if sw is not None else (int((x.grp=='SOFT').sum()) if hasattr(x,'grp') else np.nan),
                         borrowers=(x.bid.nunique() if hasattr(x,'bid') else np.nan),funds=(x.cik.nunique() if hasattr(x,'cik') else np.nan),managers=(x.family.nunique() if hasattr(x,'family') else np.nan)))
    add('All clean positions, 30 Sep 2025',s9)
    c=s9[s9.pkey.isin(keys)]; add('Frozen cohort (model label known)',c)
    R=T['h1']['headline']['n']; rows.append(dict(sample='Step 1 estimation sample (after singletons)',unit='positions',positions=R['pos'],software=R['sw'],borrowers=R['borrowers'],funds=R['funds'],managers=R['managers']))
    rf=T['h1']['fund labels on cohort']['n']; rows.append(dict(sample='Step 1, funds\' labels on the cohort',unit='positions',positions=rf['pos'],software=rf['sw'],borrowers=rf['borrowers'],funds=rf['funds'],managers=rf['managers']))
    q=T['q2']; q9=q[q.ddate==BASE]; rows.append(dict(sample='Step 2: Level 2 price at 30 Sep 2025 (registered match)',unit='positions',positions=len(q9),software=int(q9.sw.sum()),borrowers=q9.bid.nunique(),funds=q9.cik.nunique(),managers=q9.family.nunique()))
    s3=T['s3']
    for cl in ['traded','private','L3 only','traded borrower']:
        x=s3[s3.cls==cl]; rows.append(dict(sample=f'Step 3: {cl}',unit='positions',positions=len(x),software=int(x.sw.sum()),borrowers=x.bid.nunique(),funds=x.cik.nunique(),managers=x.family.nunique()))
    for k in ['h3','h4']:
        if k in T: rows.append(T[k])
    Rc=pd.DataFrame(rows); T['recon']=Rc; log('[recon]\n'+Rc.to_string())
    return Rc

def labels(T,log):
    out=sh('python3 llm_labels_score.py')
    put('NLabelled',grab(r'labelled (\d+)',out,int),'int','borrowers labelled'); put('ShareUnknown',grab(r'UNKN ([0-9.]+) \| software share',out),'pct0','share unknown')
    put('LabRecall',grab(r'all\s+n\s+\d+ \| software recall ([0-9.]+)',out),'pct0','recall vs funds labels'); put('LabPrecision',grab(r'all\s+n\s+\d+ \| software recall [0-9.]+ false-positive rate [0-9.]+ precision ([0-9.]+)',out),'pct0','precision')
    n1=grab(r'fell 3\+ points\s+n\s+(\d+)',out,int); a1=grab(r'fell 3\+ points\s+n\s+\d+ \| LLM software, manager not: ([0-9.]+)',out)
    n0=grab(r'did not\s+n\s+(\d+)',out,int); a0=grab(r'did not\s+n\s+\d+ \| LLM software, manager not: ([0-9.]+)',out)
    pp=(a1*n1+a0*n0)/(n1+n0); z=(a1-a0)/np.sqrt(pp*(1-pp)*(1/n1+1/n0))
    put('LeakFell',a1,'pct1','model software, fund not, among borrowers whose loans fell 3+'); put('LeakNot',a0,'pct1','same among the others'); put('NLeakFell',n1,'int','borrowers'); put('NLeakNot',n0,'int','borrowers')
    put('LeakZ',z,'f1','two-proportion z'); put('LeakP',2*(1-norm.cdf(abs(z))),'f4','two-sided p')
    rr=open(f'{F}/../rerun-2026-11-08/results_rerun_agreement_2026-11-08.txt').read()
    put('RerunSame',grab(r'same software flag: ([0-9.]+)',rr),'pct1','same software flag, 500-borrower rerun'); put('RerunKappa',grab(r'kappa \(software\) ([0-9.]+)',rr),'f2','kappa')
    put('RerunSwOne',grab(r'software in run 1: (\d+)',rr,int),'int','software run 1'); put('RerunSwTwo',grab(r'in run 2: (\d+)',rr,int),'int','software run 2')
    log('[labels]\n'+out[-1500:])

def links(T,log):
    H=f'{V}/handcheck'
    for s,f,nm in [('A','A_crossfund_verdicts.csv','Cross'),('B','B_nport_verdicts.csv','Nport')]:
        d=pd.read_csv(f'{H}/{f}'); n=len(d); kb=int((d.borrower!='Y').sum()); ki=int((d.instrument!='S').sum()); k=int(((d.borrower!='Y')|(d.instrument!='S')).sum())
        lo=beta.ppf(.025,k,n-k+1) if k else 0; hi=beta.ppf(.975,k+1,n-k)
        put(f'Chk{nm}N',n,'int','links checked'); put(f'Chk{nm}Borr',kb,'int','wrong or unclear borrower'); put(f'Chk{nm}Tr',ki,'int','tranche conflict'); put(f'Chk{nm}Any',k,'int','any problem')
        put(f'Chk{nm}Lo',lo,'pct1','exact 95% CI'); put(f'Chk{nm}Hi',hi,'pct1','exact 95% CI')
        if s=='A': put('ChkCrossRev',int((d.instrument=='R').sum()),'int','revolver vs term pairs')
        hm=(d.maturity_a.notna()&d.maturity_c.notna()) if s=='A' else d.maturity.notna()   # match quality with and without maturity (#24)
        bad=(d.borrower!='Y')|(d.instrument!='S')
        put(f'Chk{nm}MatN',int(hm.sum()),'int','checked links where the BDC maturity is reported'); put(f'Chk{nm}MatAny',int((bad&hm).sum()),'int','problems among them')
        put(f'Chk{nm}NoMatN',int((~hm).sum()),'int','checked links without maturity'); put(f'Chk{nm}NoMatAny',int((bad&~hm).sum()),'int','problems among them')

def exits(T,log):
    sh('python3 v2_exit_bounds_2026-11-08.py'); o=open(f'{V}/results_exit_bounds_2026-11-08.txt').read()
    put('ExitSw',grab(r'2026-06-30 0\.0\s+([0-9.]+).*?\n\s+1\.0\s+([0-9.]+)',o,float,2),'pct0','software share left by June'); put('ExitOt',grab(r'2026-06-30 0\.0\s+([0-9.]+)',o),'pct0','other share left by June')
    put('ExitLowLast',grab(r'gone, last mark <80\s+([0-9.]+)\s+([0-9.]+)',o),'pct1','other exits with last mark <80'); put('ExitLowLastSw',grab(r'gone, last mark <80\s+([0-9.]+)\s+([0-9.]+)',o,float,2),'pct1','software exits with last mark <80')
    put('ExitHighSw',grab(r'gone, last mark 97\+\s+([0-9.]+)\s+([0-9.]+)',o,float,2),'pct0','software exits with last mark 97+'); put('ExitSimSw',grab(r'similar remains\s+([0-9.]+)\s+([0-9.]+)',o,float,2),'pct0','software exits replaced')
    for tag,pat in [('LowSw',r'software leavers low \(p10\), others at last mark: mark 2026-03 ([-+0-9.]+) \([0-9.]+\); 2026-06 ([-+0-9.]+)'),
                    ('LowOt',r'software leavers at last mark, others low \(p10\): mark 2026-03 ([-+0-9.]+) \([0-9.]+\); 2026-06 ([-+0-9.]+)'),
                    ('FiftySw',r'software leavers at 50, others at last mark: mark 2026-03 ([-+0-9.]+) \([0-9.]+\); 2026-06 ([-+0-9.]+)'),
                    ('FiftyOt',r'software leavers at last mark, others at 50: mark 2026-03 ([-+0-9.]+) \([0-9.]+\); 2026-06 ([-+0-9.]+)')]:
        put(f'Exit{tag}Mar',grab(pat,o),'f2','exit imputation bound, mark',sign=True); put(f'Exit{tag}Jun',grab(pat,o,float,2),'f2','exit imputation bound, mark',sign=True)
    for tag,pat in [('LowSw',r'software leavers low \(p10\), others at last mark: .*?b99 2026-03 ([-+0-9.]+) \([0-9.]+\); 2026-06 ([-+0-9.]+)'),
                    ('LowOt',r'software leavers at last mark, others low \(p10\): .*?b99 2026-03 ([-+0-9.]+) \([0-9.]+\); 2026-06 ([-+0-9.]+)')]:
        put(f'ExitB{tag}Mar',grab(pat,o),'f1','exit bound b99',sign=True); put(f'ExitB{tag}Jun',grab(pat,o,float,2),'f1','exit bound b99',sign=True)
    put('ExitOtherHolder',grab(r'other\s+([0-9.]+); software\s+([0-9.]+)',o,float,2),'pct0','software exits whose loan another manager still held')

def registered(T,log):
    out=sh('python3 v2_confirmatory_v2_2026-11-16.py 2026-03-31 2026-06-30')
    m=re.search(r'\[model labels\] confirmatory \(not list A, no Sep FYE\): n (\d+), software (\d+), funds (\d+) \| C1 ([-+0-9.]+) \(SE ([0-9.]+); 90% CI ([-+0-9.]+), ([-+0-9.]+)\) \w+ \| C1b ([-+0-9.]+) pp \(SE ([0-9.]+)',out)
    n,nsw,nf,c1,s1,lo,hi,c1b,s1b=m.groups(); put('NConfPos',int(n),'int','rehearsal C1 positions'); put('NConfFunds',int(nf),'int','funds in rehearsal')
    putse('RehCOne',float(c1),float(s1),'rehearsal Mar->Jun 2026, confirmatory funds'); putse('RehCOneb',float(c1b),float(s1b),'C1b rehearsal',d=1)
    m=re.search(r'\[model labels\] all funds: n (\d+), software (\d+), funds (\d+) \| C1 ([-+0-9.]+) \(SE ([0-9.]+).*?C1b ([-+0-9.]+) pp \(SE ([0-9.]+)',out)
    putse('RehCOneAll',float(m.group(4)),float(m.group(5)),'rehearsal, all funds'); putse('RehCOnebAll',float(m.group(6)),float(m.group(7)),'C1b all funds',d=1)
    m=re.search(r'confirmatory \(not list A, no Sep FYE\) \[10-point trim \(registered\)\]: software pairs (\d+) \((\d+) borrowers\).*?dlambda ([-+0-9.]+) \(SE ([0-9.]+)',out)
    putse('RehCTwo',float(m.group(3)),float(m.group(4)),'C2 rehearsal'); put('NRehCTwo',int(m.group(1)),'int','software pairs')
    m=re.search(r'all funds \[10-point trim \(registered\)\]: software pairs (\d+).*?dlambda ([-+0-9.]+) \(SE ([0-9.]+)',out); putse('RehCTwoAll',float(m.group(2)),float(m.group(3)),'C2 rehearsal all funds')
    # power table (#43): probability of each verdict under true values, given the rehearsal SE, normal approximation
    s1=float(s1); s1b=float(s1b); s2=float(re.search(r'confirmatory \(not list A, no Sep FYE\) \[10-point trim \(registered\)\].*?dlambda [-+0-9.]+ \(SE ([0-9.]+)',out).group(1))
    rows=[]
    for th in [0.0,0.5,0.8]:
        rows.append(('C1',th,norm.cdf((0.5-th)/s1),norm.cdf((1.0-th)/s1)-norm.cdf((0.5-th)/s1),1-norm.cdf((1.0-th)/s1)))
    for th in [0.0,-5.0,-8.0]:
        rows.append(('C1b',th,1-norm.cdf((-5-th)/s1b),norm.cdf((-5-th)/s1b)-norm.cdf((-10-th)/s1b),norm.cdf((-10-th)/s1b)))
    for th in [0.0,-0.1,-0.2]:
        z=1.645*s2; sup=norm.cdf((-z-th)/s2); inc=norm.cdf((0-th)/s2)-sup; ag=1-norm.cdf((0-th)/s2)
        rows.append(('C2',th,sup,inc,ag))
    PW=pd.DataFrame(rows,columns=['test','true','supports','inconclusive','against']); T['power']=PW; T['power_se']=(s1,s1b,s2)
    put('PowSEOne',s1,'f2','rehearsal SE C1'); put('PowSEOneb',s1b,'f1','rehearsal SE C1b'); put('PowSETwo',s2,'f2','rehearsal SE C2')
    put('PowCOneSupFive',float(PW[(PW.test=='C1')&(PW.true==0.5)].supports.iloc[0]),'pct0','P(C1 supports | true +0.5)')
    put('PowCOneAgEight',float(PW[(PW.test=='C1')&(PW.true==0.8)].against.iloc[0]),'pct0','P(C1 counts against | true +0.8)')
    put('PowCOneSupEight',float(PW[(PW.test=='C1')&(PW.true==0.8)].supports.iloc[0]),'pct0','P(C1 supports | true +0.8)')
    log('[power]\n'+PW.round(3).to_string())

def ext(p,T,log):
    o=sh('python3 v2_h3_registered_2026-11-13.py',{'H3_COSTGUARD':'0','H3_KEYS':'frozen','H3_W':'mean'})
    m=re.search(r'Row 2 PRIMARY.*?soft x post ([-+0-9.]+) \(dyadic SE ([0-9.]+)',o); putse('HthreeReg',float(m.group(1)),float(m.group(2)),'H3 registered primary, no guards, dyadic SE')
    m=re.search(r'Sample, tranche filter as registered: loans (\d+), software (\d+)',o); put('NHthreeLoans',int(m.group(1)),'int','H3 loans'); put('NHthreeSw',int(m.group(2)),'int','software loans')
    T['h3']=dict(sample='H3: loans held by 3+ managers (registered)',unit='loans',positions=int(m.group(1)),software=int(m.group(2)),borrowers=np.nan,funds=np.nan,
                 managers=grab(r'distinct top-two managers (\d+)',o.split('tranche filter as registered')[1],int))
    m=re.search(r'Row 1 \(quarter x Sep 2025 mark bin.*?soft x post ([-+0-9.]+) \(dyadic SE ([0-9.]+)',o); putse('HthreeRowOne',float(m.group(1)),float(m.group(2)),'H3 row 1, no guards')
    m=re.search(r'unexplained part: loan bootstrap SE ([0-9.]+), 95% interval \[([-+0-9.]+), ([-+0-9.]+)\]',o); put('HthreeUnexLo',float(m.group(2)),'f2','',sign=True); put('HthreeUnexHi',float(m.group(3)),'f2','',sign=True)
    m=re.search(r'difference software-other: actual ([-+0-9.]+), explained by level ([-+0-9.]+) .*?unexplained ([-+0-9.]+)',o); put('HthreeAct',float(m.group(1)),'f2','',sign=True); put('HthreeExpl',float(m.group(2)),'f2','',sign=True); put('HthreeUnex',float(m.group(3)),'f2','',sign=True)
    o2=sh('python3 v2_h3_registered_2026-11-13.py',{'H3_COSTGUARD':'1','H3_KEYS':'frozen','H3_W':'mean'})
    m=re.search(r'Row 2 PRIMARY.*?soft x post ([-+0-9.]+) \(dyadic SE ([0-9.]+)',o2); putse('HthreeGuard',float(m.group(1)),float(m.group(2)),'H3 with data guards (added after the estimate)')
    T['h3txt']=(o,o2)
    # H4 with manager x Post and manager x Soft FE (#111)
    h4(p,T,log)

def h4(p,T,log):
    pos=pd.read_parquet(f'{V}/positions.parquet',columns=['pos_id'])
    q=p[p.ok&p.mark.between(0,110)&(p.principal>0)].copy(); q['fv']=q.mark*q.principal/100
    s25=q[q.ddate==BASE]; M=s25.groupby('family').apply(lambda g:pd.Series({'S':g.fv[g.vtype=='listed'].sum()/g.fv.sum()})).reset_index()
    L=pd.read_parquet(f'{F}/labels.parquet')[['bid','group']]
    gl=pd.read_parquet(f'{F}/position_borrower.parquet').merge(L,on='bid').merge(q[['pos_id','loan']],on='pos_id').dropna(subset=['loan']).drop_duplicates('loan').set_index('loan').group
    q['g']=q.loan.map(gl); q['mp']=q.mark*q.principal
    m=q.groupby(['loan','ddate','family']).agg(mp=('mp','sum'),pr=('principal','sum'),g=('g','first')).reset_index(); m['mark']=m.mp/m.pr
    m=m.merge(M,on='family'); m=m[m.g.notna()&~m.g.isin(['UNKN','FUND'])]
    m['nfam']=m.groupby(['loan','ddate']).family.transform('size'); m=m[m.nfam>=2].copy()
    m['sw']=(m.g=='SOFT').astype(float); m['post']=(m.ddate>=MAR).astype(float); m['ld']=m.loan.astype(str)+'_'+m.ddate.dt.strftime('%Y%m')
    m['x']=m.S; m['x_post']=m.S*m.post; m['x_sw']=m.S*m.sw; m['x_post_sw']=m.S*m.post*m.sw
    m['mpost']=m.family+'_'+m.post.astype(int).astype(str); m['msw']=m.family+'_'+m.sw.astype(int).astype(str)
    dd,b,sc,Ainv=fit(m,'mark',['x','x_post','x_sw','x_post_sw'],['ld']); e=se(dd,sc,Ainv,('loan','family'))
    putse('HfourReg',float(b['x_post_sw']),float(e[3]),'H4 registered main, loan x date FE, two-way SE'); put('HfourLo',b['x_post_sw']-1.645*e[3],'f2','90% CI',sign=True); put('HfourHi',b['x_post_sw']+1.645*e[3],'f2','90% CI',sign=True)
    put('NHfourObs',len(dd),'int','manager-loan-dates'); put('NHfourLoans',dd.loan.nunique(),'int','loans'); put('NHfourMgr',dd.family.nunique(),'int','managers')
    dd2,b2,sc2,A2=fit(m,'mark',['x_post_sw'],['ld','mpost','msw']); e2=se(dd2,sc2,A2,('loan','family'))
    putse('HfourMgrFe',float(b2['x_post_sw']),float(e2[0]),'H4 with manager x Post and manager x Soft FE')
    put('NHfourMid',int(M.S.between(1e-9,1-1e-9).sum()),'int','managers with both fund types'); put('NHfourAll',len(M),'int','managers with a BDC at Sep 2025')
    T['h4']=dict(sample='H4: loans held by 2+ managers, manager-loan-dates',unit='loans',positions=dd.loan.nunique(),software=int(dd.drop_duplicates('loan').sw.sum()),borrowers=np.nan,funds=np.nan,managers=dd.family.nunique())
    log(f'[h4] main {b["x_post_sw"]:+.3f} ({e[3]:.3f}); with mgr x post and mgr x soft FE {b2["x_post_sw"]:+.3f} ({e2[0]:.3f})')
