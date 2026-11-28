# Draft 6 (26 Nov 2026): copy of draft5/d5_numbers.py pointed at the 26 Nov September builds (v2/sep2026_d6: Carlyle added, deterministic
# tie-break). Only changes: SEP path, result-file dates, the expected 'counts against' row (+1.1115), and the draft 5 tables are written to
# argv[2] (a check folder) so the draft 6 tables (tab_*_d6.tex) are not overwritten. Usage: python3 d6_d5numbers.py <draft6 folder> <check folder>
# Draft 5 (24 Nov 2026): the small queries of the draft 5 prompt (section C) -> <out>/numbers-d5.tex and <out>/numbers-d5-log.txt.
# Nothing registered is recomputed or changed. Macros use d4_common.put4 with their own registry (prefix Dfive), so no draft 5 macro can
# redefine a macro of numbers.tex or the draft 4 files (d5_check_macros.py checks all six files for duplicates).
#   C.1 (P3)  parse the verdict column of the registered C1 file (build A) for all 12 label x sample rows; check that the draft 4 macros in
#             numbers-sep.tex carry the same verdicts; count the rows that count against. Stops if no row counts against.
#   C.2 (P1,P2) education on the rank sample of Table A4 (d4_common.changes, the same positions as the plain rank): positions and borrowers
#             in the three post quarters, the summed quarterly mark change of the borrowers that drive the group, and the plain change
#             without Pluralsight and Scaled Agile. June data only (d2_common.load()). The labelling model's ai_exposure score is NOT used.
#   C.3 (P5)  the June markdown on the estimator of C1 (change on a software dummy with fund effects, positions held at both dates, borrower
#             clusters), September 2025 to June 2026, on all funds and on the confirmatory funds. June data only.
#   C.4 (P8)  counts for the September samples (Table A2 block), read from build A without estimating anything; checks the counts against
#             the registered result files; finds the confirmatory filer that does not enter C1 and why.
#   C.5 (P9)  names of the confirmatory funds and of the June Step 3 funds without a September report (PAP v5 line 24: "left out and listed").
# Usage: python3 d5_numbers.py <draft5 folder>      (the folder must already hold numbers-sep.tex from d4_sep_numbers.py)
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..','..')))  # repo root
import sys, os, re, time, hashlib
sys.path.insert(0,(_ROOT+'/bdc/scripts/draft4'))
from d4_common import *
import statsmodels.formula.api as smf
OUT=sys.argv[1]; LOG=[]; t0=time.time()
def log(s): LOG.append(str(s)); print(s,flush=True)
B=(_ROOT+'/bdc'); SEP=f'{B}/v2/sep2026_d6'; V=CM.V; F=CM.F
EF=(_ROOT+'/raw-sec/edgar-pull/early-filers-2026-11-08/early_filers_2026-11-08.csv')
BASE,JUN,SEPD=CM.BASE,CM.JUN,pd.Timestamp('2026-09-30')
EXCL=['UNKN','FUND','ITSV','DATA','HARD']

# ---------------------------------------------------------------- C.1 verdicts of the 12 registered C1 rows
c1txt=open(f'{SEP}/results_C1_C1b_buildA_2026-11-26.txt').read()
log(f'[C1] result file sha256 {sha(f"{SEP}/results_C1_C1b_buildA_2026-11-26.txt")[:16]}')
pat=re.compile(r'\[(.+?)\] (.+?): n (\d+), software (\d+), funds (\d+) \| C1 ([-+0-9.]+) \(SE [0-9.]+; 90% CI [^)]*\) (SUPPORTS|INCONCLUSIVE|COUNTS AGAINST) \| C1b ([-+0-9.]+) pp \(SE [0-9.]+; 90% CI [^)]*\) (SUPPORTS|INCONCLUSIVE|COUNTS AGAINST)')
LAB={'model labels':'Model','fund labels':'Fund','both labels agree':'Agree'}
SMP={'confirmatory (not list A, no Sep FYE)':'Conf','all funds':'All','confirmatory + Blackstone Private Credit Fund':'Bx','confirmatory without list B':'NoB'}
VERD={'SUPPORTS':'supports','INCONCLUSIVE':'inconclusive','COUNTS AGAINST':'counts against'}
sep_tex=open(f'{OUT}/numbers-sep.tex').read()
rows=pat.findall(c1txt); assert len(rows)==12, len(rows)
against=[]
for lab,smp,n,nsw,nf,c1,v1,c1b,v1b in rows:
    tag=f'Sep{LAB[lab]}{SMP[smp]}'
    for suf,v in [('COneV',v1),('COnebV',v1b)]:
        assert f'\\newcommand{{\\{tag}{suf}}}{{{VERD[v]}}}' in sep_tex, f'numbers-sep.tex verdict for {tag}{suf} differs from the result file'
    log(f'[C1] {lab} / {smp}: C1 {c1} {v1} | C1b {c1b} {v1b}')
    if v1=='COUNTS AGAINST': against.append((lab,smp,c1))
if not against: raise SystemExit('No listed C1 row counts against: the draft 5 prompt (C.1) says stop and tell the author.')
put4('DfiveCOneNRows',len(rows),'int','label x sample rows of C1 in the registered build A file')
put4('DfiveCOneNAgainst',len(against),'int','rows whose C1 counts against by the registered column')
put4('DfiveCOneNSupports',sum(r[6]=='SUPPORTS' for r in rows),'int','rows whose C1 supports')
put4('DfiveCOneNInconcl',sum(r[6]=='INCONCLUSIVE' for r in rows),'int','rows whose C1 is inconclusive')
assert against==[('both labels agree','all funds','+1.1115')], against
log('[C1] rows that count against: '+'; '.join(f'{a} / {b} ({c})' for a,b,c in against))

# ---------------------------------------------------------------- C.2 education on the rank sample (June data only)
p=CM.load(); assert p.ddate.max()==JUN
Lb=pd.read_parquet(f'{F}/labels.parquet').drop_duplicates('bid').set_index('bid')
p['name']=p.bid.map(Lb.operating_name)
ch=changes(p); ch=ch[ch.grp.notna()&~ch.grp.isin(EXCL)]
qs=sorted(ch.ddate.unique()); post=[q for q in qs if q>BASE]; assert len(post)==3
def plain(x,g):
    x=x if g=='SOFT' else x[x.grp!='SOFT']
    return sum(c_t(x[x.ddate==q],'dmark',(x[x.ddate==q].grp==g).astype(float)) for q in post)
pe=plain(ch,'EDUC'); ps=plain(ch,'SOFT')
pap_tex=open(f'{OUT}/numbers-pap.tex').read()
assert f'\\newcommand{{\\PapPlainTopMark}}{{${pe:+.2f}$}}' in pap_tex and f'\\newcommand{{\\PapPlainSwMark}}{{${ps:+.2f}$}}' in pap_tex, 'plain change does not reproduce Table A4'
e=ch[(ch.grp=='EDUC')&ch.ddate.isin(post)]
HMH=['Houghton Mifflin Harcourt','HMH Education']   # two operating names of the same borrower in the label file
e=e.assign(nm=np.where(e.name.isin(HMH),'Houghton Mifflin Harcourt',e.name))
S2='rank sample of Table A4 (labelled positions held at both ends of a quarter), Dec 2025 to Jun 2026'
put4('DfiveEducNPos',e.pkey.nunique(),'int',S2+', education positions'); put4('DfiveEducNBorr',e.bid.nunique(),'int',S2+', education borrowers')
pp=e.groupby(['nm','pkey']).dmark.sum().groupby('nm').agg(['mean','size'])
for nm,K in [('Pluralsight','Plural'),('Scaled Agile','Scaled'),('Houghton Mifflin Harcourt','Hmh'),('Discovery Education','Disc'),('Berlitz','Berlitz')]:
    put4(f'DfiveEduc{K}Chg',float(pp.loc[nm,'mean']),'f1',S2+f', {nm}: mean over its positions of the summed quarterly mark change',sign=True)
    put4(f'DfiveEduc{K}N',int(pp.loc[nm,'size']),'int',S2+f', {nm}: positions')
two=['Pluralsight','Scaled Agile']; chx=ch[~ch.name.isin(two)]
pe2=plain(chx,'EDUC'); put4('DfiveEducPlainNoTwo',pe2,'f2','education plain change (Table A4 statistic) without Pluralsight and Scaled Agile (post hoc)',sign=True)
others=pd.Series({g:plain(chx,g) for g in chx.grp.unique()}); rk=int((others<others['SOFT']).sum())+1
put4('DfiveSwPlainRankNoTwo',rk,'int','software rank on the plain change of the mark once Pluralsight and Scaled Agile are dropped (post hoc)')
prin=p[(p.ddate==BASE)&p.ok&p.name.isin(two)].principal.sum()/1e6
put4('DfiveEducTwoPrin',prin,'f0','Sep 2025 principal, $m, of the cohort positions in Pluralsight and Scaled Agile')
log(f'[educ] plain change education {pe:+.3f}, software {ps:+.3f}; positions {e.pkey.nunique()}, borrowers {e.bid.nunique()}')
log('[educ] borrowers by mean summed change: '+'; '.join(f'{k} {r["mean"]:+.1f} ({int(r["size"])})' for k,r in pp.sort_values('mean').head(10).iterrows()))
log(f'[educ] without Pluralsight and Scaled Agile: education plain {pe2:+.3f}; software plain rank {rk} of {len(others)}; top: '+'; '.join(f'{g} {v:+.2f}' for g,v in others.sort_values().head(4).items()))

# Ferman-Pinto (d4_pap_numbers.py) for the mark: how much of the variance fit comes from education's own value (examiner P18)
base=rank_table(changes(p)); x=base[base.y=='dmark'].set_index('grp'); a=x['post']; M=x['M']; ctrl=[g for g in x.index if g!='SOFT']
def fitAB(gs):
    X=np.column_stack([np.ones(len(gs)),1/M[gs].values]); return np.linalg.lstsq(X,(a[gs]**2).values,rcond=None)[0]
cA=fitAB(ctrl); cB=fitAB([g for g in ctrl if g!='EDUC'])
v=lambda c,m: max(c[0]+c[1]/m,1e-12)
assert f'\\newcommand{{\\PapFpBMark}}{{{cA[1]:,.2f}}}'.replace(',','{,}') in pap_tex, 'Ferman-Pinto B does not reproduce numbers-pap.tex'
put4('DfiveFpEducResc',a['EDUC']*np.sqrt(v(cA,M['SOFT'])/v(cA,M['EDUC'])),'f2','education plain mark change rescaled to software size (Ferman-Pinto, d4 fit)',sign=True)
put4('DfiveFpBNoEduc',cB[1],'f1','B in var=A+B/M fitted without education (mark)'); put4('DfiveFpEducM',M['EDUC'],'f0','education mean positions per quarter (rank sample)')
put4('DfiveFpEducSqShare',(a['EDUC']**2)/(a[ctrl]**2).sum(),'pct0',"education's share of the sum of squared placebo statistics (mark)")
log(f"[FP] mark: B with education {cA[1]:.1f}, without {cB[1]:.1f}; education rescaled {MAC4['DfiveFpEducResc'][1]:+.2f}; education share of squared statistics {MAC4['DfiveFpEducSqShare'][1]:.2f}; M educ {M['EDUC']:.0f}, soft {M['SOFT']:.0f}")

# ---------------------------------------------------------------- C.3 the June markdown on the C1 estimator (June data only)
ef=pd.read_csv(EF,dtype=str); A={int(x) for x in ef[ef.reason.str.startswith('A')].cik}|{1803498}
posJ=pd.read_parquet(f'{V}/positions.parquet',columns=['cik','ddate','form','bdc'])
fyeJ=posJ[posJ.form.str.startswith('10-K',na=False)].groupby('cik').ddate.agg(lambda s:s.dt.month.mode().iloc[0]); sepfyeJ={int(c) for c in fyeJ[fyeJ==9].index}
def c1_change(pp_,t0_,t1_,mask):
    ever=set(pp_.loc[pp_.ok,'pkey']); valid=np.where(pp_.ddate<=BASE,pp_.ok,pp_.mark.between(0,110)&(pp_.principal>=250e3)&pp_.pkey.isin(ever))
    x=pp_[valid&mask&pp_.ddate.isin([t0_,t1_])]
    w=x.pivot_table(index='pkey',columns='ddate',values='mark').dropna()
    r=pd.DataFrame({'dmark':w[t1_]-w[t0_]}).join(x[x.ddate==t0_].set_index('pkey')[['cik','clu','grp','bid','family']])
    r['sw']=(r.grp=='SOFT').astype(float); return r
def c_soft(r):
    r=r.copy()
    for c in ['dmark','sw']: r[c+'_d']=r[c]-r.groupby('cik')[c].transform('mean')
    m=smf.ols('dmark_d~sw_d-1',r).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(r.clu)[0]}); return m.params.iloc[0],m.bse.iloc[0]
labJ=p.grp.notna()&~p.grp.isin(EXCL)
for nm,mask,K in [('all funds',pd.Series(True,index=p.index),'All'),('confirmatory funds',~p.cik.astype(int).isin(A|sepfyeJ),'Conf')]:
    r=c1_change(p,BASE,JUN,labJ&mask); b,s=c_soft(r)
    putse4(f'DfiveJunCOne{K}',b,s,f'C1 estimator, Sep 2025 -> Jun 2026, {nm}, model labels, positions held at both dates, June data')
    put4(f'DfiveJunCOne{K}NFunds',r.cik.nunique(),'int',f'funds, {nm}')
    put4(f'DfiveJunCOne{K}Abs',abs(b),'f2',f'absolute value of DfiveJunCOne{K}')
    log(f'[C3] June markdown on the C1 estimator, {nm}: {b:+.3f} ({s:.3f}); n {len(r)}, software {int(r.sw.sum())}, funds {r.cik.nunique()}')
c1all=float(re.search(r'\[model labels\] all funds: .*?C1 ([-+0-9.]+)',c1txt).group(1)); c1conf=float(re.search(r'\[model labels\] confirmatory \(not list A, no Sep FYE\): .*?C1 ([-+0-9.]+)',c1txt).group(1))
put4('DfiveCOneShareAll',c1all/abs(MAC4['DfiveJunCOneAll'][1]),'pct0','all-fund C1 as a share of the all-fund June markdown on the same estimator')
put4('DfiveCOneShareConf',c1conf/abs(MAC4['DfiveJunCOneConf'][1]),'pct0','confirmatory C1 as a share of the confirmatory funds\' June markdown on the same estimator')
log(f'[C3] shares: all {c1all:+.4f} / {MAC4["DfiveJunCOneAll"][1]:+.4f}; confirmatory {c1conf:+.4f} / {MAC4["DfiveJunCOneConf"][1]:+.4f}')

# ---------------------------------------------------------------- C.4 September samples (build A, counts only) and C.5 names
pA=pd.read_parquet(f'{SEP}/buildA/panel.parquet'); assert pA.ddate.max()==SEPD
L=pd.read_parquet(f'{F}/labels.parquet')[['bid','group']]
pb=pd.read_parquet(f'{SEP}/buildA_aux/position_borrower.parquet').merge(L,on='bid').merge(pA[['pos_id','pkey']],on='pos_id').drop_duplicates('pkey').set_index('pkey')
pA['grp']=pA.pkey.map(pb.group); pA['bid']=pA.pkey.map(pb.bid); pA['clu']=np.where(pA.bid.notna(),'b'+pA.bid.astype(str),'l'+pA.loan.astype(str))
posA=pd.read_parquet(f'{SEP}/buildA/positions.parquet',columns=['cik','bdc','ddate','form'])
fyeA=posA[posA.form.str.startswith('10-K',na=False)].groupby('cik').ddate.agg(lambda s:s.dt.month.mode().iloc[0]); sepfyeA={int(c) for c in fyeA[fyeA==9].index}
fyeA6=posA[posA.form.str.startswith('10-K',na=False)&(posA.ddate<=JUN)].groupby('cik').ddate.agg(lambda s:s.dt.month.mode().iloc[0]); sepfyeA6={int(c) for c in fyeA6[fyeA6==9].index}
names=posA.drop_duplicates('cik').set_index('cik').bdc.str.strip()   # first name in the file, as d4_sep_numbers.py
KEEP={'LLC','LP','BDC','II','III','IV','V','VI','SBIC','LLC.'}
def nice(s):
    out=[]
    for w in str(s).split():
        core=w.strip('.,'); out.append(w if core.upper() in KEEP else w[:1].upper()+w[1:].lower())
    return ' '.join(out).replace('Pennantpark','PennantPark')
cikA=pA.cik.astype(int); labA=pA.grp.notna()&~pA.grp.isin(EXCL)
conf_mask=~cikA.isin(A|sepfyeA); filed={int(c) for c in pA.loc[pA.ddate==SEPD,'cik'].unique()}
r1=c1_change(pA,JUN,SEPD,labA&conf_mask); ra=c1_change(pA,JUN,SEPD,labA)
m1=re.search(r'\[model labels\] confirmatory \(not list A, no Sep FYE\): n (\d+), software (\d+), funds (\d+)',c1txt).groups()
ma=re.search(r'\[model labels\] all funds: n (\d+), software (\d+), funds (\d+)',c1txt).groups()
assert (len(r1),int(r1.sw.sum()),r1.cik.nunique())==tuple(map(int,m1)), ('C1 confirmatory counts differ',len(r1),r1.sw.sum(),r1.cik.nunique(),m1)
assert (len(ra),int(ra.sw.sum()),ra.cik.nunique())==tuple(map(int,ma)), ('C1 all-fund counts differ',len(ra),m1)
def counts(r,K,S):
    put4(f'DfiveRc{K}Pos',len(r),'int',S+', positions'); put4(f'DfiveRc{K}Sw',int(r.sw.sum()),'int',S+', software positions')
    put4(f'DfiveRc{K}Borr',r.bid.nunique(),'int',S+', borrowers'); put4(f'DfiveRc{K}Funds',r.cik.nunique(),'int',S+', funds'); put4(f'DfiveRc{K}Mgr',r.family.nunique(),'int',S+', managers')
counts(r1,'COneConf','C1 sample, confirmatory funds, model labels, held 30 Jun and 30 Sep 2026 (build A)')
counts(ra,'COneAll','C1 sample, all funds, model labels, held 30 Jun and 30 Sep 2026 (build A)')
# the confirmatory cohort funds that filed but do not enter C1
c9=pA[(pA.ddate==BASE)&pA.ok&pA.grp.notna()&~pA.grp.isin(['UNKN','FUND'])]
cf={int(c) for c in c9.cik.unique()}-A-sepfyeA6; cff=cf&filed; out_=sorted(cff-set(r1.cik.astype(int)))
put4('DfiveConfFiledNotCOne',len(out_),'int','confirmatory funds with a Sep 2026 report that do not enter C1')
for c in out_:
    xj=pA[(cikA==c)&(pA.ddate==JUN)]; xs=pA[(cikA==c)&(pA.ddate==SEPD)]
    lj=xj[labA.reindex(xj.index)]; ls=xs[labA.reindex(xs.index)]
    both=set(lj.pkey)&set(ls.pkey)
    log(f'[C4] fund {c} {names.get(c,c)} filed but not in C1: June rows {len(xj)} (labelled excl. ITSV/DATA/HARD {len(lj)}), Sep rows {len(xs)} (labelled {len(ls)}), labelled keys at both dates {len(both)}')
    put4('DfiveConfNotCOneName',nice(names.get(c,c)),'str','the confirmatory filer that does not enter C1')
    put4('DfiveConfNotCOneJunLab',len(lj),'int','its labelled positions on 30 Jun 2026'); put4('DfiveConfNotCOneSepRows',len(xs),'int','its rows on 30 Sep 2026')
    jout=int((~lj.mark.between(0,110)).sum()); put4('DfiveConfNotCOneJunOut',jout,'int','its labelled 30 Jun 2026 positions with a mark outside 0 to 110')
    log(f'[C4] fund {c}: labelled June positions with a mark outside 0-110: {jout} of {len(lj)}; median June mark {lj.mark.median():.0f}, median June principal {lj.principal.median():.0f} (Sep: mark {ls.mark.median():.1f}, principal {ls.principal.median():.0f})')
    assert jout==len(lj), 'the C4 reason (June principal in thousands) does not explain the drop'
# Step 3 samples (same construction as sep2026/v2_step3_sep2026_2026-11-21.py, counts only)
CLS=f'{V}/step3_classes_2026-11-17.parquet'; assert sha(CLS)=='b68346d65f2ee22bd0968835287683e3502829ef55244d97c06ec5ffdb552754'
CL=pd.read_parquet(CLS); _m=pd.read_parquet(f'{SEP}/buildA_aux/posmap.parquet').set_index('pos_id_o').pos_id_n
CL=CL[CL.pos_id.isin(_m.index)].copy(); CL['pos_id']=CL.pos_id.map(_m)
b9=pA[pA.ddate==BASE].set_index('pos_id'); CL['pk']=b9.reindex(CL.pos_id).pkey.values
fz=pA[pA.pkey.isin(set(CL.pk))]; fz=fz[(fz.ddate<=BASE)&fz.ok|(fz.ddate>BASE)&fz.mark.between(0,110)&(fz.principal>=250e3)]
w=fz.pivot_table(index='pkey',columns='ddate',values='mark'); s=CL.set_index('pk')
s['dJun']=(w[JUN]-w[BASE]).reindex(s.index); s['dSep']=(w[SEPD]-w[JUN]).reindex(s.index); s['cik']=s.cik.astype(int)
s['family']=s.index.map(pA.drop_duplicates('pkey').set_index('pkey').family)
june_funds=set(s.dropna(subset=['dJun']).loc[lambda x:x.cls.isin(['traded','private']),'cik']); reg=june_funds&filed
s3txt=open(f'{SEP}/results_step3_buildA_2026-11-26.txt').read()
for cls,K in [('traded','SThreeQ'),('private','SThreeP')]:
    z=s[s.cik.isin(reg)&(s.cls==cls)].dropna(subset=['dSep'])
    i=s3txt.index('[model labels] change 30 Jun 2026'); blk=s3txt[s3txt.index('registered sample',i):][:600]
    ref=re.search(rf'{"Q traded" if cls=="traded" else "P private"}\s+[-+0-9.]+ \(SE [0-9.]+; 90% CI [^)]*\); n (\d+), software (\d+) \((\d+) borrowers\), funds (\d+)',blk).groups()
    assert (len(z),int(z.sw.sum()),z.cik.nunique())==(int(ref[0]),int(ref[1]),int(ref[3])), (cls,len(z),z.sw.sum(),z.cik.nunique(),ref)
    counts(z.assign(sw=z.sw.astype(float)),K,f'Step 3 September sample, {cls} positions held 30 Jun and 30 Sep 2026, June-estimate funds filed by 20 Nov (build A)')
miss3=sorted(june_funds-filed); assert len(miss3)==int(re.search(r'without a Sep 2026 report in the pull: (\d+)',s3txt).group(1))
put4('DfiveSThreeMissNames','; '.join(nice(names.get(c,c))+(' (September year-end)' if c in sepfyeA6 else '') for c in miss3),'str','June Step 3 funds without a Sep 2026 report, names as filed')
missc=sorted(cf-filed); put4('DfiveConfMissNames','; '.join(nice(names.get(c,c)) for c in missc),'str','confirmatory funds without a Sep 2026 report, names as filed')
log('[C5] confirmatory without a report: '+'; '.join(f'{c} {names.get(c,c)}' for c in missc))
log('[C5] June Step 3 funds without a report: '+'; '.join(f'{c} {names.get(c,c)}'+(' (Sep FYE)' if c in sepfyeA6 else '') for c in miss3))
# ---------------------------------------------------------------- draft 5 tables (prompt section E; rows copied unchanged from the draft 4 files)
T=f'{OUT}/tables'
WT=sys.argv[2]; os.makedirs(WT,exist_ok=True)
def W(name,body): open(f'{WT}/{name}.tex','w').write(body); log(f'[tab] {name}.tex sha256 {hashlib.sha256(body.encode()).hexdigest()[:16]}')
# E.1 Table A10 panel A: keep the rows that move the estimate; the others become one sentence with their range (macros)
src=open(f'{T}/tab_robust_more.tex').read().split('\n')
iA=next(i for i,l in enumerate(src) if 'A. Step 1: other choices' in l); iB=next(i for i,l in enumerate(src) if 'B. Step 2:' in l)
KEEPA=('Weighted by principal','Principal at least','Principal did not grow','Maturity reported','\\quad same, with time to maturity','Maturity not reported')
rowsA=[l for l in src[iA+1:iB] if '&' in l]; keep=[l for l in rowsA if l.strip().startswith(KEEPA)]; drop=[l for l in rowsA if l not in keep]
assert len(keep)==6 and len(drop)==6, (len(keep),len(drop))
num=lambda c: float(re.search(r'\$?([-+]?[0-9.]+)\$?',c.replace('$','')).group(1))
dm=[num(l.split('&')[1]) for l in drop]; db=[num(l.split('&')[2]) for l in drop]
put4('DfiveRmNDrop',len(drop),'int','Table A10 panel A rows replaced by a sentence')
put4('DfiveRmDropMarkLo',min(dm),'f2','lowest June mark among those rows',sign=True); put4('DfiveRmDropMarkHi',max(dm),'f2','highest',sign=True)
put4('DfiveRmDropBelowLo',min(db),'f1','lowest June Below99 among those rows',sign=True); put4('DfiveRmDropBelowHi',max(db),'f1','highest',sign=True)
log('[tab] Table A10 panel A rows replaced by the range sentence: '+' | '.join(l.split('&')[0].strip() for l in drop))
W('tab_robust_more_d5','\n'.join(src[:iA+1]+keep+src[iB-1:]))
# E.3 Tables A7 and A8 in one table
pw=open(f'{T}/tab_power.tex').read(); p3=open(f'{T}/tab_power_s3.tex').read()
body=lambda t: t.split('\\midrule\n',1)[1].split('\\bottomrule')[0]
W('tab_power_d5','\\begin{tabular}{lcccc}\n\\toprule\n\\multicolumn{5}{l}{\\emph{A. C1, C1b and C2}} \\\\\nTest & True value & Supports (\\%) & Inconclusive (\\%) & Counts against (\\%) \\\\\n\\midrule\n'+body(pw)+
  '\\midrule\n\\multicolumn{5}{l}{\\emph{B. Step 3 rule}} \\\\\nTrue $(P,Q)$ & R1 slow marks (\\%) & R2 better loans (\\%) & R3 overshoot (\\%) & Fits none (\\%) \\\\\n\\midrule\n'+body(p3)+'\\bottomrule\n\\end{tabular}\n')
# P8: Table A2 with the September samples
rc=open(f'{T}/tab_recon.tex').read(); i=rc.rindex('\\bottomrule')
blk='\\midrule\n\\multicolumn{6}{l}{\\emph{Registered test, held on 30 June and 30 September 2026 (build A)}} \\\\\n'+''.join(
  f'{lab} & \\DfiveRc{K}Pos & \\DfiveRc{K}Sw & \\DfiveRc{K}Borr & \\DfiveRc{K}Funds & \\DfiveRc{K}Mgr \\\\\n' for lab,K in
  [('C1 and C1b: confirmatory funds','COneConf'),('C1 and C1b: all funds','COneAll'),('Step 3, $Q$: traded, June Step 3 funds filed by 20 Nov','SThreeQ'),('Step 3, $P$: private, same funds','SThreeP')])
W('tab_recon_d5',rc[:i]+blk+rc[i:])
# P3: Table A6 with a verdict column (registered columns of the C1 file, through the numbers-sep.tex macros); macros only
def vrow(lab,tag): return f'{lab} & \\{tag}NFunds & \\{tag}COne{{}} (\\{tag}COneSE) & \\{tag}COneb{{}} (\\{tag}COnebSE) & \\{tag}COneV; \\{tag}COnebV \\\\\n'
W('tab_sep_more_d5','\\begin{tabular}{L{3.7cm}cccL{2.5cm}}\n\\toprule\n\\multicolumn{5}{l}{\\emph{A. C1 and C1b, other label and sample combinations (build A)}} \\\\\n & Funds & C1 & C1b & Verdict (C1; C1b) \\\\\n\\midrule\n'+
  vrow("Funds' own labels, all funds",'SepFundAll')+vrow("Funds' own labels, Blackstone added back",'SepFundBx')+vrow("Funds' own labels, second list dropped",'SepFundNoB')+
  vrow('Both labels agree, all funds','SepAgreeAll')+vrow('Both labels agree, Blackstone added back','SepAgreeBx')+vrow('Both labels agree, second list dropped','SepAgreeNoB')+
  vrow('Build B, confirmatory funds','SepBModelConf')+
  'Confirmatory, without the 5 software borrowers that rose most & & \\SepCOneDropFive & & exploratory \\\\\n\\quad without the 10 that rose most & & \\SepCOneDropTen & & exploratory \\\\\n\\quad without the 20 that rose most & & \\SepCOneDropTwenty & & exploratory \\\\\n'+
  '\\midrule\n\\multicolumn{5}{l}{\\emph{B. Step 3, other versions}} \\\\\n & & $Q$ (traded) & $P$ (private) & Reading \\\\\n\\midrule\n'+
  ''.join(f'{lab} & & \\{t}Q{{}} (\\{t}QSE) & \\{t}P{{}} (\\{t}PSE) & \\{t}Reading \\\\\n' for lab,t in [('Second early-filer list dropped','SThreeModelNoB'),("Confirmatory funds, funds' own labels",'SThreeFundConf'),("Second list dropped, funds' own labels",'SThreeFundNoB'),('Build B, all funds filed by 20 November','SThreeBModelReg')])+
  '\\midrule\n\\multicolumn{5}{l}{\\emph{C. Persistence checks, September 2025 to September 2026 (contain quarters already seen)}} \\\\\n & Funds & Mark & Below 99 & \\\\\n\\midrule\n'+
  '$\\beta_{\\text{Sep26}}$, build A & \\PersFunds & \\PersMark{} (\\PersMarkSE) & \\PersBelow{} (\\PersBelowSE) & \\\\\n\\quad expected; surprising & & $-4.0$ to $-1.5$; above $-1.0$ & above $+20$; below $+10$ & \\\\\n\\quad reading & & \\PersMarkV & \\PersBelowV & \\\\\n'+
  'Rank among \\PersNGroups{} groups, build A (expected 1st; surprising 4th or lower) & & \\PersRankMark & \\PersRankBelow & \\\\\n$\\beta_{\\text{Sep26}}$, build B & \\PersBFunds & \\PersBMark{} (\\PersBMarkSE) & \\PersBBelow{} (\\PersBBelowSE) & \\\\\nRank among \\PersBNGroups{} groups, build B & & \\PersBRankMark & \\PersBRankBelow & \\\\\n\\bottomrule\n\\end{tabular}\n')
# E.2: Table A5 with the post hoc block cut to three lines; macros only (rows otherwise as draft 4's tab_labels.tex)
lb=open(f'{T}/tab_labels.tex').read(); i0=lb.index('\\multicolumn{3}{l}{\\emph{Added after seeing the results (post hoc)}}'); i1=lb.rindex('\\bottomrule')
ph=('\\multicolumn{3}{l}{\\emph{Added after seeing the results (post hoc)}} \\\\\n'
    'Software share among fallers / others, main labels & \\multicolumn{2}{c}{\\OmSwFellC{} / \\OmSwOtherC} \\\\\n'
    'Open model against my coding: precision; recall & \\multicolumn{2}{c}{\\OmPrec; \\OmRec} \\\\\n'
    'Step 1, June 2026, both models agree on the flag & \\OmAgreeMark{} (\\OmAgreeMarkSE) & \\OmAgreeBelow{} (\\OmAgreeBelowSE) \\\\\n')
for k in ['OmSwFellC','OmPrec','OmAgreeMark']: assert k in lb[i0:i1], k
W('tab_labels_d5',lb[:i0]+ph+lb[i1:])
# P8: Table 8 with "the June Step 3 estimate" (one label changed; rows otherwise as d4_tables.py writes them)
ts=open(f'{T}/tab_sep.tex').read(); a_='All funds of the June estimate filed by 20 November (counts)'; assert ts.count(a_)==1
W('tab_sep_d5',ts.replace(a_,'All funds of the June Step~3 estimate filed by 20 November (counts)'))
h=write_macros(f'{OUT}/numbers-d5.tex','% numbers-d5.tex: draft 5 queries (prompt section C). Written by bdc/scripts/draft6/d6_d5numbers.py (draft 5 script on the 26 Nov September builds).\n')
write_log(f'{OUT}/numbers-d5-log.txt',f'd6_d5numbers.py run {time.strftime("%Y-%m-%d %H:%M UTC",time.gmtime())}, {time.time()-t0:.0f} s, numbers-d5.tex sha256 {h[:16]}',LOG)
print('numbers-d5.tex sha256',h)
