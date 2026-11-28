# Draft 6 (26 Nov 2026): macros for the registered September 2026 tests and the persistence checks -> <out>/numbers-sep.tex.
# Adapted copy of draft4/d4_sep_numbers.py. Changes, and nothing else:
#   - the builds are the 26 Nov deterministic rebuilds with Carlyle Credit Solutions added (bdc/v2/sep2026_d6/, built by
#     draft6/run_build_det.sh: v2_stablekey_det.py breaks exact score ties by pos_id; successor CIK 2144472 mapped to 1702510);
#   - the reruns are compared byte for byte with the result files stored on 26 Nov in bdc/v2/sep2026_d6/;
#   - added (section 6): the 21 Nov stored C1 and the PC numbers of the deviation note as macros, and three disclosed
#     additions that are not registered: two-sided 5% trims of C1 and of Step 3 (#122), and non-accrual flags (#114, saved).
# The two registered scripts (sep2026/c1x.py, sep2026/v2_step3_sep2026_2026-11-21.py) are run unchanged.
# Usage: python3 d6_sep_numbers.py <draft6 folder> [scratch folder]
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..','..')))  # repo root
import sys, os, re, time, subprocess, tempfile, filecmp
sys.path.insert(0,(_ROOT+'/bdc/scripts/draft4'))
from d4_common import *
from d2_rank import NAMES
OUT=sys.argv[1]; SCR=sys.argv[2] if len(sys.argv)>2 else tempfile.mkdtemp(prefix='d6sep_')
B=(_ROOT+'/bdc'); SEP=f'{B}/v2/sep2026_d6'; SEP21=f'{B}/v2/sep2026'; SC=f'{B}/scripts/sep2026'
LOG=[]; t0=time.time()
def log(s): LOG.append(str(s)); print(s,flush=True)
BUILDS={'A':dict(VD=f'{SEP}/buildA',OLDPANEL=f'{SEP}/buildA_aux/oldpanel_map.parquet',PB=f'{SEP}/buildA_aux/position_borrower.parquet',POSMAP=f'{SEP}/buildA_aux/posmap.parquet'),
        'B':dict(VD=f'{SEP}/buildB',OLDPANEL=f'{B}/v2/panel.parquet',PB=f'{CM.F}/position_borrower.parquet',POSMAP=None)}
TXT={}
for b,cfg in BUILDS.items():
    vd=f'{SCR}/vd{b}'; os.makedirs(vd,exist_ok=True)
    for x in ['panel','positions','links','sid']:
        l=f'{vd}/{x}.parquet'
        if not os.path.exists(l): os.symlink(f'{cfg["VD"]}/{x}.parquet',l)
    env=dict(os.environ,VD=vd,OLDPANEL=cfg['OLDPANEL'],PB=cfg['PB'])
    r=subprocess.run(['python3',f'{SC}/c1x.py'],capture_output=True,text=True,env=env,cwd=SCR); assert r.returncode==0, r.stderr[-2000:]
    f1=f'{vd}/results_confirmatory_202606_202609_v2_2026-11-16.txt'; ref=f'{SEP}/results_C1_C1b_build{b}_2026-11-26.txt'
    assert filecmp.cmp(f1,ref,shallow=False), f'C1 rerun differs from {ref}'; TXT[('c1',b)]=open(f1).read(); TXT[('c1x',b)]=r.stdout
    env2=dict(os.environ);
    if cfg['POSMAP']: env2['POSMAP']=cfg['POSMAP']
    r=subprocess.run(['python3',f'{SC}/v2_step3_sep2026_2026-11-21.py',vd],capture_output=True,text=True,env=env2,cwd=SCR); assert r.returncode==0, r.stderr[-2000:]
    f2=f'{vd}/results_step3_sep2026_2026-11-21.txt'; ref2=f'{SEP}/results_step3_build{b}_2026-11-26.txt'
    assert filecmp.cmp(f2,ref2,shallow=False), f'Step 3 rerun differs from {ref2}'; TXT[('s3',b)]=open(f2).read()
    log(f'[rerun] build {b}: C1 and Step 3 result files regenerated and identical to the stored files ({sha(ref)[:12]}, {sha(ref2)[:12]})')
# ---------------------------------------------------------------- C1 and C1b
LAB={'model labels':'Model','fund labels':'Fund','both labels agree':'Agree'}
SMP={'confirmatory (not list A, no Sep FYE)':'Conf','all funds':'All','confirmatory + Blackstone Private Credit Fund':'Bx','confirmatory without list B':'NoB'}
VERD={'SUPPORTS':'supports','INCONCLUSIVE':'inconclusive','COUNTS AGAINST':'counts against'}
pat=re.compile(r'\[(.+?)\] (.+?): n (\d+), software (\d+), funds (\d+) \| C1 ([-+0-9.]+) \(SE ([0-9.]+); 90% CI ([-+0-9.]+), ([-+0-9.]+)\) (SUPPORTS|INCONCLUSIVE|COUNTS AGAINST) \| C1b ([-+0-9.]+) pp \(SE ([0-9.]+); 90% CI ([-+0-9.]+), ([-+0-9.]+)\) (SUPPORTS|INCONCLUSIVE|COUNTS AGAINST)')
C1={}
for b in 'AB':
    for m in pat.finditer(TXT[('c1',b)]):
        lab,smp,n,nsw,nf,c1,s1,lo,hi,v1,c1b,s1b,lob,hib,v1b=m.groups(); C1[(b,lab,smp)]=m.groups()
        tag=f'Sep{"" if b=="A" else "B"}{LAB[lab]}{SMP[smp]}'
        if b=='B' and not (lab=='model labels' and smp.startswith('confirmatory (not')): continue
        S=f'build {b}, {lab}, {smp}, change 30 Jun -> 30 Sep 2026'
        put4(f'{tag}COne',float(c1),'f2',S+', C1',sign=True); put4(f'{tag}COneSE',float(s1),'f2',S)
        put4(f'{tag}COneLo',float(lo),'f2',S+', 90% CI',sign=True); put4(f'{tag}COneHi',float(hi),'f2',S,sign=True); put4(f'{tag}COneV',VERD[v1],'str',S+', registered column')
        put4(f'{tag}COneb',float(c1b),'f1',S+', C1b pp',sign=True); put4(f'{tag}COnebSE',float(s1b),'f1',S)
        put4(f'{tag}COnebV',VERD[v1b],'str',S+', registered column'); put4(f'{tag}N',int(n),'int',S+', positions'); put4(f'{tag}NSw',int(nsw),'int',S+', software positions'); put4(f'{tag}NFunds',int(nf),'int',S+', funds')
        if b=='A' and lab=='model labels' and smp.startswith('confirmatory (not'):
            put4('SepCOneFour',float(c1),'f3',S+', C1 at three decimals',sign=True); put4('SepCOneMargin',0.5-float(c1),'f3','distance of C1 below the +0.5 threshold')
            put4('SepCOnebLo',float(lob),'f1',S+', C1b 90% CI',sign=True); put4('SepCOnebHi',float(hib),'f1',S,sign=True)
x=TXT[('c1x','A')]
for k,K in [(5,'Five'),(10,'Ten'),(20,'Twenty')]:
    put4(f'SepCOneDrop{K}',float(re.search(rf'EXTRA drop {k} software borrowers with largest rises: C1 ([-+0-9.]+)',x).group(1)),'f2',f'C1 confirmatory, model labels, build A, without the {k} software borrowers with the largest rises (exploratory)',sign=True)
m=re.search(r'EXTRA software dmark mean ([-0-9.]+) median ([-0-9.]+) share>0 ([0-9.]+) share<0 ([0-9.]+) \| other mean ([-0-9.]+) median ([-0-9.]+) share>0 ([0-9.]+) share<0 ([0-9.]+)',x)
put4('SepSwMedChg',float(m.group(2)),'f2','median Jun->Sep 2026 mark change, software, confirmatory, build A'); put4('SepOtMedChg',float(m.group(6)),'f2','median, other loans')
put4('SepSwUpShare',float(m.group(3)),'pct0','share of software positions whose mark rose'); put4('SepOtUpShare',float(m.group(7)),'pct0','other loans')
put4('SepNSwBorrConf',int(re.search(r'EXTRA software borrowers (\d+)',x).group(1)),'int','software borrowers in the confirmatory C1 sample')
# list the label/sample rows whose C1 sign is opposite to the confirmatory estimate (PAP v5 "Which estimate counts")
conf=float(C1[('A','model labels','confirmatory (not list A, no Sep FYE)')][5])
opp=[(l,s) for (b,l,s),g in C1.items() if b=='A' and np.sign(float(g[5]))!=np.sign(conf)]
log('[C1] rows with the opposite sign to the confirmatory C1: '+(', '.join(f'{l} / {s} ({C1[("A",l,s)][5]})' for l,s in opp) if opp else 'none'))
put4('SepNOppSign',len(opp),'int','listed C1 rows (build A) with a sign opposite to the confirmatory estimate')
# ---------------------------------------------------------------- C2 status
c2line=re.search(r'confirmatory \(not list A, no Sep FYE\) \[10-point trim \(registered\)\]: software pairs (\d+) \((\d+) borrowers\).*?dlambda ([-+0-9.]+) \(SE ([0-9.]+); one-sided p ([0-9.]+); 90% CI ([-+0-9.]+), ([-+0-9.]+)\) (SUPPORTS|INCONCLUSIVE|COUNTS AGAINST)',TXT[('c1','B')])
NPORT=False
if c2line and int(c2line.group(1))>=50:
    NPORT=True; npairs,nb,dl,sdl,p1,lo,hi,v=c2line.groups()
    putse4('SepCTwo',float(dl),float(sdl),'C2 dlambda, confirmatory, 10-point trim (registered)'); put4('SepCTwoP',float(p1),'f3','one-sided p'); put4('SepCTwoLo',float(lo),'f2','90% CI',sign=True); put4('SepCTwoHi',float(hi),'f2','90% CI',sign=True)
    put4('SepCTwoN',int(npairs),'int','software pairs'); put4('SepCTwoV',VERD[v],'str','registered column')
    put4('CTwoCase',{'SUPPORTS':0,'INCONCLUSIVE':1,'COUNTS AGAINST':2}[v],'int','0 supports, 1 inconclusive, 2 counts against')
    put4('HTwoVerdict','supported' if (v=='SUPPORTS' and C1[('A','model labels','confirmatory (not list A, no Sep FYE)')][9]!='COUNTS AGAINST') else 'not supported','str','PAP v5: H2 supported iff C2 supports and C1 does not count against')
if NPORT and not all(k in MAC4 for k in ('SepNportCover','SepQPx','SepQPxSE','QPxCase')):
    # The draft 4 \\ifnport text also needs the share of the June match set covered (PAP v5 section 5) and the N-PORT price change next
    # to Q (Amendment 1). Their code needs the September N-PORT file and is not written yet (decisions.md D4-9). Stop rather than set
    # \\nporttrue with undefined macros.
    raise SystemExit('N-PORT leg present: add SepNportCover, SepQPx, SepQPxSE and QPxCase before setting \\nporttrue (decisions.md D4-9)')
log('[C2] '+('N-PORT leg present: '+c2line.group(0) if NPORT else 'no September 2026 N-PORT prices in the match file (or fewer than 50 software pairs): \\nportfalse, PAP fallback applies'))
# ---------------------------------------------------------------- Step 3
s3=TXT[('s3','A')]
def blk(txt,lab,smp):
    i=txt.index(f'[{lab}] change 30 Jun 2026'); j=txt.index(smp,i); return txt[j:j+700]
SS={'registered sample (June-estimate funds filed by 20 Nov)':'Reg','76 confirmatory funds of C1 (not list A, no Sep FYE)':'Conf','confirmatory without list B':'NoB'}
rp=re.compile(r'Q traded\s+([-+0-9.]+) \(SE ([0-9.]+); 90% CI ([-+0-9.]+), ([-+0-9.]+)\); n (\d+), software (\d+) \((\d+) borrowers\), funds (\d+)\s+P private\s+([-+0-9.]+) \(SE ([0-9.]+); 90% CI ([-+0-9.]+), ([-+0-9.]+)\); n (\d+), software (\d+) \((\d+) borrowers\), funds (\d+)\s+gap traded minus private, fund x sector FE \(context\) ([-+0-9.]+) \(SE ([0-9.]+)\), n (\d+)\s+reading: (.+)')
RD={'R3 Price overshoot':'price overshoot (R3)','R2 Better loans':'better loans (R2)','R1 Slow marks':'slow marks (R1)'}
for lab,L in [('model labels','Model'),('fund labels','Fund')]:
    for smp,Sm in SS.items():
        for b in ('AB' if (lab=='model labels' and Sm=='Reg') else 'A'):
            g=rp.search(blk(TXT[('s3',b)],lab,smp)).groups(); tag=f'SThree{"" if b=="A" else "B"}{L}{Sm}'; S=f'Step 3 Sep 2026, build {b}, {lab}, {smp}'
            put4(f'{tag}Q',float(g[0]),'f2',S+', Q traded',sign=True); put4(f'{tag}QSE',float(g[1]),'f2',S)
            put4(f'{tag}P',float(g[8]),'f2',S+', P private',sign=True); put4(f'{tag}PSE',float(g[9]),'f2',S)
            put4(f'{tag}Reading',RD.get(g[19].strip(),g[19].strip()),'str',S+', reading')
            if b=='A' and L=='Model':
                put4(f'{tag}QLo',float(g[2]),'f2',S,sign=True); put4(f'{tag}QHi',float(g[3]),'f2',S,sign=True); put4(f'{tag}PLo',float(g[10]),'f2',S,sign=True); put4(f'{tag}PHi',float(g[11]),'f2',S,sign=True)
                put4(f'{tag}QN',int(g[4]),'int',S+', traded positions'); put4(f'{tag}QNSw',int(g[5]),'int',S+', traded software positions'); put4(f'{tag}QNBorr',int(g[6]),'int',S+', traded software borrowers'); put4(f'{tag}QFunds',int(g[7]),'int',S+', funds')
                put4(f'{tag}PN',int(g[12]),'int',S+', private positions'); put4(f'{tag}PNSw',int(g[13]),'int',S+', private software positions'); put4(f'{tag}PNBorr',int(g[14]),'int',S); put4(f'{tag}PFunds',int(g[15]),'int',S+', funds')
                putse4(f'{tag}Gap',float(g[16]),float(g[17]),S+', traded minus private, fund x sector FE (context)')
            log(f'[S3] {tag}: Q {g[0]} ({g[1]}) P {g[8]} ({g[9]}) -> {g[19]}')
m=re.search(r'Funds in the June estimate: (\d+); of these with a 30 Sep 2026 report filed by 20 Nov 2026 \(rows in the pull\): (\d+)\s+June-estimate funds without a Sep 2026 report in the pull: (\d+) \(of which Sep fiscal year-end: (\d+)\)',s3)
put4('SThreeNJuneFunds',int(m.group(1)),'int','funds in the June Step 3 estimate'); put4('SThreeNSepFunds',int(m.group(2)),'int','of these with a Sep 2026 report by 20 Nov'); put4('SThreeNMissFunds',int(m.group(3)),'int','without'); put4('SThreeNMissSepFye',int(m.group(4)),'int','of which Sep FYE')
m=re.search(r'traded\s+([-+0-9.]+) \(SE ([0-9.]+); 90% CI [^)]*\); n \d+.*?\n\s+private\s+([-+0-9.]+) \(SE ([0-9.]+)',s3)
put4('SThreeAJuneTr',float(m.group(1)),'f2','June traded estimate recomputed on build A (registered -7.9)',sign=True); put4('SThreeAJunePr',float(m.group(3)),'f2','June private estimate on build A',sign=True)
put4('SepQShare',float(MAC4['SThreeModelRegQ'][1])/abs(float(m.group(1))),'pct0','Q as a share of the June traded markdown on the same panel')
# ---------------------------------------------------------------- persistence checks (build A counts, build B alongside)
for b in 'AB':
    cfg=BUILDS[b]; p=load_sep(cfg['VD'],cfg['PB'],cfg['OLDPANEL']); SEPD=pd.Timestamp('2026-09-30'); assert p.ddate.max()==SEPD
    keys=CM.cohort_keys(p); d0=CM.frozen(p,keys); d0['sw']=(d0.grp=='SOFT').astype(float)
    r=CM.es(d0); bm,sm=CM.get(r,'mark','2026-09'); bb,sb=CM.get(r,'b99','2026-09'); jm,_=CM.get(r,'mark','2026-06'); jb,_=CM.get(r,'b99','2026-06')
    tag='' if b=='A' else 'B'; S=f'persistence check, build {b}: Step 1 headline spec on the frozen cohort with Sep 2026 added, beta_Sep26 relative to Sep 2025'
    putse4(f'Pers{tag}Mark',bm,sm,S+', mark'); putse4(f'Pers{tag}Below',bb,sb,S+', Below99',d=1); put4(f'Pers{tag}N',r['n']['pos'],'int',S+', positions'); put4(f'Pers{tag}Funds',r['n']['funds'],'int',S+', funds')
    log(f'[pers] build {b}: cohort keys {len(keys)}; {CM.sample_note(r,"persistence")} | mark Jun {jm:+.2f} Sep {bm:+.2f} ({sm:.2f}) | b99 Jun {jb:+.1f} Sep {bb:+.1f} ({sb:.1f})')
    vm='as expected' if -4.0<=bm<=-1.5 else ('surprising' if bm>-1.0 else 'outside the expected range but not surprising')
    vb='as expected' if bb>20 else ('surprising' if bb<10 else 'outside the expected range but not surprising')
    put4(f'Pers{tag}MarkV',vm,'str','PAP v5: expectation -4.0 to -1.5; surprising above -1.0'); put4(f'Pers{tag}BelowV',vb,'str','PAP v5: expectation above +20; surprising below +10')
    ch=changes(p); R=rank_table(ch,npost=4)
    for y,Y in [('db99','Below'),('dmark','Mark')]:
        rk,n,t=rank_of(R,y); put4(f'Pers{tag}Rank{Y}',rk,'int',f'software rank among {n} groups, Dec25-Sep26 sum / (2 x pre SD), build {b}')
        nx=t[t.grp!='SOFT'].iloc[0]; put4(f'Pers{tag}Scaled{Y}',float(t.loc[t.grp=='SOFT','T'].iloc[0]),'f1','software scaled statistic, four post quarters',sign=True)
        put4(f'Pers{tag}Next{Y}',float(nx['T']),'f1','next group',sign=True); put4(f'Pers{tag}Next{Y}Name',NAMES[nx.grp].lower(),'str','next group')
        log(f'[pers] build {b} rank {y}: {rk} of {n}; top '+'; '.join(f'{r_.grp} {r_["T"]:+.2f}' for _,r_ in t.head(4).iterrows()))
    put4(f'Pers{tag}NGroups',n,'int','groups')
    if b=='A':
        # the 76 confirmatory funds and the September pull
        EF=(_ROOT+'/raw-sec/edgar-pull/early-filers-2026-11-08/early_filers_2026-11-08.csv')
        ef=pd.read_csv(EF,dtype=str); A={int(c) for c in ef[ef.reason.str.startswith('A')].cik}|{1803498}
        pos=pd.read_parquet(f'{cfg["VD"]}/positions.parquet',columns=['cik','bdc','ddate','form'])
        fye=pos[pos.form.str.startswith('10-K',na=False)&(pos.ddate<=CM.JUN)].groupby('cik').ddate.agg(lambda s:s.dt.month.mode().iloc[0]); sepfye={int(c) for c in fye[fye==9].index}
        c9=p[(p.ddate==CM.BASE)&p.pkey.isin(keys)]; cf={int(c) for c in c9.cik.unique()}-A-sepfye
        filed={int(c) for c in p.loc[p.ddate==SEPD,'cik'].unique()}; miss=sorted(cf-filed)
        nm=pos.drop_duplicates('cik').set_index('cik').bdc
        put4('SepNConfFunds',len(cf),'int','confirmatory funds (cohort funds not on list A, no Sep FYE)'); put4('SepNConfFiled',len(cf&filed),'int','of these with a 30 Sep 2026 report in the pull')
        put4('SepNConfMissing',len(miss),'int','confirmatory funds without a 30 Sep 2026 report filed by 20 Nov 2026')
        put4('SepConfMissNames','; '.join(str(nm.get(c,c)).strip() for c in miss) if miss else 'none','str','names as filed')
        pr=c9.groupby(c9.cik.astype(int)).principal.sum(); put4('SepConfFiledPrin',pr[list(cf&filed)].sum()/pr.sum(),'pct1','cohort principal share of the confirmatory funds with a Sep 2026 report')
        log(f'[funds] confirmatory {len(cf)}, filed {len(cf&filed)}, missing {len(miss)}: '+'; '.join(f'{c} {nm.get(c,c)}' for c in miss))

# ---------------------------------------------------------------- 6. draft 6 additions
# 6a. the numbers of the deviation (tie-break and Carlyle): 21 Nov stored file, deterministic rebuild without Carlyle, and the
#     two PC rebuilds of 26 Nov (read from the deviation note's table, sep2026-ccs-deviation-and-rerun-2026-11-26.md)
def c1conf(txt): return float(re.search(r'\[model labels\] confirmatory \(not list A, no Sep FYE\): .*?C1 ([-+0-9.]+)',txt).group(1))
put4('DsixSepCOneOld',c1conf(open(f'{SEP21}/results_C1_C1b_buildA_2026-11-21.txt').read()),'f2','C1, 21 Nov stored run (148 filings, old tie order)',sign=True)
put4('DsixSepCOneOldFour',c1conf(open(f'{SEP21}/results_C1_C1b_buildA_2026-11-21.txt').read()),'f3','same, three decimals',sign=True)
put4('DsixSepCOneDetNoCcs',c1conf(open(f'{SEP}/results_C1_C1b_buildA_noCCS_check_2026-11-26.txt').read()),'f3','C1, deterministic rebuild without Carlyle (148 filings)',sign=True)
dn=open((_ROOT+'/notes/sep2026-run/sep2026-ccs-deviation-and-rerun-2026-11-26.md')).read()
hy=float(re.search(r'\*\*Build A \+ CCS, 21 Nov links \(149\)\*\* \| \*\*([-+0-9.]+)',dn).group(1)); pc=float(re.search(r'\| Build A, fresh PC links \(148\) \| ([-+0-9.]+)',dn).group(1))
put4('DsixSepCOneHybrid',hy,'f3','C1 with Carlyle on the 21 Nov links (PC, deviation note)',sign=True); put4('DsixSepCOnePcFresh',pc,'f3','C1, fresh PC build without Carlyle (deviation note)',sign=True)
put4('DsixSepTieSpread',abs(hy-float(MAC4['SepCOneFour'][1])),'f2','distance between the hybrid and the deterministic C1')
put4('DsixSepNFilings',149,'int','filings in the September pull with Carlyle (manifest of the 26 Nov pull version)')
log(f'[dev] C1: 21 Nov {c1conf(open(f"{SEP21}/results_C1_C1b_buildA_2026-11-21.txt").read()):+.4f}; deterministic no CCS {c1conf(open(f"{SEP}/results_C1_C1b_buildA_noCCS_check_2026-11-26.txt").read()):+.4f}; PC fresh no CCS {pc:+.4f}; hybrid {hy:+.4f}; deterministic with CCS {MAC4["SepCOneFour"][1]}')
# 6b. two-sided 5% trims (#122, exploratory): run the registered scripts in-process on build A to reuse their samples
import runpy, contextlib, io
vd=f'{SCR}/vdA'; cfg=BUILDS['A']; os.environ.update(VD=vd,OLDPANEL=cfg['OLDPANEL'],PB=cfg['PB'],POSMAP=cfg['POSMAP'])
cwd=os.getcwd(); os.chdir(SCR); argv=sys.argv
with contextlib.redirect_stdout(io.StringIO()):
    sys.argv=['c1x.py']; G1=runpy.run_path(f'{SC}/c1x.py'); sys.argv=['s3.py',vd]; G3=runpy.run_path(f'{SC}/v2_step3_sep2026_2026-11-21.py')
sys.argv=argv; os.chdir(cwd)
lab,sw=G1['labels']['model labels']; r=G1['change'](lab&G1['samples']['confirmatory (not list A, no Sep FYE)'])
pp=G1['p']; r['sw']=sw.reindex(pp.index)[pp.pkey.drop_duplicates().index].set_axis(pp.pkey.drop_duplicates().values).reindex(r.index).astype(float); r=r.dropna(subset=['sw'])
b0,_=G1['c_soft'](r,'dmark'); assert abs(b0-float(MAC4['SepCOneFour'][1]))<5e-4, (b0,MAC4['SepCOneFour'])
q=r.groupby('sw').dmark; keep=(r.dmark>q.transform(lambda v:v.quantile(0.05)))&(r.dmark<q.transform(lambda v:v.quantile(0.95)))
bt,st=G1['c_soft'](r[keep],'dmark'); putse4('DsixSepCOneTwoTrim',bt,st,'C1 confirmatory, 5% largest falls and 5% largest rises dropped within software and within other loans (exploratory)')
log(f'[trim] C1 {b0:+.4f} -> two-sided 5% trim {bt:+.4f} ({st:.4f}), n {int(keep.sum())} of {len(r)}')
s=G3['s']; s=s[s.cik.isin(G3['samples']['registered sample (June-estimate funds filed by 20 Nov)'])]
Q0=G3['est'](s,'dSep','traded'); P0=G3['est'](s,'dSep','private'); assert abs(Q0[0]-float(MAC4['SThreeModelRegQ'][1]))<0.006 and abs(P0[0]-float(MAC4['SThreeModelRegP'][1]))<0.006
z=s.dropna(subset=['dSep']); z=z[z.cls.isin(['traded','private'])]; q=z.groupby(['cls','sw']).dSep
z=z[(z.dSep>q.transform(lambda v:v.quantile(0.05)))&(z.dSep<q.transform(lambda v:v.quantile(0.95)))]
Q1=G3['est'](z,'dSep','traded'); P1=G3['est'](z,'dSep','private')
putse4('DsixSepQTwoTrim',Q1[0],Q1[1],'Step 3 Sep Q, two-sided 5% trim in every class x sector cell (exploratory)'); putse4('DsixSepPTwoTrim',P1[0],P1[1],'Step 3 Sep P, same trim')
put4('DsixSepTwoTrimReading',RD[G3['reading'](P1[0],Q1[0])],'str','reading of the trimmed Step 3 point estimates')
log(f'[trim] Step 3 Q {Q0[0]:+.2f} -> {Q1[0]:+.2f} ({Q1[1]:.2f}); P {P0[0]:+.2f} -> {P1[0]:+.2f} ({P1[1]:.2f}); reading {G3["reading"](P1[0],Q1[0])}')
# 6c. non-accrual flags (#114, saved for draft 2): share of Step 3 software / other positions flagged non-accrual, by class and date
pos=pd.read_parquet(f'{cfg["VD"]}/positions.parquet',columns=['pos_id','nonaccrual']); na=pos.set_index('pos_id').nonaccrual
pa=pd.read_parquet(f'{cfg["VD"]}/panel.parquet',columns=['pos_id','pkey','ddate']); pa['na']=pa.pos_id.map(na)
st=s.reset_index()[['pk','cls','sw']].rename(columns={'pk':'pkey'})
for d_ in ['2025-09-30','2026-06-30','2026-09-30']:
    x=pa[pa.ddate==pd.Timestamp(d_)].merge(st,on='pkey'); t=x[x.cls.isin(['traded','private'])].groupby(['cls','sw']).na.agg(['mean','sum','size'])
    log(f'[nonaccrual] {d_}: '+'; '.join(f'{c}/{"sw" if w else "other"} {m*100:.1f}% ({int(k)} of {int(n)})' for (c,w),(m,k,n) in t.iterrows()))
log('[nonaccrual] note: the XBRL non-accrual flag is set on only '+str(int(pos.nonaccrual.sum()))+' of '+str(len(pos))+' position rows, so it undercounts; saved for draft 2, not in the text')
extra='\\newif\\ifnport\n'+('\\nporttrue\n' if NPORT else '\\nportfalse\n')
h=write_macros(f'{OUT}/numbers-sep.tex','% numbers-sep.tex: registered September 2026 tests and persistence checks. Written by bdc/scripts/draft6/d6_sep_numbers.py.\n% \\ifnport is false until the September 2026 N-PORT leg meets the PAP v5 conditions; the C2 macros exist only then.\n',extra)
write_log(f'{OUT}/numbers-sep-log.txt',f'd6_sep_numbers.py run {time.strftime("%Y-%m-%d %H:%M UTC",time.gmtime())}, {time.time()-t0:.0f} s, numbers-sep.tex sha256 {h[:16]}',LOG)
print('numbers-sep.tex sha256',h)
