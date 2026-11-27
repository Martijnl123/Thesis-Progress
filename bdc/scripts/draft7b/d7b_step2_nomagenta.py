# Draft 7b (27 Nov 2026): Step 2 without the eight Magenta Security positions (post hoc correction; examiner report
# draft 7, M1). Magenta's four tranches share one maturity (27 Jul 2028), so the N-PORT match mixes their prices.
# Reruns the registered Step 2 code (draft2/d2_step2.run) with one filter in build(), plus the parts of
# draft3/d3_extra.step2_extra, draft6/d6_tables.py, draft6/d6_derived.py and v3_numbers.py that use the Step 2 sample.
# Writes <draft folder>/numbers-d7b-step2.tex with \renewcommand for every macro whose value changes, and a log.
# Usage: python3 d7b_step2_nomagenta.py <draft folder>
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..','..')))  # repo root
import sys, re, glob
sys.path.insert(0,(_ROOT+'/bdc/scripts/draft2')); sys.path.insert(0,(_ROOT+'/bdc/scripts'))
import d2_common as CM
from d2_common import *
import d2_step2 as S2, statsmodels.formula.api as smf
CM.PCT_SIGN=True
OUT=sys.argv[1]
MAGENTA={41678,41679,41680,43748,43749,43751,43752,51490}
_build=S2.build
def build_nm(p,matchfile,labmode='model'):
    q=_build(p,matchfile,labmode); return q[~q.pkey.isin(MAGENTA)].copy()
S2.build=build_nm
L=[]; log=lambda x:(L.append(str(x)),print(x,flush=True))
p=load(); ident=p.drop_duplicates('pkey').set_index('pkey').ident
assert set(_build(p,'nport_matches_coupon.parquet').pipe(lambda q:q[q.pkey.map(ident).fillna('').str.contains('magenta',case=False)]).pkey)==MAGENTA
T={}; q=S2.run(p,T,log)
# rank (d3_extra.step2_extra, C6) and lost-quote interval, lost shares
g=p.drop_duplicates('pkey').set_index('pkey').grp; q['grp']=q.pkey.map(g); s9=q[q.ddate==BASE]
gq=s9.groupby('grp').size(); gq=gq[~gq.index.isin(['SOFT','ITSV','DATA','HARD','UNKN','FUND'])]; rows=[]
for gg in ['SOFT']+list(gq[gq>=30].index):
    z=q if gg=='SOFT' else q[q.grp!='SOFT']; z=z.copy(); z['sw']=(z.grp==gg).astype(float); th,_=S2.es2(z,'gap'); rows.append((gg,*th['202606']))
R=pd.DataFrame(rows,columns=['grp','th','se']); R['t']=R.th/R.se
for col,Y in [('th','Th'),('t','T')]:
    t=R.sort_values(col,ascending=False).reset_index(drop=True); put(f'StepTwoRank{Y}',int(t.index[t.grp=='SOFT'][0])+1,'int','rank, lag direction first')
    a=R.assign(a=R[col].abs()).sort_values('a',ascending=False).reset_index(drop=True); put(f'StepTwoRankAbs{Y}',int(a.index[a.grp=='SOFT'][0])+1,'int','rank on |.|')
put('StepTwoNRank',len(R),'int','groups'); log('[rank]\n'+R.sort_values('th').round(2).to_string())
b,s_=MAC['ThetaJunLow'][1],MAC['ThetaJunLowSE'][1]; put('ThetaJunLowHi',b+1.96*s_,'f1','',sign=True); put('ThetaJunLowLo',b-1.96*s_,'f1','',sign=True)
put('LostSw',1-MAC['KeepSw'][1],'pct0',''); put('LostOt',1-MAC['KeepOt'][1],'pct0','')
w=q[q.px.notna()].groupby(['ddate','sw']).px.mean().unstack()
put('LoanPxSwSepDec',w.loc[DEC,1.0]-w.loc[BASE,1.0],'f1','',sign=True); put('LoanPxSwDecMar',w.loc[MAR,1.0]-w.loc[DEC,1.0],'f1','',sign=True)
# var(G) ratios by quarter (draft4/d4_tables reads them from the step2 log)
vg=T['varG']; r=(vg[1.0]/vg[0.0])
for k,Q in [('202512','Dec'),('202603','Mar'),('202606','Jun')]: put(f'VarGRatio{Q}',float(r[k]),'f1','var(G) software / other')
# d6_tables: pooled identical pairs dropped, Table 6 counts
ww=S2.wide(q); xs=pd.concat([S2.pa_frame(ww,a,b_) for a,b_ in S2.QQ[1:]]); xn=xs[(xs.g0.abs()<=10)&~xs.same]
rr=smf.ols('dm~gap*soft+C(t)',xn).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(xn.clu)[0]}); putse('DsixDlPoolNosame',rr.params['gap:soft'],rr.bse['gap:soft'],'')
for k,Q in [('2025-06-30','JunPrev'),('2025-12-31','Dec'),('2026-03-31','Mar'),('2026-06-30','Jun')]:
    z=q[(q.ddate==pd.Timestamp(k))&q.px.notna()]; put(f'DsixNTheta{Q}',len(z),'int',''); put(f'DsixNTheta{Q}Sw',int(z.sw.sum()),'int','')
# derived forms used in the text
put('ThMarkJunAbs',abs(MAC['ThMarkJun'][1]),'f2',''); put('ThPxJunAbs',abs(MAC['ThPxJun'][1]),'f2','')
put('ThetaJunLowHiAbs',abs(MAC['ThetaJunLowHi'][1]),'f1','')
num=lambda n: float(re.sub(r'[^0-9.+-]','',MAC[n][0]))
put('DsixRatioQuoted',num('ThMarkJun')/num('ThPxJun'),'pct0','ratio of rounded table values')
pr=[l for f in glob.glob(f'{OUT}/numbers*.tex') for l in open(f) if '{\\PrivOnlyJun}' in l][0]
put('DsixRatioPrivate',float(re.search(r'\$?([-+0-9.]+)',pr.split('}{')[1]).group(1))/num('ThPxJun'),'pct0','')
# Table 6 and Table A6 rows (written as text for the table files)
th=T['theta']; P=T['pa']
for k in ['202506','202512','202603','202606']:
    log(f'[T6] {k}: mark {th["mark"][k][0]:+.2f} ({th["mark"][k][1]:.2f}) px {th["px"][k][0]:+.2f} ({th["px"][k][1]:.2f}) gap {th["gap"][k][0]:+.2f} ({th["gap"][k][1]:.2f})')
log('[A6]\n'+P.round(3).to_string())
# compare with the current values and write overrides
cur={}
for f in sorted(glob.glob(f'{OUT}/numbers*.tex')):
    if f.endswith('numbers-d7b-step2.tex'): continue
    for l in open(f):
        m=re.match(r'\\(?:re)?newcommand\{\\([A-Za-z]+)\}\{(.*)\}',l.strip().split(' %')[0])
        if m: cur[m.group(1)]=m.group(2)
used=set(re.findall(r'\\([A-Za-z]+)',' '.join(open(f).read() for f in glob.glob(f'{OUT}/*.tex')+glob.glob(f'{OUT}/tables/*.tex') if 'numbers' not in f)))
KEEP={'NStepTwoAll','NStepTwoAllSw'}  # count the Step 3 class search; the registered class file is unchanged
ch=[(n,cur.get(n),v[0]) for n,v in MAC.items() if n in cur and cur[n]!=v[0] and n not in KEEP]
with open(f'{OUT}/numbers-d7b-step2.tex','w') as fh:
    fh.write('% numbers-d7b-step2.tex: Step 2 without the eight Magenta Security positions (post hoc, 27 Nov 2026).\n% Written by bdc/scripts/draft7b/d7b_step2_nomagenta.py; old value in each comment.\n')
    for n,o,v in ch: fh.write(f'\\renewcommand{{\\{n}}}{{{v}}} % was {o}{"" if n in used else " (not used in text)"}\n')
    fh.write('% Draft 7 values with Magenta kept (draft7-2026-11-26/numbers.tex, numbers-d6der.tex), quoted once in Section 5.3:\n')
    fh.write('\\newcommand{\\DsevenbOldDlSep}{'+cur['DlSep']+'}\n\\newcommand{\\DsevenbOldThetaJun}{'+cur['ThetaJun']+'}\n\\newcommand{\\DsevenbOldThetaJunSE}{'+cur['ThetaJunSE']+'}\n\\newcommand{\\DsevenbNMagenta}{eight}\n')
    fh.write('% NStepTwoAll/NStepTwoAllSw kept at draft 7 values: they count the Step 3 class search, and the Step 3 class file is registered and unchanged.\n')
log(f'[changed] {len(ch)} macros: '+'; '.join(f'{n} {o} -> {v}' for n,o,v in ch))
log('[new, not in current files] '+', '.join(n for n in MAC if n not in cur))
open(f'{OUT}/numbers-d7b-step2-log.txt','w').write('\n'.join(L)+'\n')
