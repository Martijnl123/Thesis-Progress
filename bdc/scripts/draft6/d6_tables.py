# Draft 6 (26 Nov 2026): count fixes and the rebuilt tables of the final rewrite -> <out>/numbers-d6tab.tex and <out>/tables/*_d6.tex.
# Data to 30 June 2026 (d2_common.load refuses later rows); September rows only through numbers-sep.tex macros.
# Numbers either come from data here (macros with prefix Dsix) or are rows copied verbatim from tables that v3_numbers.py and the
# draft 4/5 scripts wrote (tab_robust.tex, tab_robust_more_d5.tex, tab_step2.tex, tab_step3.tex, tab1_cohort.tex); no number is typed.
#   #62  Table 1 'Level 2 price' row on the registered Step 2 sample (one definition, as in the text)
#   #52  funds'-labels sample: cohort positions with a fund label, of which IT services / other tech, singletons
#   #152 Step 1 managers on 30 Sep 2025 (Table A2 counts managers over all quarters)
#   #154 Step 3 traded positions: directly matched vs co-held in a matched tranche
#   #107 Table 6 pooled 'identical pairs dropped' cell; #106 positions per row of Table 6 panel A
#   #162 blind sample 2x2 counts; #118 fund counts of the Step 3 funds'-label rows
# Usage: python3 d6_tables.py <draft6 folder>
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..','..')))  # repo root
import sys, re, time
sys.path.insert(0,(_ROOT+'/bdc/scripts/draft4')); sys.path.insert(0,(_ROOT+'/bdc/scripts/draft2'))
from d4_common import *
import d2_step2 as S2, statsmodels.formula.api as smf
OUT=sys.argv[1]; TAB=f'{OUT}/tables'; LOG=[]; t0=time.time()
def log(s): LOG.append(str(s)); print(s,flush=True)
BASE,JUN=CM.BASE,CM.JUN; p=CM.load(); assert p.ddate.max()==JUN
keys=CM.cohort_keys(p); d0=CM.frozen(p,keys); d0['sw']=(d0.grp=='SOFT').astype(float); c9=d0[d0.ddate==BASE]
# ---- #62 Table 1 Level 2 row
q=S2.build(p,'nport_matches_coupon.parquet'); s9=q[q.ddate==BASE]; assert len(s9)==2316
for nm,m in [('Sw',c9.sw==1),('Ot',c9.sw==0),('All',c9.sw>=0)]:
    k=c9[m].pkey; put4(f'DsixLtwo{nm}',k.isin(set(s9.pkey)).mean(),'pct1',f'cohort positions with a verified Level 2 price in the registered Step 2 sample, {nm}')
t1=open(f'{TAB}/tab1_cohort.tex').read(); old=[l for l in t1.split('\n') if l.startswith('Level 2 price')][0]
t1=t1.replace(old,'Level 2 price, registered Step 2 sample (\\%) & \\DsixLtwoSw & \\DsixLtwoOt & \\DsixLtwoAll \\\\'); open(f'{TAB}/tab1_cohort_d6.tex','w').write(t1)
log(f'[#62] Level 2 share sw {MAC4["DsixLtwoSw"][0]} other {MAC4["DsixLtwoOt"][0]} all {MAC4["DsixLtwoAll"][0]} (old row: {old})')
# ---- #52 funds' labels sample
fl=c9.fsec.notna(); it=c9.fsec.isin(['it_services','tech_other'])
r=CM.es(d0[d0.pkey.isin(set(c9[fl&~it].pkey))].assign(sw=lambda z:(z.fsec=='software').astype(float)))
put4('DsixFundLabPos',int(fl.sum()),'int','cohort positions with a funds label (pre-Sep-2025 vote)')
put4('DsixFundLabItsv',int((fl&it).sum()),'int','of which filed as IT services or other technology (left out of the funds-label sample)')
put4('DsixFundLabEst',r['n']['pos'],'int','funds-label Step 1 estimation sample'); put4('DsixFundLabSingle',int((fl&~it).sum())-r['n']['pos'],'int','dropped as singletons')
log(f'[#52] fund label {int(fl.sum())}, IT/tech {int((fl&it).sum())}, estimation {r["n"]["pos"]}, singletons {int((fl&~it).sum())-r["n"]["pos"]}')
# ---- #152 managers
r1=CM.es(d0); pos_est=r1['n']['pos']
multi=d0.groupby('pkey').ddate.nunique(); k2=set(multi[multi>1].index)
put4('DsixHoneMgrSep',c9[c9.pkey.isin(k2)].family.nunique(),'int','managers of Step 1 positions on 30 Sep 2025 (positions seen in 2+ quarters)')
put4('DsixCohortMgrAll',d0.family.nunique(),'int','manager names over all quarters of the cohort panel')
log(f'[#152] Step 1 managers over all quarters {r1["n"]["managers"]}; on 30 Sep 2025 {MAC4["DsixHoneMgrSep"][0]}; cohort panel all quarters {d0.family.nunique()}')
# ---- #154 Step 3 traded positions
CL=pd.read_parquet(f'{CM.V}/step3_classes_2026-11-17.parquet'); ntr=int((CL.cls=='traded').sum())
put4('DsixSThreeTrPos',ntr,'int','Step 3 traded positions'); put4('DsixSThreeTrViaTranche',ntr-2870,'int','traded because a co-held position in the same tranche has a verified price (3,400 minus the 2,870 matched directly)')
assert MAC4['DsixSThreeTrPos'][1]==3400
# ---- #107/#106 Table 6
w=S2.wide(q); xs=pd.concat([S2.pa_frame(w,a,b_) for a,b_ in S2.QQ[1:]]); xn=xs[(xs.g0.abs()<=10)&~xs.same]
rr=smf.ols('dm~gap*soft+C(t)',xn).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(xn.clu)[0]}); putse4('DsixDlPoolNosame',rr.params['gap:soft'],rr.bse['gap:soft'],'Step 2 pooled dlambda, 10-point trim, identical pairs dropped')
put4('DsixNDlPoolNosameSw',int(xn.soft.sum()),'int','software pairs')
for k,Q in [('2025-06-30','JunPrev'),('2025-12-31','Dec'),('2026-03-31','Mar'),('2026-06-30','Jun')]:
    z=q[(q.ddate==pd.Timestamp(k))&q.px.notna()]; put4(f'DsixNTheta{Q}',len(z),'int',f'Step 2 positions with a price at {k}'); put4(f'DsixNTheta{Q}Sw',int(z.sw.sum()),'int','software')
t6=open(f'{TAB}/tab_step2.tex').read()
t6=t6.replace(' & BDC mark & Market price & Gap, $\\theta_k$ & & \\\\',' & BDC mark & Market price & Gap, $\\theta_k$ & Positions & Software \\\\')
for lab,Q in [('June 2025','JunPrev'),('December 2025','Dec'),('March 2026','Mar'),('June 2026','Jun')]:
    l=[x for x in t6.split('\n') if x.startswith(lab+' &')][0]; assert l.endswith('& & \\\\'); t6=t6.replace(l,l[:-len('& & \\\\')]+f'& \\DsixNTheta{Q} & \\DsixNTheta{Q}Sw \\\\')
l=[x for x in t6.split('\n') if x.startswith('Pooled')][0]; assert l.count(' & & ')==1; t6=t6.replace(l,l.replace(' & & ',' & \\DsixDlPoolNosame{} (\\DsixDlPoolNosameSE) & '))
open(f'{TAB}/tab_step2_d6.tex','w').write(t6); log(f'[#107] pooled identical pairs dropped {MAC4["DsixDlPoolNosame"][0]} ({MAC4["DsixDlPoolNosameSE"][0]})')
# ---- #162 blind sample 2x2
K=pd.read_csv((_ROOT+'/bdc/industry_llm/validation/key_DO_NOT_OPEN_BEFORE_CODING_2026-11-08.csv')); H=pd.read_csv((_ROOT+'/bdc/industry_llm/validation/coded-2026-11-21/coder1_final.csv'))
z=H.merge(K,on='code_id'); z=z[z.is_software_0_1.notna()&(z.group_code!='UNKN')]; z['m']=z.stratum.str.startswith('SOFT'); z['h']=z.is_software_0_1==1
for a,A in [(True,'Ms'),(False,'Mn')]:
    for b,Bb in [(True,'Hs'),(False,'Hn')]: put4(f'DsixVal{A}{Bb}',int(((z.m==a)&(z.h==b)).sum()),'int',f'blind sample count, model software={a}, my coding software={b}')
log(f'[#162] 2x2 n={len(z)}: '+str({k:MAC4[k][0] for k in MAC4 if k.startswith('DsixVal')}))
# ---- #118 fund counts, Step 3 funds'-label rows (build A result file of 26 Nov)
s3=open((_ROOT+'/bdc/v2/sep2026_d6/results_step3_buildA_2026-11-26.txt')).read()
i=s3.index('[fund labels] change 30 Jun 2026'); blk=s3[i:]
for smp,T_ in [('registered sample','Reg'),('76 confirmatory funds','Conf')]:
    j=blk.index(smp); m=re.search(r'P private .*?funds (\d+)',blk[j:j+700]); put4(f'DsixSThreeFund{T_}PFunds',int(m.group(1)),'int',f'Step 3 Sep, funds labels, {smp}, funds in P')
# ---- rebuilt tables from rows of generated tables
def rows(f): return {l.split(' & ')[0].strip():l for l in open(f'{TAB}/{f}').read().split('\n') if ' & ' in l}
R3=rows('tab_robust.tex'); Rm=rows('tab_robust_more_d5.tex')
def row(lab,src=R3,new=None): l=src[lab]; return (l if new is None else l.replace(lab,new,1))+'\n'
hdr='\\begin{tabular}{p{9.0cm}cc}\n\\toprule\n & Mark & Below99 \\\\\n\\midrule\n'
T5=(hdr+'\\multicolumn{3}{l}{\\emph{A. Label source (June 2026 coefficient of Equation~\\eqref{eq:h1})}} \\\\\n'
   +row('Headline (model labels)',new='Model labels (headline; \\NHonePos{} positions)')
   +row("Funds' own labels, same cohort",new="Funds' own labels, same cohort (\\DsixFundLabEst{} positions)")
   +row('Both labels agree')
   +'Model labels corrected for label error with the blind sample (DSL) & \\ValDslMarkJun{} (\\ValDslMarkJunSE) & \\ValDslBelowJun{} (\\ValDslBelowJunSE) \\\\\n'
   +'Open-weights model, knowledge cutoff June 2024 (disclosed addition) & \\OmFlagMark{} (\\OmFlagMarkSE) & \\OmFlagBelow{} (\\OmFlagBelowSE) \\\\\n'
   +'\\midrule\n\\multicolumn{3}{l}{\\emph{B. Identification}} \\\\\n'
   +row('Placebo cohort September 2023, shock March 2024 (funds\' labels)')+row('Placebo cohort September 2024, shock March 2025 (funds\' labels)')
   +row('Common shock: software change in March 2026 predicted at the pre-period slope')+row('\\quad actual software change in March 2026')
   +'Drift $\\bar M$ that brings zero into the June interval, model labels & \\BdModelMarkJun & \\BdModelBelowJun \\\\\n'
   +"\\quad same, funds' labels & \\BdFundMarkJun & \\BdFundBelowJun \\\\\n"
   +'\\midrule\n\\multicolumn{3}{l}{\\emph{C. Sample}} \\\\\n'
   +row('Leave one of the 25 largest managers out (range)')+row('Loans that left filled in, 10th percentile mark (range)')
   +'\\bottomrule\n\\end{tabular}\n')
open(f'{TAB}/tab_robust_d6.tex','w').write(T5)
# Appendix: checks listed in the plan (one row each)
P0=open(f'{TAB}/tab_pap.tex').read(); a=P0.index('\\multicolumn{3}{l}{\\emph{A.'); b_=P0.index('\\midrule',a)
step1=['Labels of the second model run where available','IT services, data and hardware left out','Weighted by principal','Manager $\\times$ quarter instead of fund $\\times$ quarter effects',
       'Standard errors by borrower and manager','Principal at least \\$1 million','Without credit lines and delayed-draw loans','Principal did not grow (no interest added)',
       'Positions still held in June 2026','Maturity reported','\\quad same, with time to maturity $\\times$ quarter','Maturity not reported']
extra=''.join(row(l,new=('Manager $\\times$ quarter instead of fund $\\times$ quarter effects (\\NMqGroups{} instead of \\NFqGroups{} groups)' if l.startswith('Manager') else None)) for l in step1)
extra+=row('Both labels agree')
PA=P0[:b_]+extra+P0[b_:]
c_=PA.index('Stricter comparison'); PA=PA[:c_]+('Rank, last 8 pre-period changes as the scale & \\DsixRankEightMark & \\DsixRankEightBelow \\\\\n'
   'Rank, last 10 pre-period changes as the scale & \\DsixRankTenMark & \\DsixRankTenBelow \\\\\n'
   'Rank, sum from March 2026 & \\DsixRankMarMark & \\DsixRankMarBelow \\\\\n'
   'Rank, scale from overlapping three-quarter pre-period sums (disclosed addition) & \\DsixRankOvMark & \\DsixRankOvBelow \\\\\n')+PA[c_:]
PA=PA.replace('Below 99','Below99'); open(f'{TAB}/tab_pap_d6.tex','w').write(PA)
# Step 3 table: add the rows the text cites (#111, #112)
t7=open(f'{TAB}/tab_step3.tex').read(); l=[x for x in t7.split('\n') if x.startswith('Managers holding the loan')][0]
t7=t7.replace(l,l+'\nHeld by seven or more managers (\\%), software & \\TrSevenShare & \\PrSevenShare & & \\\\\nMedian BDC principal per borrower (\\$m), software & \\MedBorrPrinTr & \\MedBorrPrinPr & & \\\\')
l=[x for x in t7.split('\n') if x.startswith('Lost more than 5 points')][0]
t7=t7.replace(l,l+'\nShare below 99, change to June, relative to other loans of the class (pp) & \\WithinTrBelowJun & \\WithinPrBelowJun & & \\\\')
l=[x for x in t7.split('\n') if x.startswith('Traded minus private')][0]
t7=t7.replace(l,l+'\n\\quad funds\' own labels: traded minus private; private only & \\multicolumn{4}{c}{\\GapFund{} (\\GapFundSE); \\PrivOnlyFund{} (\\PrivOnlyFundSE)} \\\\\n\\quad winsorised at 5\\% and 95\\%; two-sided 5\\% trim & \\multicolumn{4}{c}{\\GapWins{} (\\GapWinsSE); \\DsixGapTwoTrim{} (\\DsixGapTwoTrimSE)} \\\\')
open(f'{TAB}/tab_step3_d6.tex','w').write(t7)
h=write_macros(f'{OUT}/numbers-d6tab.tex','% numbers-d6tab.tex: count fixes for draft 6. Written by bdc/scripts/draft6/d6_tables.py.\n')
write_log(f'{OUT}/numbers-d6tab-log.txt',f'd6_tables.py run {time.strftime("%Y-%m-%d %H:%M UTC",time.gmtime())}, {time.time()-t0:.0f} s, sha256 {h[:16]}',LOG)
print('done',h[:16])
