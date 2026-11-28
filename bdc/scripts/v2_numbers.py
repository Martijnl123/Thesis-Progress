# One source for every number in draft 2 (17 Nov 2026). Reads the final parquet files, runs every estimate quoted in the
# text, and writes <draft2>/numbers.tex (LaTeX macros), <draft2>/tables/*.tex, <draft2>/figures/*.pdf and <draft2>/numbers-log.txt.
# Data through 30 June 2026 only. The September 2026 hold-out stays locked: every loader asserts the last date.
# Sample building steps that must run first (they write parquet files this script reads):
#   v2_nport_match_all_2026-11-17.py   -> v2/nport_matches_all_2026-11-17.parquet  (N-PORT search on all positions, #68)
#   v2_step3_classes_2026-11-17.py      -> v2/step3_classes_2026-11-17.parquet     (tranche-level classes, #25)
# Usage: python3 v2_numbers.py <draft2 folder>
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..')))  # repo root
import sys, os, time, datetime, hashlib
sys.path.insert(0,(_ROOT+'/bdc/scripts/draft2'))
import d2_common as CM, d2_h1, d2_rank, d2_step2, d2_step3, d2_misc, d2_figs
from d2_common import put, putse, MAC, np, pd
OUT=sys.argv[1] if len(sys.argv)>1 else (_ROOT+'/outputs/draft2-2026-11-17')
TAB=f'{OUT}/tables'; FIG=f'{OUT}/figures'; os.makedirs(TAB,exist_ok=True); os.makedirs(FIG,exist_ok=True)
LOG=[]; t0=time.time()
def log(s): LOG.append(str(s)); print(s,flush=True)
def m(n): return MAC[n][0] if (n in MAC or os.environ.get('D2_DEV')!='1') else '\\textbf{??}'
def cell(b,s,d=2): return f'${b:+.{d}f}$ ({s:.{d}f})'

p=CM.load(); T={}
import traceback, pickle
DEV=os.environ.get('D2_DEV')=='1'; CACHE=os.environ.get('D2_CACHE')
def stage(name,fn,*a):
    """Run one block. In dev mode (D2_DEV=1) a failure is logged and the run goes on; the final run has no dev mode."""
    t1=time.time()
    try: r=fn(*a)
    except Exception:
        if not DEV: raise
        log(f'!!! STAGE FAILED: {name}\n'+traceback.format_exc()); r=None
    log(f'[time] {name} {time.time()-t1:.0f} s')
    if CACHE:
        with open(CACHE,'wb') as fh: pickle.dump({'T':{k:v for k,v in T.items() if k not in ('q2',)},'MAC':dict(MAC)},fh)
    return r
log(f'panel rows {len(p)}, last date {p.ddate.max().date()} (hold-out lock: must be 2026-06-30)'); assert p.ddate.max()==CM.JUN
res=stage('h1',d2_h1.run,p,T,log); d0=res[0] if res else None
stage('pyfixest',d2_h1.pyfixest_check,p,T,log)
stage('rank',d2_rank.rank,p,T,log); stage('fake',d2_rank.fake_cohorts,p,T,log); stage('bounds',d2_rank.bounds,p,T,log); stage('outside',d2_rank.outside,p,T,log)
stage('audit',d2_rank.late_audit,d0,T,log)
stage('step2',d2_step2.run,p,T,log); stage('step3',d2_step3.run,p,T,log)
stage('data',d2_misc.data_numbers,p,T,log); stage('labels',d2_misc.labels,T,log); stage('links',d2_misc.links,T,log); stage('exits',d2_misc.exits,T,log)
stage('registered',d2_misc.registered,T,log); stage('ext',d2_misc.ext,p,T,log); stage('recon',d2_misc.recon,p,T,log)
def conf_and_equity():
    # confirmatory sample share of principal (#42)
    EF=(_ROOT+'/raw-sec/edgar-pull/early-filers-2026-11-08/early_filers_2026-11-08.csv')
    ef=pd.read_csv(EF,dtype=str); A={int(x) for x in ef[ef.reason.str.startswith('A')].cik}|{1803498}
    pos=pd.read_parquet(f'{CM.V}/positions.parquet',columns=['cik','ddate','form']); fye=pos[pos.form.str.startswith('10-K',na=False)].groupby('cik').ddate.agg(lambda s:s.dt.month.mode().iloc[0])
    sep={int(c) for c in fye[fye==9].index}; c=T['cohort_c']; conf=~c.cik.astype(int).isin(A|sep)
    put('ConfPrinShare',c[conf].principal.sum()/c.principal.sum(),'pct1','cohort principal held by confirmatory funds'); put('NConfCohortFunds',c[conf].cik.nunique(),'int','confirmatory funds in the cohort')
    # equity series (figure 2)
    E=pd.read_csv(f'{CM.V}/equity_quarter_ends_2026-11-17.csv',parse_dates=['date']).set_index('date')
    for col,nm in [('software','Sw'),('broad','Br')]:
        s=E[col]; put(f'Eq{nm}SepMar',s[CM.MAR]/s[CM.BASE]-1,'pct0','equity change Sep25-Mar26',sign=True); put(f'Eq{nm}DecMar',s[CM.MAR]/s[CM.DEC]-1,'pct0','Dec25-Mar26',sign=True)
        put(f'Eq{nm}SepDec',s[CM.DEC]/s[CM.BASE]-1,'pct0','Sep-Dec 2025',sign=True); put(f'Eq{nm}MarJun',s[CM.JUN]/s[CM.MAR]-1,'pct0','Mar-Jun 2026',sign=True)
    q=T['q2']; w=q[q.px.notna()].groupby(['ddate','sw']).px.mean().unstack()
    put('LoanPxSwSepDec',w.loc[CM.DEC,1.0]-w.loc[CM.BASE,1.0],'f1','mean N-PORT price, traded software, Step 2 sample, Sep-Dec 2025',sign=True)
    put('LoanPxSwDecMar',w.loc[CM.MAR,1.0]-w.loc[CM.DEC,1.0],'f1','Dec-Mar',sign=True)
stage('conf_equity',conf_and_equity)
stage('figs',d2_figs.make,p,T,FIG,log)

def tables():
    # ---------------- tables ----------------
    def W(name,body): open(f'{TAB}/{name}.tex','w').write(body)
    T1=T['table1']; W('tab1_cohort','\\begin{tabular}{lrrr}\n\\toprule\n & Software & Other & All \\\\\n\\midrule\n'+''.join(f'{i} & '+' & '.join(str(v).replace(',','{,}') for v in r.values)+' \\\\\n' for i,r in T1.iterrows())+'\\bottomrule\n\\end{tabular}\n')
    # Step 1 main table
    H=T['h1']; rows=''
    for q_,Q in [('2025-12','December 2025'),('2026-03','March 2026'),('2026-06','June 2026')]:
        a=[CM.get(H[k],y,q_) for k in ['headline','fund labels on cohort','both agree'] for y in ['mark','b99']]
        rows+=f'{Q} & '+' & '.join(f'${b:+.2f}$' if i%2==0 else f'${b:+.1f}$' for i,(b,s) in enumerate(a))+' \\\\\n & '+' & '.join(f'({s:.2f})' if i%2==0 else f'({s:.1f})' for i,(b,s) in enumerate(a))+' \\\\\n'
    n=[H[k]['n'] for k in ['headline','fund labels on cohort','both agree']]
    W('tab_h1','\\begin{tabular}{lcccccc}\n\\toprule\n & \\multicolumn{2}{c}{Model labels} & \\multicolumn{2}{c}{Funds\' labels} & \\multicolumn{2}{c}{Both labels agree} \\\\\n & Mark & Below 99 & Mark & Below 99 & Mark & Below 99 \\\\\n\\midrule\n'+rows+
      '\\midrule\nPositions & '+' & '.join(f'\\multicolumn{{2}}{{c}}{{{x["pos"]:,}}}'.replace(',','{,}') for x in n)+' \\\\\nof which software & '+' & '.join(f'\\multicolumn{{2}}{{c}}{{{x["sw"]:,}}}'.replace(',','{,}') for x in n)+' \\\\\nFunds & '+' & '.join(f'\\multicolumn{{2}}{{c}}{{{x["funds"]}}}' for x in n)+' \\\\\n\\bottomrule\n\\end{tabular}\n')
    # robustness table (June 2026)
    def r1(lbl,k):
        r=H[k]; (bm,sm),(bb,sb)=CM.get(r,'mark','2026-06'),CM.get(r,'b99','2026-06'); return f'{lbl} & {cell(bm,sm)} & {cell(bb,sb,1)} \\\\\n'
    rd=H['disagree']
    A=('\\multicolumn{3}{l}{\\emph{A. Step 1: software minus other loans in the same fund, June 2026}} \\\\\n'+r1('Headline (model labels)','headline')+r1('Funds\' own labels, same cohort','fund labels on cohort')+r1('Both labels agree','both agree')
       +f'Model software, fund not (against both not software) & {cell(*CM.get(rd,"mark","2026-06","myfn"))} & {cell(*CM.get(rd,"b99","2026-06","myfn"),1)} \\\\\n'
       +f'Fund software, model not (against both not software) & {cell(*CM.get(rd,"mark","2026-06","mnfy"))} & {cell(*CM.get(rd,"b99","2026-06","mnfy"),1)} \\\\\n'
       +r1('Labels of the second model run where available','rerun labels')+r1('IT services, data and hardware left out','no adjacent')+r1('Weighted by principal','principal weighted')
       +r1('Manager $\\times$ quarter instead of fund $\\times$ quarter effects','manager x quarter')+r1('Standard errors by borrower and manager','two-way SE')
       +r1('Principal at least \\$1 million','1m floor')+r1('Without credit lines and delayed-draw loans','no revolvers')+r1('Principal did not grow (no interest added)','principal not grown')
       +r1('Positions still held in June 2026','held in June')+r1('Maturity reported','maturity reported')+r1('\\quad same, with time to maturity $\\times$ quarter','maturity reported, ttm control')+r1('Maturity not reported','maturity not reported')
       +f'Leave one of the 25 largest managers out (range) & [{m("LooMarkLo")}, {m("LooMarkHi")}] & [{m("LooBelowLo")}, {m("LooBelowHi")}] \\\\\n'
       +f'Loans that left filled in, 10th percentile mark (range) & [{m("ExitLowSwJun")}, {m("ExitLowOtJun")}] & [{m("ExitBLowOtJun")}, {m("ExitBLowSwJun")}] \\\\\n'
       +f'Placebo cohort September 2023, shock March 2024 (funds\' labels) & {cell(*T["fake"][("Three","mark")])} & {cell(*T["fake"][("Three","b99")],1)} \\\\\n'
       +f'Placebo cohort September 2024, shock March 2025 (funds\' labels) & {cell(*T["fake"][("Four","mark")])} & {cell(*T["fake"][("Four","b99")],1)} \\\\\n'
       +f'Real cohort under the placebo rules (funds\' labels) & {cell(*T["fake"][("Real","mark")])} & {cell(*T["fake"][("Real","b99")],1)} \\\\\n')
    B=('\\midrule\n\\multicolumn{3}{l}{\\emph{B. Step 2: gap between BDC mark and market price ($\\theta$, June 2026) and speed ($\\Delta\\lambda$, pooled)}} \\\\\n & $\\theta_{\\text{Jun26}}$ & $\\Delta\\lambda$ \\\\\n'
       +f'Headline (registered match, 10-point trim for $\\Delta\\lambda$) & {m("ThetaJun")} ({m("ThetaJunSE")}) & {m("DlPool")} ({m("DlPoolSE")}) \\\\\n'
       +f'No trim & & {m("DlPoolNotrim")} ({m("DlPoolNotrimSE")}) \\\\\n'
       +f'Fund $\\times$ quarter effects in the speed regression & & {m("DlPoolFq")} ({m("DlPoolFqSE")}) \\\\\n'
       +f'Lost quotes set to the last price & {m("ThetaJunStale")} ({m("ThetaJunStaleSE")}) & \\\\\n'
       +f'Lost quotes set to a low price & {m("ThetaJunLow")} ({m("ThetaJunLowSE")}) & \\\\\n'
       +f'Funds\' own labels & {m("ThetaJunFund")} ({m("ThetaJunFundSE")}) & {m("DlPoolFund")} ({m("DlPoolFundSE")}) \\\\\n'
       +f'N-PORT search on all positions & {m("ThetaJunAll")} ({m("ThetaJunAllSE")}) & {m("DlPoolAll")} ({m("DlPoolAllSE")}) \\\\\n')
    C=('\\midrule\n\\multicolumn{3}{l}{\\emph{C. Step 3: change in the mark to June 2026, software relative to other loans of the same class}} \\\\\n & Estimate & \\\\\n'
       +f'Private loans only (fund effects) & {m("PrivOnlyJun")} ({m("PrivOnlyJunSE")}) & \\\\\n'
       +f'Traded loans only (fund effects) & {m("TrOnlyJun")} ({m("TrOnlyJunSE")}) & \\\\\n'
       +f'Traded minus private (fund $\\times$ sector effects) & {m("GapJun")} ({m("GapJunSE")}) & \\\\\n'
       +f'\\quad 5\\% largest falls dropped in every class and sector & {m("GapTrim")} ({m("GapTrimSE")}) & \\\\\n'
       +f'\\quad changes winsorised at 5\\% and 95\\% & {m("GapWins")} ({m("GapWinsSE")}) & \\\\\n'
       +f'\\quad without the 20 traded software borrowers that fell most & {m("GapNoTwenty")} ({m("GapNoTwentySE")}) & \\\\\n'
       +f'\\quad funds\' own labels & {m("GapFund")} ({m("GapFundSE")}) & \\\\\n'
       +f'Private loans only, funds\' own labels & {m("PrivOnlyFund")} ({m("PrivOnlyFundSE")}) & \\\\\n')
    W('tab_robust','\\begin{tabular}{p{9.2cm}cc}\n\\toprule\n & Mark & Below 99 \\\\\n\\midrule\n'+A+B+C+'\\bottomrule\n\\end{tabular}\n')
    # Step 2 table
    th=T['theta']; rows=''
    for k,Q in [('202506','June 2025'),('202512','December 2025'),('202603','March 2026'),('202606','June 2026')]:
        rows+=f'{Q} & {cell(*th["mark"][k])} & {cell(*th["px"][k])} & {cell(*th["gap"][k])} \\\\\n'
    P=T['pa']; rowsb=''
    for t_,Q in [('202509','Jun to Sep 2025'),('202512','Sep to Dec 2025'),('202603','Dec 2025 to Mar 2026'),('202606','Mar to Jun 2026')]:
        a=P[(P.t==t_)&(P.v=='trim')].iloc[0]; b=P[(P.t==t_)&(P.v=='notrim')].iloc[0]; c_=P[(P.t==t_)&(P.v=='nosame')].iloc[0]
        rowsb+=f'{Q} & {a.lam:.2f} & {cell(a.dl,a.se)} & {cell(c_.dl,c_.se)} & {cell(b.dl,b.se)} & {int(a.nsw)} \\\\\n'
    rowsb+=f'Pooled, Sep 2025 to Jun 2026 & {MAC["LamPool"][1]:.2f} & {m("DlPool")} ({m("DlPoolSE")}) & & {m("DlPoolNotrim")} ({m("DlPoolNotrimSE")}) & {m("NDlPoolSw")} \\\\\n'
    W('tab_step2','\\begin{tabular}{lccccc}\n\\toprule\n\\multicolumn{6}{l}{\\emph{A. Software minus other loans, same fund, change since September 2025}} \\\\\n & BDC mark & Market price & Gap, $\\theta_k$ & & \\\\\n\\midrule\n'
      +rows.replace(' \\\\\n',' & & \\\\\n')+'\\midrule\n\\multicolumn{6}{l}{\\emph{B. Share of the distance to the new price closed in the quarter}} \\\\\n & $\\lambda$, other & $\\Delta\\lambda$ & $\\Delta\\lambda$, identical & $\\Delta\\lambda$, & Software \\\\\n & loans & (10-point trim) & pairs dropped & no trim & pairs \\\\\n\\midrule\n'
      +rowsb+'\\bottomrule\n\\end{tabular}\n')
    # Step 3 table
    def st(nm): return [m(f'{nm}SwTr'),m(f'{nm}SwPr'),m(f'{nm}OtTr'),m(f'{nm}OtPr')]
    W('tab_step3','\\begin{tabular}{lcccc}\n\\toprule\n & \\multicolumn{2}{c}{Software} & \\multicolumn{2}{c}{Other loans} \\\\\n & Traded & Private & Traded & Private \\\\\n\\midrule\n\\multicolumn{5}{l}{\\emph{A. 30 September 2025}} \\\\\n'
      +'Positions (borrowers) & '+' & '.join(f'{m(f"NCls{a}")} ({m(f"NCls{a}Borr")})' for a in ['SwTr','SwPr','OtTr','OtPr'])+' \\\\\n'
      +'Mean mark & '+' & '.join(st('Mark'))+' \\\\\nSpread (bp) & '+' & '.join(st('Spr'))+' \\\\\nPaying interest in kind (\\%) & '+' & '.join(st('Pik'))+' \\\\\nManagers holding the loan & '+' & '.join(st('Hold'))+' \\\\\n'
      +'\\midrule\n\\multicolumn{5}{l}{\\emph{B. Change in the mark since September 2025 (points), positions held at the date}} \\\\\n'
      +'To March 2026, mean & '+' & '.join([m(f'Chg{a}Mar') for a in ['SwTr','SwPr','OtTr','OtPr']])+' \\\\\n'
      +'To June 2026, mean & '+' & '.join(m(f'Chg{a}Jun') for a in ['SwTr','SwPr','OtTr','OtPr'])+' \\\\\nTo June 2026, median & '+' & '.join(m(f'Med{a}Jun') for a in ['SwTr','SwPr','OtTr','OtPr'])+' \\\\\n'
      +'Lost more than 5 points by June (\\%) & '+' & '.join(st('Five'))+' \\\\\nMark unchanged, December to March (\\%) & '+' & '.join(st('Unch'))+' \\\\\n'
      +'\\midrule\n\\multicolumn{5}{l}{\\emph{C. Software relative to other loans of the same class, fund effects, June 2026}} \\\\\n'
      +f'Estimate & \\multicolumn{{2}}{{c}}{{traded: {m("TrOnlyJun")} ({m("TrOnlyJunSE")})}} & \\multicolumn{{2}}{{c}}{{private: {m("PrivOnlyJun")} ({m("PrivOnlyJunSE")})}} \\\\\n'
      +f'Traded minus private, fund $\\times$ sector effects & \\multicolumn{{4}}{{c}}{{{m("GapJun")} ({m("GapJunSE")}); 95\\% interval {m("GapJunLo")} to {m("GapJunHi")}}} \\\\\n'+'\\bottomrule\n\\end{tabular}\n')
    # power table
    PW=T['power']; rows=''.join(f'{r.test} & {r.true:+.1f} & {100*r.supports:.0f} & {100*r.inconclusive:.0f} & {100*r.against:.0f} \\\\\n'.replace('+0.0','0.0') for r in PW.itertuples())
    W('tab_power','\\begin{tabular}{lcccc}\n\\toprule\nTest & True value & Supports (\\%) & Inconclusive (\\%) & Counts against (\\%) \\\\\n\\midrule\n'+rows+'\\bottomrule\n\\end{tabular}\n')
    # reconciliation table
    Rc=T['recon']; f=lambda v: '' if pd.isna(v) else f'{int(v):,}'.replace(',','{,}')
    W('tab_recon','\\begin{tabular}{p{7.2cm}rrrrr}\n\\toprule\nSample & Positions & Software & Borrowers & Funds & Managers \\\\\n\\midrule\n'+''.join(f'{r.sample}{" (loans)" if r.unit=="loans" else ""} & {f(r.positions)} & {f(r.software)} & {f(r.borrowers)} & {f(r.funds)} & {f(r.managers)} \\\\\n' for r in Rc.itertuples())+'\\bottomrule\n\\end{tabular}\n')
    # outside-cohort table (#31)
    O=T['outside']; W('tab_outside','\\begin{tabular}{lcccc}\n\\toprule\n & All & Small loans & Middle & Large loans \\\\\n\\midrule\n'+''.join(f'{"Outside the cohort" if g=="Out" else "Inside the cohort"} & '+' & '.join(cell(r.b,r.se) for r in O[O.grp==g].itertuples())+' \\\\\n' for g in ['Out','In'])+'\\bottomrule\n\\end{tabular}\n')
    # filters table (Appendix A)
    W('tab_filter','\\begin{tabular}{lr}\n\\toprule\nStep & Positions \\\\\n\\midrule\n'+f'Debt positions, December 2022 to June 2026 & {m("NRawPos")} \\\\\nCost between 90\\% and 105\\% of principal & {m("NFiltCost")} \\\\\nMark between 20 and 110 & {m("NFiltMark")} \\\\\nPrincipal at least \\$250{{,}}000 & {m("NFiltPrin")} \\\\\n\\midrule\nClean positions on 30 September 2025 & {m("NCleanSep")} \\\\\n\\quad borrower outside the labelling run & {m("NNoLabelRun")} \\\\\n\\quad labelled unknown & {m("NUnknown")} \\\\\n\\quad labelled as an investment fund & {m("NFundLab")} \\\\\nCohort & {m("NCohort")} \\\\\n'+'\\bottomrule\n\\end{tabular}\n')


stage('tables',tables)

# ---------------- write numbers.tex and the log ----------------
hdr=f'% numbers.tex: generated by thesis/2-data-and-code/bdc/scripts/v2_numbers.py. Do not edit by hand.\n% Data through 30 June 2026. {len(MAC)} macros.\n'
open(f'{OUT}/numbers.tex','w').write(hdr+''.join(f'\\newcommand{{\\{k}}}{{{v[0]}}}\n' for k,v in sorted(MAC.items())))
h=hashlib.sha256(open(f'{OUT}/numbers.tex','rb').read()).hexdigest()[:16]
open(f'{OUT}/numbers-log.txt','w').write(f'v2_numbers.py run {datetime.datetime.now():%Y-%m-%d %H:%M} UTC, {time.time()-t0:.0f} s, numbers.tex sha256 {h}\n\nMACROS (name | value | sample)\n'
    +''.join(f'{k} | {v[0]} | {v[2]}\n' for k,v in sorted(MAC.items()))+'\n\nRUN LOG\n'+'\n'.join(LOG)+'\n')
print(f'done: {len(MAC)} macros, numbers.tex sha256 {h}, {time.time()-t0:.0f} s')
