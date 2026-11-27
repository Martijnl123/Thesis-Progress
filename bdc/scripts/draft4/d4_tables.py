# Draft 4 (23 Nov 2026): tables for draft 4 and numbers-tab.tex. Run after v3_numbers.py and the three d4_*_numbers.py scripts.
# Usage: python3 d4_tables.py <draft4 folder>
#  - Splits v3_numbers.py's 40-row tables/tab_robust.tex (draft 3 Table 5, examiner R3) into a main table (tab_robust_main.tex)
#    and an appendix table (tab_robust_more.tex). Rows are copied unchanged, so every number still comes from v3_numbers.py.
#  - Writes the new tables (PAP-listed checks, September 2026 test, its robustness, label checks) as LaTeX that holds only
#    macros from numbers.tex, numbers-pap.tex, numbers-sep.tex and numbers-labels.tex: no number is typed here.
#  - Writes numbers-tab.tex: var(G) ratio per quarter, read from the var(G) block of numbers-log.txt that v3_numbers.py writes
#    (examiner N4: the draft 3 range 0.7 to 4.6 mixed the pre-fall quarter with the two quarters of the fall).
import sys, os, re, hashlib
OUT=sys.argv[1]; T=f'{OUT}/tables'
src=open(f'{T}/tab_robust.tex').read().split('\n')
# ---------------------------------------------------------------- split Table 5
MAIN_A=['Headline (model labels)',"Funds' own labels, same cohort",'Both labels agree','Model software, fund not','Fund software, model not',
        '\\quad difference of the two groups','Leave one of the 25 largest managers out','Loans that left filled in, 10th percentile',
        'Placebo cohort September 2023','Placebo cohort September 2024','Real cohort under the placebo rules','Common shock: software change',
        '\\quad actual software change in March 2026']
MAIN_B=[]  # panel B (Step 2) moved to the appendix table to save main-text words
def find(prefix):
    hit=[l for l in src if l.startswith(prefix)]; assert len(hit)==1,(prefix,hit); return hit[0]
iA=src.index([l for l in src if l.startswith('\\multicolumn{3}{l}{\\emph{A.')][0]); iB=src.index([l for l in src if l.startswith('\\multicolumn{3}{l}{\\emph{B.')][0])
iC=src.index([l for l in src if l.startswith('\\multicolumn{3}{l}{\\emph{C.')][0]); iEnd=src.index('\\bottomrule')
rowsA=[l for l in src[iA+1:iB] if l and not l.startswith('\\midrule')]; rowsB=[l for l in src[iB+2:iC] if l and not l.startswith('\\midrule')]
rowsC=[l for l in src[iC+2:iEnd] if l]
headB=src[iB+1]; headC=src[iC+1]
mainA=[find(p) for p in MAIN_A]; mainB=[find(p) for p in MAIN_B]
moreA=[l for l in rowsA if l not in mainA]; moreB=[l for l in rowsB if l not in mainB]
assert len(rowsA)+len(rowsB)+len(rowsC)==len(mainA)+len(mainB)+len(moreA)+len(moreB)+len(rowsC)
top=src[:iA]  # tabular head, toprule, column header, midrule
main=top+[src[iA]]+mainA+(['\\midrule',src[iB],headB]+mainB if mainB else [])+['\\bottomrule','\\end{tabular}']
more=top+['\\multicolumn{3}{l}{\\emph{A. Step 1: other choices, June 2026}} \\\\']+moreA+\
     ['\\midrule',src[iB],headB]+moreB+['\\midrule',src[iC],headC]+rowsC+['\\bottomrule','\\end{tabular}']
open(f'{T}/tab_robust_main.tex','w').write('\n'.join(main)+'\n'); open(f'{T}/tab_robust_more.tex','w').write('\n'.join(more)+'\n')
nmain=len(mainA)+len(mainB); nmore=len(moreA)+len(moreB)+len(rowsC)
# ---------------------------------------------------------------- var(G) ratios from numbers-log.txt
lg=open(f'{OUT}/numbers-log.txt').read(); blk=lg[lg.index('[step2] var(G) by quarter'):].split('\n')[1:7]
vg={}
for l in blk:
    m=re.match(r'(\d{6})\s+([0-9.]+)\s+([0-9.]+)',l.strip())
    if m: vg[m.group(1)]=float(m.group(3))/float(m.group(2))
assert set(vg)=={'202512','202603','202606'}, vg
mac={'VarGRatioDec':f'{vg["202512"]:.1f}','VarGRatioMar':f'{vg["202603"]:.1f}','VarGRatioJun':f'{vg["202606"]:.1f}',
     'NRobustMainRows':str(nmain),'NRobustMoreRows':str(nmore)}
body='% numbers-tab.tex: written by bdc/scripts/draft4/d4_tables.py from numbers-log.txt (v3_numbers.py) and tables/tab_robust.tex.\n'+\
     ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k,v in sorted(mac.items()))
open(f'{OUT}/numbers-tab.tex','w').write(body)
# ---------------------------------------------------------------- new tables (macros only)
W={}
W['tab_pap']=r'''\begin{tabular}{L{9.0cm}cc}
\toprule
 & Mark & Below 99 \\
\midrule
\multicolumn{3}{l}{\emph{A. Robustness rows listed in the plan, June 2026 coefficient of Equation~\eqref{eq:h1}}} \\
Headline & \PapHeadMark{} (\PapHeadMarkSE) & \PapHeadBelow{} (\PapHeadBelowSE) \\
December 2025 dropped & \PapNoDecMark{} (\PapNoDecMarkSE) & \PapNoDecBelow{} (\PapNoDecBelowSE) \\
December year-end funds only (\PapDecFyeN{} positions) & \PapDecFyeMark{} (\PapDecFyeMarkSE) & \PapDecFyeBelow{} (\PapDecFyeBelowSE) \\
Descriptions of 255 characters or more dropped (\PapNLong{} positions) & \PapNoLongMark{} (\PapNoLongMarkSE) & \PapNoLongBelow{} (\PapNoLongBelowSE) \\
September 2025 mark bin $\times$ quarter effects & \PapMarkBinMark{} (\PapMarkBinMarkSE) & \PapMarkBinBelow{} (\PapMarkBinBelowSE) \\
PIK controls off & \PapNoPikMark{} (\PapNoPikMarkSE) & \PapNoPikBelow{} (\PapNoPikBelowSE) \\
\midrule
\multicolumn{3}{l}{\emph{B. Loans that leave}} \\
Software $\times$ June 2026 on leaving the cohort (pp), all leavers & \multicolumn{2}{c}{\PapExitJun{} (\PapExitJunSE)} \\
Worst case: leavers at the ends of the logical range (interval) & [\PapWorstMarkLo, \PapWorstMarkHi] & [\PapWorstBelowLo, \PapWorstBelowHi] \\
\midrule
\multicolumn{3}{l}{\emph{C. Software's rank and the alternative tests (September 2025 to June 2026)}} \\
Scaled rank among \NGroups{} groups (headline) & \RankMark{} & \RankBelow{} \\
Ratio of software's scaled statistic to the next group's & \PapRatioMark{} & \PapRatioBelow{} \\
Rank on the plain (unscaled) change & \PapRankPlainMark{} & \PapRankPlainBelow{} \\
Rank on the loan-clustered $t$-statistic & \PapRankTMark{} & \PapRankTBelow{} \\
Ferman--Pinto rescaled placebo test, one-sided $p$ & \PapFpMark{} & \PapFpBelow{} \\
Rank among \PapNTwentySix{} groups (IT services, data, hardware included) & \PapRankTwentySixMark{} & \PapRankTwentySixBelow{} \\
\quad rank of IT services among \PapNTwentySix{} & \PapRankItsvMark{} & \PapRankItsvBelow{} \\
\quad rank of data providers among \PapNTwentySix{} & \PapRankDataMark{} & \PapRankDataBelow{} \\
\quad rank of hardware among \PapNTwentySix{} & \PapRankHardMark{} & \PapRankHardBelow{} \\
Stricter comparison without education, media and marketing (\PapNStrict{} groups) & \PapRankStrictMark{} & \PapRankStrictBelow{} \\
\bottomrule
\end{tabular}
'''
W['tab_sep']=r'''\begin{tabular}{L{4.3cm}ccccL{2.7cm}}
\toprule
\multicolumn{6}{l}{\emph{A. C1 (mark) and C1b (Below 99): June to September 2026, software relative to other loans in the same fund}} \\
 & Funds & C1 & 90\% interval & C1b & Verdict (C1; C1b) \\
\midrule
Confirmatory funds (the estimate that counts) & \SepModelConfNFunds & \SepModelConfCOne{} (\SepModelConfCOneSE) & \SepModelConfCOneLo{} to \SepModelConfCOneHi & \SepModelConfCOneb{} (\SepModelConfCOnebSE) & \SepModelConfCOneV; \SepModelConfCOnebV \\
All funds & \SepModelAllNFunds & \SepModelAllCOne{} (\SepModelAllCOneSE) & \SepModelAllCOneLo{} to \SepModelAllCOneHi & \SepModelAllCOneb{} (\SepModelAllCOnebSE) & \SepModelAllCOneV; \SepModelAllCOnebV \\
Confirmatory, Blackstone added back & \SepModelBxNFunds & \SepModelBxCOne{} (\SepModelBxCOneSE) & \SepModelBxCOneLo{} to \SepModelBxCOneHi & \SepModelBxCOneb{} (\SepModelBxCOnebSE) & \SepModelBxCOneV; \SepModelBxCOnebV \\
Confirmatory, second early-filer list dropped & \SepModelNoBNFunds & \SepModelNoBCOne{} (\SepModelNoBCOneSE) & \SepModelNoBCOneLo{} to \SepModelNoBCOneHi & \SepModelNoBCOneb{} (\SepModelNoBCOnebSE) & \SepModelNoBCOneV; \SepModelNoBCOnebV \\
Confirmatory, funds' own labels & \SepFundConfNFunds & \SepFundConfCOne{} (\SepFundConfCOneSE) & \SepFundConfCOneLo{} to \SepFundConfCOneHi & \SepFundConfCOneb{} (\SepFundConfCOnebSE) & \SepFundConfCOneV; \SepFundConfCOnebV \\
Confirmatory, both labels agree & \SepAgreeConfNFunds & \SepAgreeConfCOne{} (\SepAgreeConfCOneSE) & \SepAgreeConfCOneLo{} to \SepAgreeConfCOneHi & \SepAgreeConfCOneb{} (\SepAgreeConfCOnebSE) & \SepAgreeConfCOneV; \SepAgreeConfCOnebV \\
\midrule
\multicolumn{6}{l}{\emph{B. Step 3: June to September 2026, software relative to other loans of the same class}} \\
 & Funds ($P$) & $Q$ (traded) & $P$ (private) & & Reading \\
\midrule
All funds of the June estimate filed by 20 November (counts) & \SThreeModelRegPFunds & \SThreeModelRegQ{} (\SThreeModelRegQSE) & \SThreeModelRegP{} (\SThreeModelRegPSE) & & \SThreeModelRegReading \\
Confirmatory funds & \SThreeModelConfPFunds & \SThreeModelConfQ{} (\SThreeModelConfQSE) & \SThreeModelConfP{} (\SThreeModelConfPSE) & & \SThreeModelConfReading \\
All funds, funds' own labels & & \SThreeFundRegQ{} (\SThreeFundRegQSE) & \SThreeFundRegP{} (\SThreeFundRegPSE) & & \SThreeFundRegReading \\
\bottomrule
\end{tabular}
'''
W['tab_sep_more']=r'''\begin{tabular}{L{5.0cm}L{2.5cm}cc}
\toprule
\multicolumn{4}{l}{\emph{A. C1 and C1b, other label and sample combinations (build A)}} \\
 & Funds & C1 & C1b \\
\midrule
Funds' own labels, all funds & \SepFundAllNFunds & \SepFundAllCOne{} (\SepFundAllCOneSE) & \SepFundAllCOneb{} (\SepFundAllCOnebSE) \\
Funds' own labels, Blackstone added back & \SepFundBxNFunds & \SepFundBxCOne{} (\SepFundBxCOneSE) & \SepFundBxCOneb{} (\SepFundBxCOnebSE) \\
Funds' own labels, second list dropped & \SepFundNoBNFunds & \SepFundNoBCOne{} (\SepFundNoBCOneSE) & \SepFundNoBCOneb{} (\SepFundNoBCOnebSE) \\
Both labels agree, all funds & \SepAgreeAllNFunds & \SepAgreeAllCOne{} (\SepAgreeAllCOneSE) & \SepAgreeAllCOneb{} (\SepAgreeAllCOnebSE) \\
Both labels agree, Blackstone added back & \SepAgreeBxNFunds & \SepAgreeBxCOne{} (\SepAgreeBxCOneSE) & \SepAgreeBxCOneb{} (\SepAgreeBxCOnebSE) \\
Both labels agree, second list dropped & \SepAgreeNoBNFunds & \SepAgreeNoBCOne{} (\SepAgreeNoBCOneSE) & \SepAgreeNoBCOneb{} (\SepAgreeNoBCOnebSE) \\
Build B, confirmatory funds & \SepBModelConfNFunds & \SepBModelConfCOne{} (\SepBModelConfCOneSE) & \SepBModelConfCOneb{} (\SepBModelConfCOnebSE) \\
Confirmatory, without the 5 software borrowers that rose most & & \SepCOneDropFive & \\
\quad without the 10 that rose most & & \SepCOneDropTen & \\
\quad without the 20 that rose most & & \SepCOneDropTwenty & \\
\midrule
\multicolumn{4}{l}{\emph{B. Step 3, other versions}} \\
 & Reading & $Q$ (traded) & $P$ (private) \\
\midrule
Second early-filer list dropped & \SThreeModelNoBReading & \SThreeModelNoBQ{} (\SThreeModelNoBQSE) & \SThreeModelNoBP{} (\SThreeModelNoBPSE) \\
Confirmatory funds, funds' own labels & \SThreeFundConfReading & \SThreeFundConfQ{} (\SThreeFundConfQSE) & \SThreeFundConfP{} (\SThreeFundConfPSE) \\
Second list dropped, funds' own labels & \SThreeFundNoBReading & \SThreeFundNoBQ{} (\SThreeFundNoBQSE) & \SThreeFundNoBP{} (\SThreeFundNoBPSE) \\
Build B, all funds filed by 20 November & \SThreeBModelRegReading & \SThreeBModelRegQ{} (\SThreeBModelRegQSE) & \SThreeBModelRegP{} (\SThreeBModelRegPSE) \\
\midrule
\multicolumn{4}{l}{\emph{C. Persistence checks, September 2025 to September 2026 (contain quarters already seen)}} \\
 & Funds & Mark & Below 99 \\
\midrule
$\beta_{\text{Sep26}}$, build A & \PersFunds & \PersMark{} (\PersMarkSE) & \PersBelow{} (\PersBelowSE) \\
\quad expected; surprising & & $-4.0$ to $-1.5$; above $-1.0$ & above $+20$; below $+10$ \\
\quad reading & & \PersMarkV & \PersBelowV \\
Rank among \PersNGroups{} groups, build A (expected 1st; surprising 4th or lower) & & \PersRankMark & \PersRankBelow \\
$\beta_{\text{Sep26}}$, build B & \PersBFunds & \PersBMark{} (\PersBMarkSE) & \PersBBelow{} (\PersBBelowSE) \\
Rank among \PersBNGroups{} groups, build B & & \PersBRankMark & \PersBRankBelow \\
\bottomrule
\end{tabular}
'''
W['tab_labels']=r'''\begin{tabular}{L{9.4cm}cc}
\toprule
\multicolumn{3}{l}{\emph{A. Blind sample: model labels against my own coding (\ValNCoded{} of \ValNDrawn{} borrowers, design-weighted)}} \\
Precision (90\% interval) & \multicolumn{2}{c}{\ValPrec{} [\ValPrecLo, \ValPrecHi]} \\
Recall & \multicolumn{2}{c}{\ValRec{} [\ValRecLo, \ValRecHi]} \\
False omission rate & \multicolumn{2}{c}{\ValFo{} [\ValFoLo, \ValFoHi]} \\
Same industry group & \multicolumn{2}{c}{\ValSameGroup} \\
Model says software, I do not: fallers / other borrowers & \multicolumn{2}{c}{\ValLeakFell{} / \ValLeakOther} \\
 & Mark & Below 99 \\
Step 1, June 2026, model labels (same sample) & \ValModMarkJun{} (\ValModMarkJunSE) & \ValModBelowJun{} (\ValModBelowJunSE) \\
Step 1, June 2026, corrected for label error (DSL) & \ValDslMarkJun{} (\ValDslMarkJunSE) & \ValDslBelowJun{} (\ValDslBelowJunSE) \\
\midrule
\multicolumn{3}{l}{\emph{B. Open-weights model (rule of 21 November 2026; \OmParsed{} of \OmItems{} borrowers parsed)}} \\
Agreement with the main labels, borrowers known to both ($n=\OmNBoth$): software flag; $\kappa$ & \multicolumn{2}{c}{\OmAgreeFlag; \OmKappa} \\
Main leakage test: main labels software, open model not, fallers minus others (pp) & \multicolumn{2}{c}{\OmTwoDiff{} [\OmTwoLo, \OmTwoHi]: indicated} \\
Reverse share, fallers minus others (pp) & \multicolumn{2}{c}{\OmRevDiff{} [\OmRevLo, \OmRevHi]} \\
Difference-in-differences (pp) & \multicolumn{2}{c}{\OmDid{} [\OmDidLo, \OmDidHi]: not indicated} \\
 & Mark & Below 99 \\
Step 1, June 2026, open-model flag on the cohort & \OmFlagMark{} (\OmFlagMarkSE) & \OmFlagBelow{} (\OmFlagBelowSE) \\
Step 1, June 2026, open-model flag, borrowers known to both & \OmBothOpenMark{} (\OmBothOpenMarkSE) & \OmBothOpenBelow{} (\OmBothOpenBelowSE) \\
\multicolumn{3}{l}{\emph{Added after seeing the results (post hoc)}} \\
Open model disagrees, given main label says software: fallers / others & \multicolumn{2}{c}{\OmCondFell{} / \OmCondOther{}; \OmCondDiff{} pp [\OmCondLo, \OmCondHi]} \\
Software share among fallers / others, main labels & \multicolumn{2}{c}{\OmSwFellC{} / \OmSwOtherC} \\
Open model against my coding: precision; recall & \multicolumn{2}{c}{\OmPrec; \OmRec} \\
Open model, investment-fund codes; unknown codes (main labels) & \multicolumn{2}{c}{\OmNFund{} (\ClNFund); \OmNUnkn{} (\ClNUnkn)} \\
Same group as the pilot run, \OmPilotN{} validation borrowers; same software flag & \multicolumn{2}{c}{\OmPilotSameGroup; \OmPilotSameFlag} \\
Step 1, June 2026, both models agree on the flag & \OmAgreeMark{} (\OmAgreeMarkSE) & \OmAgreeBelow{} (\OmAgreeBelowSE) \\
Step 1, June 2026, open model's own cohort & \OmOwnMark{} (\OmOwnMarkSE) & \OmOwnBelow{} (\OmOwnBelowSE) \\
\bottomrule
\end{tabular}
'''
for k,v in W.items(): open(f'{T}/{k}.tex','w').write(v)
print(f'[tables] tab_robust_main {nmain} rows, tab_robust_more {nmore} rows; var(G) ratios', {k:round(v,2) for k,v in vg.items()})
for f in ['numbers-tab.tex']+[f'tables/{k}.tex' for k in ['tab_robust_main','tab_robust_more']+list(W)]:
    print(f, hashlib.sha256(open(f'{OUT}/{f}','rb').read()).hexdigest())
