# Draft 4 (23 Nov 2026): macros for the label checks -> <out>/numbers-labels.tex.
# Sources (read, not recomputed, except where marked COMPUTED):
#   industry_llm/validation/results_label_validation_marty_final_2026-11-21.txt   blind validation, coder 1 (the author), 20 weak rows recoded blind
#   industry_llm/validation/results_open_model_fullrun_on_validation200_2026-11-23.txt   open-model full run scored against the author's codes
#   industry_llm/open-model-2026-11-21/results_open_model_stage2_2026-11-23.txt  the four tests of open-model-relabel-rule-2026-11-21.md
#   industry_llm/open-model-2026-11-21/out_gpt-oss_20b.csv (+ the pilot file)   COMPUTED: code counts and full-run vs pilot agreement
# The scoring scripts are not rerun here because they write dated result files into the shared folders; the sha256 of every
# source file is written to the log so a later rerun can be compared.
# Usage: python3 d4_labels_numbers.py <draft4 folder>
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..','..')))  # repo root
import sys, re, time
sys.path.insert(0,(_ROOT+'/bdc/scripts/draft4'))
from d4_common import *
OUT=sys.argv[1]; LOG=[]; t0=time.time()
def log(s): LOG.append(str(s)); print(s,flush=True)
I=(_ROOT+'/bdc/industry_llm')
FV=f'{I}/validation/results_label_validation_marty_final_2026-11-21.txt'; FO=f'{I}/validation/results_open_model_fullrun_on_validation200_2026-11-23.txt'
FS=f'{I}/open-model-2026-11-21/results_open_model_stage2_2026-11-23.txt'; FC=f'{I}/open-model-2026-11-21/out_gpt-oss_20b.csv'; FP=f'{I}/open-model-2026-11-21/out_gpt-oss_20b_run_input_pilot200.csv'
for f in [FV,FO,FS,FC,FP]: log(f'source {f} sha256 {sha(f)}')
def g(pat,txt,i=1,cast=float):
    m=re.search(pat,txt,re.S); assert m, pat; return cast(m.group(i))
v=open(FV).read()
put4('ValNCoded',g(r'Coded by coder 1: (\d+) of (\d+)',v,1,int),'int','validation borrowers coded by coder 1'); put4('ValNDrawn',g(r'Coded by coder 1: (\d+) of (\d+)',v,2,int),'int','borrowers drawn')
for k,K in [('precision','Prec'),('recall','Rec'),('false_omission','Fo')]:
    m=re.search(rf'{k}: ([0-9.]+) \[([0-9.]+), ([0-9.]+)\]',v); put4(f'Val{K}',float(m.group(1)),'f2',f'{k}, design-weighted, model vs coder 1'); put4(f'Val{K}Lo',float(m.group(2)),'f2','90% bootstrap'); put4(f'Val{K}Hi',float(m.group(3)),'f2','90% bootstrap')
put4('ValHumanShare',g(r'human_share: ([0-9.]+)',v),'pct0','design-weighted software share by coder 1'); put4('ValModelShare',g(r'model_share: ([0-9.]+)',v),'pct0','model software share')
put4('ValSameGroup',g(r'design-weighted ([0-9.]+)',v),'pct0','same 26-group code as the model, design-weighted')
put4('ValLeakFell',g(r'fell 3\+ points: model says software where coder does not ([0-9.]+)',v),'pct1','fallers: model software, coder not')
put4('ValLeakOther',g(r'other borrowers: model says software where coder does not ([0-9.]+)',v),'pct1','other borrowers: model software, coder not')
put4('ValMissFell',g(r'fell 3\+ points: .*?coder says software where model does not ([0-9.]+)',v),'pct1','fallers: coder software, model not')
put4('ValMissOther',g(r'other borrowers: .*?coder says software where model does not ([0-9.]+)',v),'pct1','others: coder software, model not')
for q,Q in [('2026-03','Mar'),('2026-06','Jun')]:
    for y,Y,d in [('mark','Mark',2),('Below99','Below',1)]:
        m=re.search(rf'{q} {y}: model labels ([-+0-9.]+) \(SE ([0-9.]+)\); DSL-corrected ([-+0-9.]+) \(SE ([0-9.]+)\)',v)
        putse4(f'ValDsl{Y}{Q}',float(m.group(3)),float(m.group(4)),f'DSL-corrected Step 1, {q} {y}, borrower bootstrap (validation script sample)',d=d)
        putse4(f'ValMod{Y}{Q}',float(m.group(1)),float(m.group(2)),f'model labels on the same DSL sample, {q} {y}',d=d)
o=open(FO).read()
blk=o[o.index('open model vs the author'):o.index('Claude, same rows')]
for k,K in [('precision','Prec'),('recall','Rec'),('false_omission','Fo')]:
    m=re.search(rf'{k}: ([0-9.]+) \[([0-9.]+), ([0-9.]+)\]',blk); put4(f'Om{K}',float(m.group(1)),'f2',f'open model (full run) vs coder 1, {k}'); put4(f'Om{K}Lo',float(m.group(2)),'f2','90%'); put4(f'Om{K}Hi',float(m.group(3)),'f2','90%')
put4('OmValN',g(r'open model vs the author \(n (\d+)',o,1,int),'int','validation borrowers scored for the open model')
s=open(FS).read()
put4('OmItems',g(r'Items (\d+), open model parsed (\d+) \(missing (\d+)\)',s,1,int),'int','borrower names given to the open model'); put4('OmParsed',g(r'Items (\d+), open model parsed (\d+)',s,2,int),'int','parsed')
put4('OmMissing',g(r'\(missing (\d+)\)',s,1,int),'int','not parsed'); put4('OmMissShare',g(r'\(missing (\d+)\)',s,1,int)/g(r'Items (\d+)',s,1,int),'pct1','share not parsed')
put4('OmNFallers',g(r'Fallers among items: (\d+)',s,1,int),'int','borrowers whose positions fell 3+ points')
m=re.search(r'Agreement, known to both \(n (\d+)\)\s+same software flag ([0-9.]+); Cohen kappa ([0-9.]+); same 26-group code ([0-9.]+)',s)
put4('OmNBoth',int(m.group(1)),'int','borrowers known to both models'); put4('OmAgreeFlag',float(m.group(2)),'pct0','same software flag'); put4('OmKappa',float(m.group(3)),'f2','Cohen kappa'); put4('OmSameGroup',float(m.group(4)),'pct0','same 26-group code')
m=re.search(r'software share: Claude ([0-9.]+), open model ([0-9.]+)',s); put4('OmSwShareC',float(m.group(1)),'pct0','software share known to both, Claude'); put4('OmSwShareO',float(m.group(2)),'pct0','open model')
m=re.search(r'fallers ([0-9.]+) \(n (\d+)\), others ([0-9.]+) \(n (\d+)\)\s+difference ([-+0-9.]+) 90% CI \[([-+0-9.]+), ([-+0-9.]+)\] -> leakage indicated: (\w+)',s)
put4('OmDFell',float(m.group(1)),'pct1','test 2: Claude SOFT & open not, fallers'); put4('OmDOther',float(m.group(3)),'pct1','others'); put4('OmNFellBoth',int(m.group(2)),'int','fallers known to both')
put4('OmTwoDiff',100*float(m.group(5)),'f1','test 2 difference',sign=True); put4('OmTwoLo',100*float(m.group(6)),'f1','90% CI',sign=True); put4('OmTwoHi',100*float(m.group(7)),'f1','90% CI',sign=True); assert m.group(8)=='True'
m=re.search(r'Reverse .*?difference ([-+0-9.]+) \[([-+0-9.]+), ([-+0-9.]+)\]\s+DiD ([-+0-9.]+) 90% CI \[([-+0-9.]+), ([-+0-9.]+)\] -> leakage indicated: (\w+)',s,re.S)
put4('OmRevDiff',100*float(m.group(1)),'f1','test 3 reverse difference',sign=True); put4('OmRevLo',100*float(m.group(2)),'f1','90%',sign=True); put4('OmRevHi',100*float(m.group(3)),'f1','90%',sign=True)
put4('OmDid',100*float(m.group(4)),'f1','test 3 DiD',sign=True); put4('OmDidLo',100*float(m.group(5)),'f1','90%',sign=True); put4('OmDidHi',100*float(m.group(6)),'f1','90%',sign=True); assert m.group(7)=='False'
m=re.search(r'software share fallers/others: Claude ([0-9.]+)/([0-9.]+), open ([0-9.]+)/([0-9.]+)',s)
put4('OmSwFellC',float(m.group(1)),'pct0','software share among fallers, Claude (post hoc)'); put4('OmSwOtherC',float(m.group(2)),'pct0','among others')
def es_line(tag,lab):
    m=re.search(re.escape(lab)+r'.*?: pos (\d+) \(sw (\d+)\), borrowers (\d+) \| mark Mar ([-+0-9.]+) \(([0-9.]+)\) Jun ([-+0-9.]+) \(([0-9.]+)\) \| b99 Mar ([-+0-9.]+) \(([0-9.]+)\) Jun ([-+0-9.]+) \(([0-9.]+)\)',s)
    putse4(f'Om{tag}Mark',float(m.group(6)),float(m.group(7)),f'Step 1 June mark, {lab}'); putse4(f'Om{tag}Below',float(m.group(10)),float(m.group(11)),f'Step 1 June Below99, {lab}',d=1); put4(f'Om{tag}N',int(m.group(1)),'int',lab+', positions')
es_line('Flag','(a) same cohort, open-model flag'); es_line('BothOpen','(b) known to both, open-model flag'); es_line('BothClaude','(b1) known to both, Claude flag')
es_line('Agree','EXTRA known to both and models agree'); es_line('Own','EXTRA open model own cohort')
assert re.search(r'Reading rule \(a\) and \(b\) both hold: True',s)
m=re.search(r'both: June mark ([-+0-9.]+) \(([0-9.]+)\), b99 ([-+0-9.]+).*?conly: June mark ([-+0-9.]+) \(([0-9.]+)\), b99 ([-+0-9.]+).*?oonly: June mark ([-+0-9.]+) \(([0-9.]+)\), b99 ([-+0-9.]+) \(([0-9.]+)\)',s,re.S)
put4('OmSplitBothMark',float(m.group(1)),'f2','both say software, June mark vs both-not (post hoc)',sign=True); put4('OmSplitCMark',float(m.group(4)),'f2','Claude only',sign=True); put4('OmSplitOMark',float(m.group(7)),'f2','open only',sign=True)
put4('OmSplitBothBelow',float(m.group(3)),'f1','both, Below99',sign=True); put4('OmSplitCBelow',float(m.group(6)),'f1','Claude only, Below99',sign=True); put4('OmSplitOBelow',float(m.group(9)),'f1','open only, Below99',sign=True)
m=re.search(r'EXTRA P\(open not SOFT \| Claude SOFT\): fallers ([0-9.]+), others ([0-9.]+), difference ([-+0-9.]+) 90% CI \[([-+0-9.]+), ([-+0-9.]+)\]',s)
put4('OmCondFell',float(m.group(1)),'pct0','P(open not SOFT | Claude SOFT), fallers (post hoc)'); put4('OmCondOther',float(m.group(2)),'pct0','others')
put4('OmCondDiff',100*float(m.group(3)),'f1','difference',sign=True); put4('OmCondLo',100*float(m.group(4)),'f1','90%',sign=True); put4('OmCondHi',100*float(m.group(5)),'f1','90%',sign=True)
# COMPUTED from the open-model output
om=pd.read_csv(FC,dtype=str).drop_duplicates('item'); gg=om.group.str.strip().str.upper(); L=pd.read_parquet(f'{CM.F}/labels.parquet')
put4('OmNFund',int((gg=='FUND').sum()),'int','open model FUND codes, full run'); put4('OmNUnkn',int((gg=='UNKN').sum()),'int','open model UNKN codes')
put4('ClNFund',int((L.group=='FUND').sum()),'int','Claude FUND codes'); put4('ClNUnkn',int((L.group=='UNKN').sum()),'int','Claude UNKN codes')
pl=pd.read_csv(FP,dtype=str).drop_duplicates('item'); mm=pl.merge(om,on='item',suffixes=('_p','_f')); a=mm.group_p.str.strip().str.upper(); b=mm.group_f.str.strip().str.upper()
put4('OmPilotN',len(mm),'int','validation borrowers in both the pilot and the full run'); put4('OmPilotSameGroup',(a==b).mean(),'pct0','full run vs pilot, same group (same model, temperature 0)')
put4('OmPilotSameFlag',((a=='SOFT')==(b=='SOFT')).mean(),'pct0','full run vs pilot, same software flag')
log(f'[computed] open FUND {int((gg=="FUND").sum())} UNKN {int((gg=="UNKN").sum())}; Claude FUND {int((L.group=="FUND").sum())} UNKN {int((L.group=="UNKN").sum())}; pilot vs full n {len(mm)} same group {(a==b).mean():.3f} same flag {((a=="SOFT")==(b=="SOFT")).mean():.3f}')
h=write_macros(f'{OUT}/numbers-labels.tex','% numbers-labels.tex: blind label validation and open-model relabel. Written by bdc/scripts/draft4/d4_labels_numbers.py.\n')
write_log(f'{OUT}/numbers-labels-log.txt',f'd4_labels_numbers.py run {time.strftime("%Y-%m-%d %H:%M UTC",time.gmtime())}, {time.time()-t0:.0f} s, numbers-labels.tex sha256 {h[:16]}',LOG)
print('numbers-labels.tex sha256',h)
