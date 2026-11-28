# Draft 6 (26 Nov 2026): derived numbers for the text -> <out>/numbers-d6der.tex
#  (1) word forms of small counts (#48: one to nine in running text): for every integer macro 0..9 in the macro files listed
#      below, \<name>W gives the word ('one' ...). (2) ratios of reported estimates (#115, #131): quoted software marks against
#      the same loans' prices, and private software marks against the traded price, both relative to other loans, June 2026,
#      computed from the unrounded estimates in numbers-log.txt where available, else from the macro values. (3) #137 scenario:
#      share of cohort principal in private software loans (Step 3 class file) and the NAV effect per 100 of assets if those
#      loans had lost as much as the traded ones relative to their own class (traded minus private markdown), with the NAV effect
#      per 100 of equity at a debt-to-equity ratio of one (the 1:1 floor before 2018; BDCs may now run 2:1). A labelled scenario.
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..','..')))  # repo root
import sys, re
sys.path.insert(0,(_ROOT+'/bdc/scripts/draft4')); sys.path.insert(0,(_ROOT+'/bdc/scripts/draft2'))
from d4_common import *
OUT=sys.argv[1]
W=['zero','one','two','three','four','five','six','seven','eight','nine']
vals={}
for f in ['numbers.tex','numbers-pap.tex','numbers-sep.tex','numbers-labels.tex','numbers-d5.tex','numbers-d6.tex','numbers-d6tab.tex']:
    for k,v in re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}',open(f'{OUT}/{f}').read()): vals[k]=v
for k,v in sorted(vals.items()):
    if re.fullmatch(r'[0-9]',v) and k+'W' not in vals: put4(k+'W',W[int(v)],'str',f'word form of \\{k}')
num=lambda k: float(re.sub(r'[^0-9.+-]','',vals[k].replace('$','')))
put4('DsixRatioQuoted',num('ThMarkJun')/num('ThPxJun'),'pct0','Step 2: software mark change / software price change, same loans, relative to other traded loans, June 2026 (ratio of the rounded table values)')
put4('DsixRatioPrivate',num('PrivOnlyJun')/num('ThPxJun'),'pct0','private software markdown relative to other private loans / traded software price fall (ratio of rounded table values; different loans)')
p=CM.load(); C=pd.read_parquet(f'{CM.V}/step3_classes_2026-11-17.parquet'); c9=p[(p.ddate==CM.BASE)&p.pkey.isin(CM.cohort_keys(p))]
pr=c9.set_index('pkey').principal; C=C[C.pkey.isin(pr.index)]; C['prin']=C.pkey.map(pr)
sh=C[(C.cls=='private')&(C.sw==1)].prin.sum()/pr.sum(); put4('DsixPrivSwPrinShare',sh,'pct1','share of cohort principal in private software positions, 30 Sep 2025')
gap=num('TrOnlyJun')-num('PrivOnlyJun'); put4('DsixNavScenario',-sh*gap,'f2','scenario: NAV effect per 100 of assets if private software loans had lost the traded-minus-private markdown (points)')
put4('DsixNavScenarioEq',-2*sh*gap,'f2','same per 100 of equity at debt-to-equity of one')
# (4) unsigned forms of the equity moves of numbers-d6fig.tex for running text (#65: 'falling 24%', not 'moving -24%')
for k,v in re.findall(r'\\newcommand\{\\(DsixEq[A-Za-z]+)\}\{(.*)\}',open(f'{OUT}/numbers-d6fig.tex').read()):
    if '$' in v: put4(k+'Abs',re.sub(r'\$[-+]?([0-9.]+)\$',r'\1',v),'str',f'unsigned form of \\{k}')
# (5) #4: no sign after 'rose', 'fell', 'recovered': an unsigned twin \<name>Un for every signed macro ($+x$ or $-x$)
for k,v in sorted(vals.items()):
    m=re.fullmatch(r'\$[-+]([0-9.,{}]+)\$',v)
    if m and k+'Un' not in vals: put4(k+'Un',m.group(1),'str',f'unsigned form of \\{k}')
h=write_macros(f'{OUT}/numbers-d6der.tex','% numbers-d6der.tex: word forms and derived ratios. Written by bdc/scripts/draft6/d6_derived.py.\n')
print(h[:16],{k:MAC4[k][0] for k in MAC4 if not k.endswith('W')}, len([k for k in MAC4 if k.endswith('W')]),'word macros')
