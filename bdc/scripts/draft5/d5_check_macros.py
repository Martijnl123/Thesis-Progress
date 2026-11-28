# Draft 5 (24 Nov 2026): copy of draft4/d4_check_macros.py that also requires numbers-d5.tex. Draft 4 header: macro check. Fails (exit 1) if
#  - a macro is defined in more than one numbers*.tex file, or twice in one;
#  - the text or the tables use a capitalised macro that no numbers*.tex file defines and that is not a LaTeX command;
#  - a macro used only inside the \ifnport ... \else branch is not one d4_sep_numbers.py writes when the N-PORT leg is present.
# It also lists the numbers.tex macros the draft 4 text no longer uses (for numbers-diff.md).
# Usage: python3 d5_check_macros.py <draft5 folder>
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..','..')))  # repo root
import sys, re, glob, os
D=sys.argv[1]; bad=[]
defs={}
for f in sorted(glob.glob(f'{D}/numbers*.tex')):
    for m in re.finditer(r'\\newcommand\{\\([A-Za-z]+)\}',open(f).read()):
        defs.setdefault(m.group(1),[]).append(os.path.basename(f))
FILES={os.path.basename(f) for f in glob.glob(f'{D}/numbers*.tex')}
need={'numbers.tex','numbers-pap.tex','numbers-sep.tex','numbers-labels.tex','numbers-tab.tex','numbers-d5.tex'}
if need-FILES: bad.append(f'missing macro files: {sorted(need-FILES)}')
dups={k:v for k,v in defs.items() if len(v)>1}
if dups: bad.append(f'duplicate definitions: {dups}')
SEP=open((_ROOT+'/bdc/scripts/draft4/d4_sep_numbers.py')).read()
nport_names=set(re.findall(r"put(?:se)?4\('((?:SepCTwo|CTwoCase|HTwoVerdict|SepNport)[A-Za-z]*)'",SEP))
nport_names|={n+'SE' for n in nport_names if n=='SepCTwo'}
# written by d4_sep_numbers.py only once the September N-PORT code exists; that script stops if N-PORT prices appear before then (D4-9)
PENDING={'SepNportCover','SepQPx','SepQPxSE','QPxCase'}; nport_names|=PENDING
texs=[f for f in glob.glob(f'{D}/*.tex') if not os.path.basename(f).startswith('numbers')]+glob.glob(f'{D}/tables/*.tex')
LATEX=set('''LaTeX TeX FloatBarrier S P L H O AA AE OE Huge LARGE Large Delta Gamma Lambda Omega Phi Pi Psi Sigma Theta Upsilon Xi'''.split())
used=set(); used_true=set()
def strip_comments(s): return re.sub(r'(?<!\\)%.*','',s)
for f in texs:
    s=strip_comments(open(f).read())
    # \ifnport TRUE \else FALSE \fi : collect macros in the true branch separately
    for m in re.finditer(r'\\ifnport(.*?)\\else(.*?)\\fi(?:%|\b|\s)',s,re.S):
        used_true|=set(re.findall(r'\\([A-Z][A-Za-z]+)',m.group(1)))
    s2=re.sub(r'\\ifnport(.*?)\\else',r'\\ifnport\\else',s,flags=re.S)
    used|=set(re.findall(r'\\([A-Z][A-Za-z]+)',s2))
undef=sorted(u for u in used if u not in defs and u not in LATEX)
if undef: bad.append(f'undefined macros: {undef}')
t_only=used_true-used
t_bad=sorted(u for u in t_only if u not in nport_names and u not in defs and u not in LATEX)
if t_bad: bad.append(f'macros in the \\ifnport true branch that d4_sep_numbers.py does not write: {t_bad}')
# words that stand for a macro value (examiner N10: "none" instead of "0.0%")
n3txt=open(f'{D}/numbers.tex').read()
if 'against none of the private ones' in open(f'{D}/results.tex').read() and '\\newcommand{\\PrSevenShare}{0.0\\%}' not in n3txt:
    bad.append('results.tex says "none of the private ones" but PrSevenShare is not 0.0%')
n3=set(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}',open(f'{D}/numbers.tex').read()))
print(f'[macros] defined {len(defs)} in {len(set(x for v in defs.values() for x in v))} files; used {len(used)} outside the N-PORT branch; '
      f'{len(t_only)} only in it ({", ".join(sorted(t_only))}); numbers.tex macros unused in draft 5: {len(n3-used-used_true)}')
open(f'{D}/macros-unused.txt','w').write('\n'.join(sorted(n3-used-used_true))+'\n')
if bad: print('[macros] FAIL'); [print('  '+b) for b in bad]; sys.exit(1)
print('[macros] OK' + ('; pending for \\nporttrue: '+', '.join(sorted(PENDING&t_only)) if PENDING&t_only else ''))
