#!/usr/bin/env python3
# d5_wordcount.py <main.pdf> [<draft folder>]: word counts by the draft 4 method (fix-ledger-draft4.md).
# pdftotext -layout; main text from the last "1 Introduction" heading to the last "A Data" heading (references, tables,
# figure text and captions included, abstract excluded); appendices from "A Data" to the end. With a draft folder, writes numbers-wc.tex.
import sys, re, subprocess
pdf=sys.argv[1]; t=subprocess.run(['pdftotext','-layout',pdf,'-'],capture_output=True,text=True).stdout
lines=t.split('\n')
# skip the table of contents: take the LAST match of each heading
def last(pat):
    m=[i for i,l in enumerate(lines) if re.match(pat,l.strip())]; return m[-1]
H=[('Abstract',r'^Abstract$'),('Introduction',r'^1\s+Introduction$'),('Literature',r'^2\s+Literature$'),('Data',r'^3\s+Data$'),('Method',r'^4\s+Method$'),('Results',r'^5\s+Results$'),('Conclusion',r'^6\s+Conclusion$'),('References',r'^References$'),('Appendices',r'^A\s+Data$')]
pos={k:last(p) for k,p in H}
pos['Contents']=last(r'^Contents$')
def wc(a,b): return len(' '.join(lines[a:b]).split())
out={}
out['Abstract']=wc(pos['Abstract']+1,pos['Contents'])
keys=['Introduction','Literature','Data','Method','Results','Conclusion','References','Appendices']
for k,n in zip(keys,keys[1:]+[None]):
    out[k]=wc(pos[k],pos[n] if n else len(lines))
out['Main text']=sum(out[k] for k in keys[:-1])
for k,v in out.items(): print(f'{k:12s} {v:6,d}')
print(f'{"Total":12s} {out["Main text"]+out["Appendices"]+out["Abstract"]:6,d}')
if len(sys.argv)>2:
    import os
    with open(os.path.join(sys.argv[2],'numbers-wc.tex'),'w') as f:
        f.write('% numbers-wc.tex: written by draft5/d5_wordcount.py from main.pdf (pdftotext, draft 4 method). The count excludes the title page, so it is stable across rebuilds.\n')
        f.write('\\newcommand{\\WordCountMain}{%s}\n' % f"{out['Main text']:,d}".replace(',','{,}'))
