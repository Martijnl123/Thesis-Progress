# Run-to-run agreement of the industry labels (PAP v5 section 6), 8 Nov 2026.
# 500 borrowers drawn at random (seed 20261108) from the 5,686 of the 26 Oct run, relabelled with the same frozen prompt and
# the same "sonnet" model alias via Claude Code subagents (a pinned version and temperature cannot be set there).
# Measures reproducibility, not leakage (same model family).
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..')))  # repo root
import glob, pandas as pd, numpy as np
D=(_ROOT+'/bdc/industry_llm')
old=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{D}/full/batches/out_*.csv'))])
new=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{D}/rerun-2026-11-08/out_*.csv'))])
ins=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{D}/rerun-2026-11-08/in_*.csv'))])
assert set(new.item)==set(ins.item), (len(new),len(ins))
m=old.merge(new,on='item',suffixes=('_1','_2'))
def kappa(a,b):
    cats=sorted(set(a)|set(b)); po=(a==b).mean()
    pe=sum((a==c).mean()*(b==c).mean() for c in cats); return (po-pe)/(1-pe)
out=[f'borrowers compared: {len(m)}']
out.append(f'same group: {(m.group_1==m.group_2).mean():.3f} | Cohen kappa (groups) {kappa(m.group_1,m.group_2):.3f}')
out.append(f'same software flag: {(m.software_1==m.software_2).mean():.3f} | kappa (software) {kappa(m.software_1,m.software_2):.3f}')
u=m[(m.group_1!='UNKN')&(m.group_2!='UNKN')]
out.append(f'both runs placed the borrower (not UNKN): {len(u)}; same group among them {(u.group_1==u.group_2).mean():.3f}')
out.append(f'UNKN in run 1 {(m.group_1=="UNKN").mean():.3f}, run 2 {(m.group_2=="UNKN").mean():.3f}; UNKN in one run only {((m.group_1=="UNKN")!=(m.group_2=="UNKN")).mean():.3f}')
s=pd.crosstab(m.software_1,m.software_2); out.append('software flag, run 1 (rows) x run 2 (cols):\n'+s.to_string())
out.append('software in run 1: '+str(int(m.software_1.sum()))+' | in run 2: '+str(int(m.software_2.sum())))
ch=m[m.group_1!=m.group_2].groupby(['group_1','group_2']).size().sort_values(ascending=False).head(12)
out.append('most frequent group changes:\n'+ch.to_string())
m[m.group_1!=m.group_2][['item','operating_name_1','group_1','operating_name_2','group_2']].to_csv(f'{D}/rerun-2026-11-08/disagreements.csv',index=False)
open(f'{D}/rerun-2026-11-08/results_rerun_agreement_2026-11-08.txt','w').write('\n'.join(out)+'\n'); print('\n'.join(out))
