# Stage 1 of open-model-relabel-rule-2026-11-21.md: score the open model's labels on the 200 blind validation borrowers
# against the author's blind codes, next to the Claude labels, with the same design weights as v2_label_validation_score_2026-11-16.py.
#   python3 open_model_pilot_score_2026-11-21.py <out_<model>_run_input_pilot200.csv> <coder1_final.csv>
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..')))  # repo root
import sys, numpy as np, pandas as pd
F=(_ROOT+'/bdc/industry_llm/full'); O=(_ROOT+'/bdc/industry_llm/validation')
om=pd.read_csv(sys.argv[1],dtype=str); c1=pd.read_csv(sys.argv[2]).set_index('code_id')
K=pd.read_csv(f'{O}/key_DO_NOT_OPEN_BEFORE_CODING_2026-11-08.csv').set_index('code_id')
L=pd.read_parquet(f'{F}/labels.parquet'); L['item']=L['item'].astype(str)
om['group']=om.group.str.strip().str.upper(); om=om.drop_duplicates('item').merge(L[['item','bid']],on='item',how='left')
S=K.join(c1[['group_code','is_software_0_1']]).join(L.set_index('bid')[['group']],on='bid').join(om.set_index('bid')[['group','known']].rename(columns={'group':'og','known':'oknown'}),on='bid')
S['hsw']=pd.to_numeric(S.is_software_0_1,errors='coerce'); S.loc[S.group_code.astype(str).str.upper()=='UNKN','hsw']=np.nan
S['csw']=(S.group=='SOFT').astype(float); S['osw']=(S.og=='SOFT').astype(float); S.loc[S.og.isna(),'osw']=np.nan
out=[]
def say(s): out.append(s); print(s,flush=True)
say(f'Open-model pilot scored {pd.Timestamp.today():%d %b %Y %H:%M}. Model file: {sys.argv[1]}')
say(f'Open model labelled {int(S.og.notna().sum())} of {len(S)}; UNKN {int((S.og=="UNKN").sum())}, FUND {int((S.og=="FUND").sum())}, known=1 {int((S.oknown=="1").sum())}')
def wmean(x,w): return (x*w).sum()/w.sum() if w.sum()>0 else np.nan
def stats(C,m):
    ms,hs,w=C[m]==1,C.hsw==1,C.w
    return dict(precision=wmean(hs[ms].astype(float),w[ms]),recall=wmean(ms[hs].astype(float),w[hs]),false_omission=wmean(hs[~ms].astype(float),w[~ms]))
rng=np.random.default_rng(20261121)
for m,nm,extra in [('csw','Claude (26 Oct)',None),('osw','open model',None),('csw','Claude, same rows as open model known',True),('osw','open model, rows it knows (not UNKN/FUND)',True)]:
    C=S[S.hsw.notna()&S[m].notna()].copy()
    if extra: C=C[C.og.notna()&~C.og.isin(['UNKN','FUND'])]
    C['w']=C.N_stratum/C.groupby('stratum').hsw.transform('size')
    st=stats(C,m); B=pd.DataFrame([stats(pd.concat([g.iloc[rng.integers(0,len(g),len(g))] for _,g in C.groupby('stratum')]),m) for _ in range(1000)])
    say(f'\n{nm} vs the author (n {len(C)}, design-weighted, 90% bootstrap)')
    for k,v in st.items(): say(f'   {k}: {v:.3f} [{B[k].quantile(.05):.3f}, {B[k].quantile(.95):.3f}]')
    fell=C.stratum.str.endswith('fell')
    for lab,mm in [('fallers',fell),('others',~fell)]:
        x=C[mm]; say(f'   {lab}: says software where the hand coder does not {wmean(((x[m]==1)&(x.hsw==0)).astype(float),x.w):.3f}; misses software {wmean(((x[m]==0)&(x.hsw==1)).astype(float),x.w):.3f}')
J=S[S.og.notna()]; say(f'\nClaude vs open model software flag, all labelled pilot rows: agreement {(J.csw==J.osw).mean():.3f} (n {len(J)}); same group {(J.group==J.og).mean():.3f}')
open(f'{O}/results_open_model_pilot_{pd.Timestamp.today():%Y-%m-%d}.txt','w').write('\n'.join(out)+'\n')
