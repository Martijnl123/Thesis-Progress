# Blind label-validation sample (PAP v5 section 6), drawn 8 Nov 2026. Data to 30 Jun 2026 only.
# Universe: the 5,686 borrowers in the labelling run with a usable label (not UNKN/FUND).
# Strata: model label class (SOFT; adjacent = ITSV/DATA/HARD; other) x markdown (borrower's mean mark change
# 30 Sep 2025 -> 30 Jun 2026 over positions held at both dates <= -3 points: 'fell'; otherwise or not held at June: 'not').
# Output: coding_sheet.csv (random order, strata and model labels hidden), key_DO_NOT_OPEN_BEFORE_CODING.csv
# (stratum, design weight), and a summary. Coders: humans only, sources dated before 1 Jan 2026.
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..')))  # repo root
import numpy as np, pandas as pd
V=(_ROOT+'/bdc/v2'); F=(_ROOT+'/bdc/industry_llm/full')
O=(_ROOT+'/bdc/industry_llm/validation')
p=pd.read_parquet(f'{V}/panel.parquet',columns=['pos_id','pkey','ddate','mark','ok']); assert p.ddate.max()<=pd.Timestamp('2026-06-30')
L=pd.read_parquet(f'{F}/labels.parquet'); B=pd.read_parquet(f'{F}/borrowers.parquet')
pb=pd.read_parquet(f'{F}/position_borrower.parquet').merge(p[['pos_id','pkey']],on='pos_id').drop_duplicates('pkey')
p=p.merge(pb[['pkey','bid']].drop_duplicates('pkey'),on='pkey')
a=p[(p.ddate=='2025-09-30')&p.ok].set_index('pkey').mark; b=p[p.ddate=='2026-06-30'].groupby('pkey').mark.mean()
ch=(b-a.reindex(b.index)).dropna(); chb=ch.groupby(pb.set_index('pkey').bid.reindex(ch.index)).mean()
U=L[~L.group.isin(['UNKN','FUND'])].copy()
U['cls']=np.where(U.group=='SOFT','SOFT',np.where(U.group.isin(['ITSV','DATA','HARD']),'ADJ','OTHER'))
U['fell']=np.where(U.bid.map(chb).le(-3),'fell','not')
U['stratum']=U.cls+'_'+U.fell
alloc={'SOFT_fell':40,'SOFT_not':30,'ADJ_fell':20,'ADJ_not':15,'OTHER_fell':50,'OTHER_not':45}
rng=np.random.default_rng(20261108); parts=[]
for s,n in alloc.items():
    g=U[U.stratum==s]; k=min(n,len(g)); x=g.iloc[rng.choice(len(g),k,replace=False)].copy(); x['weight']=len(g)/k; x['N_stratum']=len(g); parts.append(x)
S=pd.concat(parts).sample(frac=1,random_state=20261108).reset_index(drop=True)
S['code_id']=[f'V{i+1:03d}' for i in range(len(S))]
S=S.merge(B[['bid','descs','trade_names']],on='bid',how='left')
sheet=S[['code_id','descs','trade_names']].rename(columns={'descs':'names_in_filings','trade_names':'other_names'})
for c in ['operating_company','group_code','is_software_0_1','source_url','source_date','confidence_1_3','notes']: sheet[c]=''
sheet.to_csv(f'{O}/coding_sheet_2026-11-08.csv',index=False)
S[['code_id','bid','stratum','weight','N_stratum']].to_csv(f'{O}/key_DO_NOT_OPEN_BEFORE_CODING_2026-11-08.csv',index=False)
t=U.groupby('stratum').size().rename('N').to_frame().join(S.groupby('stratum').size().rename('n'))
t.to_csv(f'{O}/strata_summary_2026-11-08.csv'); print(t.to_string()); print('sample',len(S),'| seed 20261108')
