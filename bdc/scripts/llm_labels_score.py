# Check and score the full LLM industry run (protocol: notes/llm-labels-protocol-2026-10-26.md).
# 1. Format check per batch. 2. Agreement with managers' own Sep 2025 labels (coarse sectors, software recall and precision).
# 3. Leakage test: LLM/manager disagreement on software for loans that later fell 3+ points vs the rest.
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..')))  # repo root
import sys, glob, os, numpy as np, pandas as pd
F=(_ROOT+'/bdc/industry_llm/full'); V=(_ROOT+'/bdc/v2')
G=['SOFT','ITSV','DATA','HARD','TELE','MEDI','MKTG','HCPS','HCEQ','PHAR','INSU','FINS','PROF','COMM','EDUC','CONS','CPRD','FOOD','CAPG','CONB',
   'TRAN','AUTO','CHEM','ENER','DIST','REAL','FUND','UNKN']
COARSE={'SOFT':'software','ITSV':'it_services','DATA':'business_services','HARD':'tech_other','TELE':'energy_utilities','MEDI':'media_marketing',
        'MKTG':'media_marketing','HCPS':'health','HCEQ':'health','PHAR':'health','INSU':'finance','FINS':'finance','REAL':'finance','PROF':'business_services',
        'COMM':'business_services','EDUC':'education','CONS':'consumer','CPRD':'consumer','FOOD':'consumer','CAPG':'industrials','CONB':'industrials',
        'TRAN':'industrials','AUTO':'industrials','CHEM':'industrials','DIST':'industrials','ENER':'energy_utilities'}
out=[]; bad=[]
for f in sorted(glob.glob(f'{F}/batches/in_*.csv')):
    o=f.replace('/in_','/out_')
    i=pd.read_csv(f)
    if not os.path.exists(o): bad.append((os.path.basename(o),'missing')); continue
    try: x=pd.read_csv(o)
    except Exception as e: bad.append((os.path.basename(o),f'unreadable: {e}')); continue
    cols=['item','operating_name','group','software','it_services','ai_exposure','known']
    if list(x.columns)!=cols: bad.append((os.path.basename(o),f'header {list(x.columns)}')); continue
    probs=[]
    if sorted(x.item)!=sorted(i.item): probs.append(f'items {len(x)} of {len(i)}')
    if not x.group.isin(G).all(): probs.append(f'bad groups {sorted(set(x.group)-set(G))}')
    if ((x.group=='SOFT').astype(int)!=x.software).any(): probs.append('software flag not equal to group')
    if not x.ai_exposure.isin([0,1,2,3]).all() or not x.known.isin([0,1]).all(): probs.append('bad field values')
    if probs: bad.append((os.path.basename(o),'; '.join(probs)))
    out.append(x[x.item.isin(i.item)].drop_duplicates('item'))
print('format problems:',bad if bad else 'none')
L=pd.concat(out) if out else pd.DataFrame()
if len(L)==0: sys.exit()
B=pd.read_parquet(f'{F}/borrowers.parquet')
L=L.merge(B[['item','bid','principal','mgr_sector','mgr_share','only_lab_fam']],on='item')
L['coarse']=L.group.map(COARSE)
print('labelled',len(L),'| groups:',L.group.value_counts().to_dict())
print('known',L.known.mean().round(3),'| UNKN',(L.group=='UNKN').mean().round(3),'| software share',L.software.mean().round(3),
      '(by principal',round((L.software*L.principal).sum()/L.principal.sum(),3),')')

def score(x,lab):
    x=x[x.mgr_sector.notna()&~x.mgr_sector.isin(['other'])&~x.group.isin(['UNKN','FUND'])]
    ms=x.mgr_sector.eq('software'); ls=x.software.eq(1)
    rec=ls[ms].mean(); fpr=ls[~ms&~x.mgr_sector.isin(['it_services','tech_other'])].mean(); prec=ms[ls].mean()
    agree=(x.coarse==x.mgr_sector).mean()
    print(f'  {lab:46s} n {len(x):5d} | software recall {rec:.2f} false-positive rate {fpr:.3f} precision {prec:.2f} | coarse-sector agreement {agree:.2f}')
print('\nagainst managers\' own labels (borrowers with a manager label; UNKN and FUND left out)')
score(L,'all')
score(L[~L.only_lab_fam],'input not only from label-printing managers')
score(L[L.known==1],'recognised by the model')
score(L[L.known==0],'not recognised')

# leakage test: change in mark Sep 2025 -> Jun 2026 per borrower (mean over its positions held at both dates)
p=pd.read_parquet(f'{V}/panel.parquet',columns=['pos_id','pkey','ddate','mark'])
pb=pd.read_parquet(f'{F}/position_borrower.parquet')
s0=p[p.ddate=='2025-09-30'].merge(pb,on='pos_id')[['pkey','bid','mark']]
s6=p[(p.ddate=='2026-06-30')&p.mark.between(0,110)][['pkey','mark']].rename(columns={'mark':'m6'})
ch=s0.merge(s6,on='pkey').assign(d=lambda z:z.m6-z.mark).groupby('bid').d.mean()
x=L[L.mgr_sector.notna()&~L.mgr_sector.isin(['other'])&~L.group.isin(['UNKN','FUND'])].copy(); x['d']=x.bid.map(ch); x=x[x.d.notna()]
x['fell']=x.d<=-3; ms=x.mgr_sector.eq('software'); ls=x.software.eq(1)
print('\nleakage test (borrowers with a manager label and marks at Sep 2025 and Jun 2026)')
for f,lab in [(True,'fell 3+ points'),(False,'did not')]:
    y=x[x.fell==f]; m=y.mgr_sector.eq('software'); l=y.software.eq(1)
    print(f'  {lab:15s} n {len(y):5d} | LLM software, manager not: {(l&~m).mean():.3f} | manager software, LLM not: {(m&~l).mean():.3f} | software recall {l[m].mean():.2f} | false-positive rate {l[~m].mean():.3f}')
L.to_parquet(f'{F}/labels.parquet',index=False)
