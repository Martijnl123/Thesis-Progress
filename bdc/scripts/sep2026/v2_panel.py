# Build the position panel for the AI-shock tests (v2: fixed filing choice, the re-posted Aug 2026 file, and the stable
# within-fund key from v2_stablekey.py as the position key; the old description key is kept as nkey).
# Sector labels come only from managers that print an industry in their descriptions (own label, or a co-holder's label on the same loan).
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..','..')))  # repo root
import re, numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
B=(_ROOT+'/bdc')
V=__import__('os').environ['OUTV']
IND=r'software|it services|information technology|internet|health ?care|pharma|biotech|life sciences|insurance|financial|capital markets|banks?\b|professional services|commercial (?:services|& professional)|business services|consumer|retail|media|entertainment|chemicals|construction|engineering|capital goods|building products|machinery|aerospace|transport|logistics|automobile|auto components|automotive|food|beverage|household|hotels|leisure|education|real estate|energy|utilit|telecom|equipment|distributors|containers|packaging|electrical|electronic|trading compan|metals|mining|oil|environmental|textiles|apparel|semiconductors|hardware|wholesale|paper|plastics|personal (?:care|products)|durables|marketing|advertising|publishing|broadcasting|industrial|technology|high tech'
SECT=[('software',r'software|saas|high tech'),
      ('it_services',r'it services|information technology|technology services|internet services|data processing'),
      ('tech_other',r'technology|internet|electronic|semiconductor|hardware|computer|communications equipment|digital'),
      ('media_marketing',r'media|publishing|broadcasting|entertainment|advertising|marketing'),
      ('business_services',r'professional services|commercial services|business services|services: business|commercial & professional|commercial and professional'),
      ('education',r'education|diversified consumer services'),
      ('health',r'health|pharma|biotech|life sciences|medical'),
      ('finance',r'insurance|financ|capital markets|bank|asset management|real estate'),
      ('consumer',r'consumer|retail|food|beverage|household|personal|hotels|restaurants|leisure|textiles|apparel|durables|gaming'),
      ('energy_utilities',r'energy|oil|gas\b|utilit|power|telecom|wireless'),
      ('industrials',r'capital goods|machinery|building|construction|aerospace|defense|electrical|industrial|trading compan|distributors|transport|logistics|freight|containers|packaging|chemicals|metals|mining|paper|plastics|materials|environmental|automotive|automobile|auto components|wholesale')]
def sector(s):
    s=s.lower()
    for k,r in SECT:
        if re.search(r,s): return k
    return 'other'
p=pd.read_parquet(f'{V}/positions.parquet',columns=['pos_id','cik','family','ptype','ddate','ident','mark','cost','principal','fv','spread','rate','pik'])
p=p.merge(pd.read_parquet(f'{V}/sid.parquet'),on='pos_id')
v=pd.read_parquet(f'{B}/managers/vehicles.parquet')[['cik','listed_from','perpetual']]
p=p.merge(v,on='cik',how='left')
p['vtype']=np.where(p.listed_from.notna()&(p.listed_from<=p.ddate),'listed',np.where(p.perpetual.fillna(False),'perpetual','private'))
p['ok']=(p.cost/p.principal).between(0.9,1.05)&p.mark.between(20,110)&(p.principal>=250e3)
p['has_ind']=p.ident.str.contains(IND,case=False,regex=True)
share=p.groupby('family').has_ind.mean()
src_fam=set(share[share>=0.6].index)
p['src']=p.family.isin(src_fam)&p.has_ind
print('label-source families', len(src_fam), sorted(src_fam))
# position key: the stable within-fund id; nkey is the old key (same fund, same description without numbers and punctuation)
p['nonum']=p.ident.str.lower().str.replace(r'[^a-z ]',' ',regex=True).str.replace(r'\s+',' ',regex=True).str.strip()
p['nkey']=pd.factorize(p.cik.astype(str)+'|'+p.nonum)[0]
p['pkey']=pd.factorize(p.sid)[0]
# sector votes: own label (if source) + co-holders' labels on the same loan at the same date
pr=pd.read_parquet(f'{V}/pairs_0.5.parquet'); pr=pr[pr.cos>=0.6]
src=p.set_index('pos_id')
sec_of=src.loc[src.src,'ident'].map(sector)
votes=pd.concat([pd.DataFrame({'pos_id':sec_of.index,'sector':sec_of.values}),
                 pd.DataFrame({'pos_id':pr.pa.values,'sector':sec_of.reindex(pr.pc.values).values})]).dropna()
votes['pkey']=src.pkey.reindex(votes.pos_id).values
lab=votes.groupby(['pkey','sector']).size().rename('n').reset_index().sort_values(['pkey','n'],ascending=[True,False])
top=lab.drop_duplicates('pkey').set_index('pkey')
tot=lab.groupby('pkey').n.sum()
top['agree']=top.n/tot.reindex(top.index)
p['sector']=p.pkey.map(top.sector); p['agree']=p.pkey.map(top.agree)
# loan id across funds and time: connected components of position keys linked by same-loan pairs
pk=src.pkey
a=pk.reindex(pr.pa.values).values; c=pk.reindex(pr.pc.values).values
n=p.pkey.max()+1
_,comp=connected_components(coo_matrix((np.ones(len(a)),(a,c)),shape=(n,n)),directed=False)
p['loan']=comp[p.pkey.values].astype('int64')
size=p.groupby('loan').pkey.nunique()
print('loan components: positions keys per loan, 99th pct', size.quantile(.99), 'max', size.max(), 'keys in loans with >100 keys', int(size[size>100].sum()))
big=size[size>100].index
p.loc[p.loan.isin(big),'loan']=n+p.loc[p.loan.isin(big),'pkey']   # break up over-merged components
# pre-shock controls, fixed per position
pre=p[p.ddate<='2025-09-30']
p['spread_pre']=p.pkey.map(pre.groupby('pkey').spread.mean())
p['pik_pre']=p.pkey.map(pre.groupby('pkey').pik.max()).fillna(0)
p['held_sep25']=p.pkey.isin(set(p.loc[p.ddate=='2025-09-30','pkey']))
keep=['pos_id','cik','family','ptype','vtype','ddate','mark','cost','principal','spread','pik','ok','pkey','nkey','loan','sector','agree','spread_pre','pik_pre','held_sep25']
p[keep].to_parquet(f'{V}/panel.parquet',index=False)
q=p[p.ok]
print('clean positions', len(q), 'with a sector label', int(q.sector.notna().sum()))
print(q[q.sector.notna()].sector.value_counts().to_string())
print('label coverage by family (top 12 by size):')
print(q.groupby('family').sector.apply(lambda s:s.notna().mean()).reindex(q.family.value_counts().index[:12]).round(2).to_string())
