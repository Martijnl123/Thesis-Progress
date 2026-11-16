# 30 Oct 2026: specialisation vs diversification. Do managers that hold more software (specialists) mark software
# differently in the shock? Exposure s_m = manager's software share of BDC principal at Sep 2025 (LLM labels, UNKN/FUND dropped).
# (1) descriptive: distribution of s_m; ex-ante quality of specialists' software loans (spread, PIK, mark at Sep 2025).
# (2) H1 on the frozen cohort with sw x post x s_m (position + fund x quarter FE; s_m x post is absorbed by fund x quarter).
# (3) same loan, same date: manager-level marks on loans held by 2+ managers, loan x date FE, s_m x post x software.
#     Holding the loan fixed removes selection (specialists picking better loans), so (3) is about valuation, (2) mixes both.
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..')))  # repo root
import sys, warnings, numpy as np, pandas as pd
sys.path.insert(0,(_ROOT+'/bdc/scripts'))
from ai_rev2_lib import fit, se
warnings.filterwarnings('ignore')
V=(_ROOT+'/bdc/v2'); F=(_ROOT+'/bdc/industry_llm/full')
p=pd.read_parquet(f'{V}/panel.parquet')
p['fq']=p.cik.astype(str)+'_'+p.ddate.dt.strftime('%Y%m')
L=pd.read_parquet(f'{F}/labels.parquet')[['bid','group']]
g=pd.read_parquet(f'{F}/position_borrower.parquet').merge(L,on='bid').merge(p[['pos_id','pkey']],on='pos_id').drop_duplicates('pkey').set_index('pkey').group
p['grp']=p.pkey.map(g); lab=p.grp.notna()&~p.grp.isin(['UNKN','FUND'])
b=p[(p.ddate=='2025-09-30')&p.ok&lab].copy(); b['sw']=b.grp.eq('SOFT')
sm=b.groupby('family').apply(lambda x: x.principal[x.sw].sum()/x.principal.sum()).rename('s_m')
nm=b.groupby('family').size()
print(f'managers {len(sm)}; software share of principal: mean {sm.mean():.3f}, median {sm.median():.3f}, p10 {sm.quantile(.1):.3f}, p90 {sm.quantile(.9):.3f}, max {sm.max():.3f}')
print('  principal-weighted mean', (sm*b.groupby('family').principal.sum()).sum()/b.principal.sum())
print('  top 10:', ', '.join(f'{k} {v:.2f} (n {nm[k]})' for k,v in sm.sort_values().tail(10).items()))
b=b.merge(sm,left_on='family',right_index=True); b['spec']=b.s_m>=sm.median()
b['spread_bp']=b.spread*1e4
print('Sep 2025 software loans, specialists (s_m >= median) vs others: means')
for c in ['spread_bp','pik','mark','principal']:
    x=b[b.sw]; print(f'   {c}: specialists {x.loc[x.spec,c].mean():.2f}  others {x.loc[~x.spec,c].mean():.2f}')
# (2) frozen cohort H1 with interaction
keys=set(b.pkey); fz=p[p.pkey.isin(keys)&lab]; post=fz.ddate>'2025-09-30'
d=fz[~post&fz.ok | post&fz.mark.between(0,110)&(fz.principal>=250e3)].copy()
d=d.merge(sm,left_on='family',right_index=True)
d['sw']=d.grp.eq('SOFT').astype(float); d['post']=(d.ddate>='2026-03-31').astype(float); d['b99']=100*(d.mark<99)
d['sp_post']=d.spread_pre.fillna(0)*100*d.post; d['spm_post']=d.spread_pre.isna()*d.post; d['pik_post']=d.pik_pre*d.post
d['sw_post']=d.sw*d.post; z=(d.s_m-sm.mean())/sm.std(); d['sw_post_s']=d.sw_post*z
C=['sp_post','spm_post','pik_post']
print('(2) frozen cohort, Q1 2026 start; s_m standardised across managers (1 SD =',f'{sm.std():.3f})')
for y in ['mark','b99']:
    dd,bb,sc,Ainv=fit(d,y,['sw_post','sw_post_s']+C,['pkey','fq']); e=se(dd,sc,Ainv,('loan','family'))
    print(f'   {y}: sw x post {bb["sw_post"]:+.2f} ({e[0]:.2f}); x exposure {bb["sw_post_s"]:+.2f} ({e[1]:.2f})  [SE loan+family]')
# (3) same loan, manager-level marks
q=p[p.ok&lab&(p.ddate>='2024-12-31')].copy()
q=q.groupby(['loan','ddate','family']).apply(lambda x: pd.Series({'mark':np.average(x.mark,weights=x.principal),'grp':x.grp.iloc[0]})).reset_index()
q=q.merge(sm,left_on='family',right_index=True)
q['n']=q.groupby(['loan','ddate']).family.transform('size'); q=q[q.n>=2]
q['sw']=q.grp.eq('SOFT').astype(float); q['post']=(q.ddate>='2026-03-31').astype(float); q['b99']=100*(q.mark<99)
q['ld']=q.loan.astype(str)+'_'+q.ddate.dt.strftime('%Y%m'); q['s']=(q.s_m-sm.mean())/sm.std()
q['s_post']=q.s*q.post; q['s_sw']=q.s*q.sw; q['s_post_sw']=q.s*q.post*q.sw
print(f'(3) same loan, Dec 2024 - Jun 2026: obs {len(q)}, loans {q.loan.nunique()}, managers {q.family.nunique()}')
for y in ['mark','b99']:
    X=['s','s_post','s_sw','s_post_sw']; dd,bb,sc,Ainv=fit(q,y,X,['ld']); e=se(dd,sc,Ainv,('loan','family'))
    print(f'   {y}: '+'; '.join(f'{c} {bb[c]:+.3f} ({e[i]:.3f})' for i,c in enumerate(X))+'  [SE loan+family]')
