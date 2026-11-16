# Code audit, 16 Nov 2026. The pre-analysis plan says the funds' own label must use filings up to 30 Sep 2025 only.
# v2_panel.py votes over all dates. This script rebuilds the vote with dates <= 30 Sep 2025 (same rules otherwise),
# writes it NEXT TO the panel (v2/fund_label_pre_2026-11-16.parquet: pkey, sector_pre, agree_pre) without changing panel.parquet,
# and compares the Step 1 estimate under both. Data to 30 Jun 2026 only.
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..')))  # repo root
import re, sys, warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
B=(_ROOT+'/bdc'); V=f'{B}/v2'; O=(_ROOT+'/bdc/audit')
src_code=open(f'{B}/scripts/v2_panel.py').read()
exec(src_code[src_code.index('IND=r'):src_code.index("p=pd.read_parquet")])   # IND, SECT, sector() exactly as in v2_panel.py
base=pd.Timestamp('2025-09-30'); out=[]
def say(s): out.append(s); print(s,flush=True)
pan=pd.read_parquet(f'{V}/panel.parquet'); pan['cik']=pan.cik.astype('int64')
P=pd.read_parquet(f'{V}/positions.parquet',columns=['pos_id','ident'])
p=pan.merge(P,on='pos_id')
p['has_ind']=p.ident.str.contains(IND,case=False,regex=True)
share=p.groupby('family').has_ind.mean(); src_fam=set(share[share>=0.6].index)
p['src']=p.family.isin(src_fam)&p.has_ind
pr=pd.read_parquet(f'{V}/pairs_0.5.parquet'); pr=pr[pr.cos>=0.6]
s=p.set_index('pos_id'); sec_of=s.loc[s.src,'ident'].map(sector)
def vote(maxdate):
    ok_pos=set(s.index[s.ddate<=maxdate]) if maxdate is not None else set(s.index)
    own=pd.DataFrame({'pos_id':sec_of.index,'sector':sec_of.values}); co=pd.DataFrame({'pos_id':pr.pa.values,'sector':sec_of.reindex(pr.pc.values).values})
    v=pd.concat([own,co]).dropna(); v=v[v.pos_id.isin(ok_pos)]; v['pkey']=s.pkey.reindex(v.pos_id).values
    lab=v.groupby(['pkey','sector']).size().rename('n').reset_index().sort_values(['pkey','n'],ascending=[True,False])
    top=lab.drop_duplicates('pkey').set_index('pkey'); top['agree']=top.n/lab.groupby('pkey').n.sum().reindex(top.index); return top
allv=vote(None); prev=vote(base)
chk=pan.drop_duplicates('pkey').set_index('pkey').sector
say(f'Vote over all dates reproduces panel.sector: {(allv.sector.reindex(chk.index).fillna("NA")==chk.fillna("NA")).mean():.4f}')
pd.DataFrame({'sector_pre':prev.sector,'agree_pre':prev.agree}).reset_index().to_parquet(f'{V}/fund_label_pre_2026-11-16.parquet',index=False)
c=pan[(pan.ddate==base)&pan.ok].copy(); c['pre']=c.pkey.map(prev.sector)
say(f'Clean Sep 2025 positions: {len(c)}; fund label (all dates) {c.sector.notna().mean():.3f}; fund label (to Sep 2025) {c.pre.notna().mean():.3f}')
both=c[c.sector.notna()&c.pre.notna()]; say(f'  where both exist: same label {(both.sector==both.pre).mean():.4f}; software flag differs for {int(((both.sector=="software")!=(both.pre=="software")).sum())} positions')
lost=c[c.sector.notna()&c.pre.isna()]; say(f'  labelled only thanks to later filings: {len(lost)} positions, of which software {int((lost.sector=="software").sum())}')
m3=pan[(pan.ddate=='2026-03-31')&pan.mark.between(0,110)&(pan.principal>=250e3)].set_index('pkey').mark; m6=pan[(pan.ddate=='2026-06-30')&pan.mark.between(0,110)&(pan.principal>=250e3)].set_index('pkey').mark
for col,nm in [('sector','all dates (current panel)'),('pre','filings to Sep 2025 only (as registered)')]:
    x=c[c[col].notna()&~c[col].isin(['it_services','tech_other'])].copy(); x['sw']=(x[col]=='software').astype(int)
    x['sp']=x.spread_pre.fillna(0)*100; x['spm']=x.spread_pre.isna().astype(int); x['pk']=x.pik_pre.fillna(0); r=[]
    for q,m in [('Mar 2026',m3),('Jun 2026',m6)]:
        y=x.assign(dy=x.pkey.map(m)-x.mark,db=100*((x.pkey.map(m)<99).astype(float)-(x.mark<99))).dropna(subset=['dy'])
        f1=smf.ols('dy~sw+sp+spm+pk+C(cik)',y).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(y.loan)[0]}); f2=smf.ols('db~sw+sp+spm+pk+C(cik)',y).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(y.loan)[0]})
        r.append(f'{q}: mark {f1.params.sw:+.2f} ({f1.bse.sw:.2f}), Below99 {f2.params.sw:+.1f} ({f2.bse.sw:.1f}), n {len(y)}, software {int(y.sw.sum())}')
    say(f'Step 1, cross-section route, fund label voted over {nm}: '+' | '.join(r))
open(f'{O}/results_a04_fund_label_pre_2026-11-16.txt','w').write('\n'.join(out)+'\n')
