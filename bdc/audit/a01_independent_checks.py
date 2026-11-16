# Code audit, 16 Nov 2026. Independent re-computation of the draft 1 headline numbers. Data to 30 Jun 2026 only.
# 1. Headline event study with pyfixest (independent package) vs our own fit()/se() code.
# 2. A second route to the same number: cross-section of within-position changes, software dummy, fund effects.
# 3. Leave one manager out; borrower-level collapse; share of managers with a negative software effect.
# 4. Traded / no-market-price split by the cross-section route.
# 5. Why traded loans "lose" their Level 2 match: BDC-side filter or no N-PORT line.
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..')))  # repo root
import warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
V=(_ROOT+'/bdc/v2'); F=(_ROOT+'/bdc/industry_llm/full')
O=(_ROOT+'/bdc/audit')
base=pd.Timestamp('2025-09-30'); out=[]
def say(s): out.append(s); print(s,flush=True)
p=pd.read_parquet(f'{V}/panel.parquet'); p['cik']=p.cik.astype('int64'); assert p.ddate.max()<=pd.Timestamp('2026-06-30')
L=pd.read_parquet(f'{F}/labels.parquet')[['bid','group']]
pb=pd.read_parquet(f'{F}/position_borrower.parquet').merge(L,on='bid').merge(p[['pos_id','pkey']],on='pos_id').drop_duplicates('pkey').set_index('pkey')
p['grp']=p.pkey.map(pb.group); p['bid']=p.pkey.map(pb.bid); p['b99']=100*(p.mark<99)
lab=p.grp.notna()&~p.grp.isin(['UNKN','FUND'])
keys=set(p.loc[(p.ddate==base)&p.ok&lab,'pkey'])
fz=p[p.pkey.isin(keys)]; post=fz.ddate>base
d=fz[(~post&fz.ok)|(post&fz.mark.between(0,110)&(fz.principal>=250e3))].copy(); d['sw']=(d.grp=='SOFT').astype(int)
d['q']=d.ddate.dt.strftime('%Y%m'); d['fq']=d.cik.astype(str)+'_'+d.q
d['vint']=d.pkey.map(p.groupby('pkey').ddate.min()).dt.strftime('%Y%m'); d['vq']=d.vint+'_'+d.q
d['sp']=d.spread_pre.fillna(0)*100; d['spm']=d.spread_pre.isna().astype(int); d['pk']=d.pik_pre.fillna(0)
say('1. Headline event study with pyfixest (position, fund x quarter, vintage x quarter FE; spread, missing spread, PIK x quarter; SE by borrower)')
try:
    import pyfixest as pf
    for y in ['mark','b99']:
        m=pf.feols(f'{y} ~ i(q, sw, ref="202509") + i(q, sp, ref="202509") + i(q, spm, ref="202509") + i(q, pk, ref="202509") | pkey + fq + vq',data=d,vcov={'CRV1':'bid'})
        t=m.tidy(); t=t[t.index.str.contains('sw')]
        say(f'   {y}: '+'; '.join(f'{i} {r["Estimate"]:+.2f} ({r["Std. Error"]:.2f})' for i,r in t.iterrows() if any(k in i for k in ['202512','202603','202606'])))
    say('   our code (results_draft1_tables): mark Dec +0.01 (0.13) Mar -2.16 (0.36) Jun -2.43 (0.47) | b99 Dec +1.83 (1.09) Mar +30.29 (2.36) Jun +30.25 (2.56)')
except Exception as e: say(f'   pyfixest failed: {e!r}')
say('\n2. Cross-section route: change Sep 2025 -> quarter for positions held at both, on a software dummy with fund effects, SE by borrower')
w=d.pivot_table(index='pkey',columns='q',values='mark'); info=d.drop_duplicates('pkey').set_index('pkey')[['cik','family','bid','sw','sp','spm','pk']]
cls=pd.read_parquet(f'{V}/traded_untraded_sep25_2026-11-13.parquet').set_index('pkey').cls
def xs(q,mask=None,controls=True,y='mark'):
    x=pd.DataFrame({'dy':(w[q]-w['202509']) if y=='mark' else 100*((w[q]<99).astype(float)-(w['202509']<99))}).join(info); x=x[w[q].notna()&w['202509'].notna()]
    if mask is not None: x=x[mask.reindex(x.index).fillna(False)]
    f='dy~sw'+('+sp+spm+pk' if controls else '')+'+C(cik)'
    m=smf.ols(f,x).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(x.bid)[0]}); return m.params.sw,m.bse.sw,len(x),x
for q in ['202512','202603','202606']:
    b,s,n,_=xs(q); b2,s2,_,_=xs(q,y='b99'); say(f'   {q}: mark {b:+.2f} ({s:.2f}); Below99 {b2:+.1f} ({s2:.1f}); n {n}')
say('\n3. Who drives it (March 2026, cross-section route)')
b0,s0,n0,x0=xs('202603')
loo={}
for fam in x0.family.value_counts().index[:25]:
    xx=x0[x0.family!=fam]; m=smf.ols('dy~sw+sp+spm+pk+C(cik)',xx).fit(); loo[fam]=m.params.sw
say(f'   leave one manager out (25 largest): min {min(loo.values()):+.2f} ({min(loo,key=loo.get)}), max {max(loo.values()):+.2f} ({max(loo,key=loo.get)}); full {b0:+.2f}')
per=[]
for fam,g in x0.groupby('family'):
    if g.sw.sum()>=10 and (1-g.sw).sum()>=10:
        gg=g.assign(dd=g.dy-g.groupby('cik').dy.transform('mean'),ss=g.sw-g.groupby('cik').sw.transform('mean')); per.append((fam,(gg.dd*gg.ss).sum()/(gg.ss**2).sum(),len(g)))
per=pd.DataFrame(per,columns=['fam','b','n']); say(f'   managers with 10+ software and 10+ other loans: {len(per)}; software effect negative in {int((per.b<0).sum())}; median {per.b.median():+.2f}; quartiles {per.b.quantile(.25):+.2f} / {per.b.quantile(.75):+.2f}')
say('   largest 8 managers: '+'; '.join(f'{r.fam} {r.b:+.2f}' for r in per.sort_values('n',ascending=False).head(8).itertuples()))
bb=x0.groupby('bid').agg(dy=('dy','mean'),sw=('sw','first')); m=smf.ols('dy~sw',bb).fit(cov_type='HC1')
say(f'   one observation per borrower (no fund effects): software {m.params.sw:+.2f} ({m.bse.sw:.2f}); borrowers {len(bb)}; medians software {bb[bb.sw==1].dy.median():+.2f}, other {bb[bb.sw==0].dy.median():+.2f}')
ml=x0.assign(loan=p.drop_duplicates('pkey').set_index('pkey').loan.reindex(x0.index)).groupby(['family','loan']).agg(dy=('dy','mean'),sw=('sw','first'),bid=('bid','first')).reset_index()
m=smf.ols('dy~sw+C(family)',ml).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(ml.bid)[0]}); say(f'   one observation per manager and loan, manager effects: {m.params.sw:+.2f} ({m.bse.sw:.2f}); n {len(ml)}')
say('\n4. Split by market price at Sep 2025, cross-section route')
for nm,mask in [('traded',cls=='traded'),('no market price (private, L3 only, not matched)',cls.isin(['untraded','L3 only','not matched'])),('private',cls=='untraded'),('L3 only',cls=='L3 only'),('not matched',cls=='not matched')]:
    r=[]
    for q in ['202603','202606']:
        b,s,n,_=xs(q,mask); b2,s2,_,_=xs(q,mask,y='b99'); r.append(f'{q} mark {b:+.2f} ({s:.2f}), Below99 {b2:+.1f} ({s2:.1f}), n {n}')
    say(f'   {nm}: '+' | '.join(r))
say('\n5. Software positions with a verified Level 2 price at Sep 2025: what happened to the match by June 2026')
M=pd.read_parquet(f'{V}/nport_matches_coupon.parquet'); M2=M[M.verified&(M.lvl=='2')]
s9=set(M2[M2.ddate==base].pos_id); k9=p[(p.ddate==base)&p.pos_id.isin(s9)&p.pkey.isin(keys)]
j=p[(p.ddate=='2026-06-30')&p.pkey.isin(set(k9.pkey))].copy(); j['held']=j.mark.between(0,110)&(j.principal>=250e3)
j['l2']=j.pos_id.isin(set(M2[M2.ddate=='2026-06-30'].pos_id)); j['anym']=j.pos_id.isin(set(M[M.ddate=='2026-06-30'].pos_id)); j['l3v']=j.pos_id.isin(set(M[M.verified&(M.lvl=='3')&(M.ddate=='2026-06-30')].pos_id))
for nm,g in j[j.held].groupby(j.grp.eq('SOFT').map({True:'software',False:'other'})):
    lost=g[~g.l2]; say(f'   {nm}: held in June {len(g)}; still Level 2 {g.l2.mean():.3f}; lost {len(lost)}: BDC row fails the sample filter in June (so it was never searched) {int((~lost.ok).sum())}; searched, verified Level 3 line only {int((lost.ok&lost.l3v).sum())}; searched, name match but nothing verified {int((lost.ok&~lost.l3v&lost.anym).sum())}; searched, no N-PORT line {int((lost.ok&~lost.anym).sum())}; mean June mark of lost {lost.mark.mean():.1f} vs kept {g[g.l2].mark.mean():.1f}')
open(f'{O}/results_a01_independent_checks_2026-11-16.txt','w').write('\n'.join(out)+'\n')
