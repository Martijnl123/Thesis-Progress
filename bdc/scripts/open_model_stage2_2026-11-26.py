# Stage 2 of open-model-relabel-rule-2026-11-21.md (full run, gpt-oss:20b, 5,622 of 5,686 items parsed). Written 23 Nov 2026.
# Tests 1-4 exactly as the rule fixes them, plus extra rows marked EXTRA. Data to 30 Jun 2026 only.
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..')))  # repo root
import sys, numpy as np, pandas as pd
sys.path.insert(0,(_ROOT+'/bdc/scripts/draft2')); sys.path.insert(0,(_ROOT+'/bdc/scripts'))
from d2_common import *
D=(_ROOT+'/bdc/industry_llm/open-model-2026-11-21')
out=[]
def say(s=''): out.append(str(s)); print(s,flush=True)
om=pd.read_csv(f'{D}/out_gpt-oss_20b.csv',dtype=str); om['og']=om.group.str.strip().str.upper(); om=om.drop_duplicates('item')
L=pd.read_parquet(f'{F}/labels.parquet'); L['item']=L['item'].astype(str)
B=L[['item','bid','group','principal']].merge(om[['item','og','known']].rename(columns={'known':'oknown'}),on='item',how='left')
say(f'Open-model stage 2, {pd.Timestamp.today():%d %b %Y %H:%M}. Items {len(B)}, open model parsed {B.og.notna().sum()} (missing {B.og.isna().sum()})')
# fallers: same rule as the validation strata
p=pd.read_parquet(f'{V}/panel.parquet',columns=['pos_id','pkey','ddate','mark','ok']); assert p.ddate.max()<=CAP
pb=pd.read_parquet(f'{F}/position_borrower.parquet').merge(p[['pos_id','pkey']],on='pos_id').drop_duplicates('pkey')
a=p[(p.ddate==BASE)&p.ok].drop_duplicates('pkey').set_index('pkey').mark; b=p[p.ddate==JUN].groupby('pkey').mark.mean()
ch=(b-a.reindex(b.index)).dropna(); chb=ch.groupby(pb.set_index('pkey').bid.reindex(ch.index)).mean()
B['fell']=B.bid.map(chb).le(-3)
say(f'Fallers among items: {int(B.fell.sum())}')
kb=B.og.notna()&~B.group.isin(['UNKN','FUND'])&~B.og.isin(['UNKN','FUND'])
K=B[kb].copy(); K['c']=(K.group=='SOFT').astype(int); K['o']=(K.og=='SOFT').astype(int)
say(f'\n1. Agreement, known to both (n {len(K)})')
po=(K.c==K.o).mean(); pe=K.c.mean()*K.o.mean()+(1-K.c.mean())*(1-K.o.mean())
say(f'   same software flag {po:.3f}; Cohen kappa {(po-pe)/(1-pe):.3f}; same 26-group code {(K.group==K.og).mean():.3f}')
say(f'   software share: Claude {K.c.mean():.3f}, open model {K.o.mean():.3f}; Claude SOFT & open not {int(((K.c==1)&(K.o==0)).sum())}, open SOFT & Claude not {int(((K.c==0)&(K.o==1)).sum())}')
say('   where Claude says SOFT and open model does not, the open model says: '+K[(K.c==1)&(K.o==0)].og.value_counts().head(8).to_dict().__str__())
say('   where open says SOFT and Claude does not, Claude says: '+K[(K.c==0)&(K.o==1)].group.value_counts().head(8).to_dict().__str__())
say('   EXTRA all parsed items, open-model group when Claude SOFT: '+B[(B.group=='SOFT')&B.og.notna()].og.value_counts().head(6).to_dict().__str__())
def dstats(X):
    f=X.fell; D1=((X.c==1)&(X.o==0)).astype(float); D2=((X.c==0)&(X.o==1)).astype(float)
    return D1[f].mean(),D1[~f].mean(),D2[f].mean(),D2[~f].mean()
rng=np.random.default_rng(20261121); st=dstats(K)
bs=np.array([dstats(K.iloc[rng.integers(0,len(K),len(K))]) for _ in range(1000)])
def ci(v): return f'[{np.quantile(v,.05):+.3f}, {np.quantile(v,.95):+.3f}]'
d1=st[0]-st[1]; d1b=bs[:,0]-bs[:,1]; d2=st[2]-st[3]; d2b=bs[:,2]-bs[:,3]
say(f'\n2. Leakage test, main: D = Claude SOFT & open not. fallers {st[0]:.3f} (n {int(K.fell.sum())}), others {st[1]:.3f} (n {int((~K.fell).sum())})')
say(f'   difference {d1:+.3f} 90% CI {ci(d1b)} -> leakage indicated: {d1>0 and np.quantile(d1b,.05)>0}')
say(f'\n3. Reverse (open SOFT & Claude not): fallers {st[2]:.3f}, others {st[3]:.3f}, difference {d2:+.3f} {ci(d2b)}')
say(f'   DiD {d1-d2:+.3f} 90% CI {ci(d1b-d2b)} -> leakage indicated: {(d1-d2)>0 and np.quantile(d1b-d2b,.05)>0}')
# EXTRA: software share among fallers vs others in each model (raw), known to both
say(f'   EXTRA software share fallers/others: Claude {K.c[K.fell].mean():.3f}/{K.c[~K.fell].mean():.3f}, open {K.o[K.fell].mean():.3f}/{K.o[~K.fell].mean():.3f}')
# 4. Step 1
p=load(); keys=cohort_keys(p); d0=frozen(p,keys); omap=B.set_index('bid').og
d0['og']=d0.bid.map(omap); d0['sw']=(d0.grp=='SOFT').astype(float)
def row(nm,d):
    r=es(d); n=r['n']
    mj,ms=get(r,'mark','2026-06'); bj,bss=get(r,'b99','2026-06'); mm,mms=get(r,'mark','2026-03'); bm,bms=get(r,'b99','2026-03')
    ok=(mj+1.645*ms<0) and (bj-1.645*bss>0)
    say(f'   {nm}: pos {n["pos"]} (sw {n["sw"]}), borrowers {n["borrowers"]} | mark Mar {mm:+.2f} ({mms:.2f}) Jun {mj:+.2f} ({ms:.2f}) | b99 Mar {bm:+.1f} ({bms:.1f}) Jun {bj:+.1f} ({bss:.1f}) | June both 90% CIs exclude 0 right side: {ok}')
    return ok
say('\n4. Step 1, headline spec (frozen Sep 2025 cohort, position + fund x quarter + vintage x quarter FE, borrower-clustered)')
row('headline, Claude labels',d0)
x=d0.copy(); x['sw']=(x.og=='SOFT').astype(float); a4=row('(a) same cohort, open-model flag in place of Claude (open UNKN/FUND/missing = not software)',x)
kb2=x.og.notna()&~x.og.isin(['UNKN','FUND']); kk=set(x.loc[kb2&(x.ddate==BASE),'pkey'])
y=d0[d0.pkey.isin(kk)].copy(); row('(b1) known to both, Claude flag',y); y['sw']=(y.og=='SOFT').astype(float); b4=row('(b) known to both, open-model flag',y)
say(f'   Reading rule (a) and (b) both hold: {a4 and b4}')
# EXTRA: both models agree software vs both agree not; open model's own cohort
z=y.copy(); z=z[(z.grp=='SOFT')==(z.og=='SOFT')]; row('EXTRA known to both and models agree on the flag',z)
p['og']=p.bid.map(omap); kk3=set(p.loc[(p.ddate==BASE)&p.ok&p.og.notna()&~p.og.isin(['UNKN','FUND']),'pkey'])
w=frozen(p,kk3); w['sw']=(w.og=='SOFT').astype(float); row('EXTRA open model own cohort (its known borrowers, Claude labels ignored)',w)
# EXTRA disagreement groups in one regression, known to both
q=y.copy(); q['both']=((q.grp=='SOFT')&(q.og=='SOFT')).astype(float); q['conly']=((q.grp=='SOFT')&(q.og!='SOFT')).astype(float); q['oonly']=((q.grp!='SOFT')&(q.og=='SOFT')).astype(float)
r=es(q,swcols=('both','conly','oonly')); u=q.drop_duplicates('pkey')
say(f'   EXTRA disagreement split (vs both-not), known to both: positions both {int(u.both.sum())}, Claude-only {int(u.conly.sum())}, open-only {int(u.oonly.sum())}')
for c_ in ['both','conly','oonly']:
    say(f'      {c_}: June mark {get(r,"mark","2026-06",c_)[0]:+.2f} ({get(r,"mark","2026-06",c_)[1]:.2f}), b99 {get(r,"b99","2026-06",c_)[0]:+.1f} ({get(r,"b99","2026-06",c_)[1]:.1f})')
open((_ROOT+'/bdc/industry_llm/open-model-2026-11-21/results_open_model_stage2_2026-11-26_full5686.txt'),'w').write('\n'.join(out)+'\n')
# EXTRA (added after the first run, 23 Nov): test 2 conditional on Claude's software call, to separate composition from leakage
def cond(X):
    f=X.fell; s=X.c==1; return ((X.o==0)&s&f).sum()/(s&f).sum(), ((X.o==0)&s&~f).sum()/(s&~f).sum()
cs=cond(K); cb=np.array([cond(K.iloc[rng.integers(0,len(K),len(K))]) for _ in range(1000)])
say(f'\nEXTRA P(open not SOFT | Claude SOFT): fallers {cs[0]:.3f}, others {cs[1]:.3f}, difference {cs[0]-cs[1]:+.3f} 90% CI {ci(cb[:,0]-cb[:,1])}')
open((_ROOT+'/bdc/industry_llm/open-model-2026-11-21/results_open_model_stage2_2026-11-26_full5686.txt'),'w').write('\n'.join(out)+'\n')
