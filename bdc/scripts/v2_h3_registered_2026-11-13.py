# Registered H3 (PAP v5, OSF 10.17605/OSF.IO/PSZ5D, section 7), run 13 Nov 2026 on quarters through June 2026 only.
# The Sep 2026 hold-out lock stays closed: this script reads only v2/panel.parquet, which ends at 30 June 2026, and stops otherwise.
#
# Design as registered:
# - Unit = manager (family); a manager's mark on a loan = mean of its funds' positions in that loan.
# - Revolver and delayed-draw positions dropped, and positions whose lien differs from the loan's (principal-weighted majority)
#   lien, BEFORE the top two holders are chosen (regexes as v2_h3_tranche_filter.py).
# - Sample: loans held by 3+ managers at 30 Sep 2025, every manager's mark >= 98, usable model label (not UNKN/FUND).
# - Top two holders fixed at Sep 2025 by principal. Outcome: |mark_top1 - mark_top2| each quarter (frozen Sep 2025 position keys).
# - Row 1: gap on soft x Dec25 and soft x post (post = Mar 2026 on), loan FE + quarter x Sep 2025 mark-bin FE.
# - Row 2 (primary): same, with quarter x bin of the OTHER holders' mean mark in that quarter.
# - Row 3: decomposition with the gap-to-level relation estimated on non-software loans before the shock.
# - SEs dyadic on both managers of the pair (Aronow, Samii and Assenova 2015); by loan as a check.
# - Sector placebos with software's rank; checks: 4+ managers, range across all holders by holder count, rounding, Dec 2025
#   dropped, both managers December fiscal year-end.
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..')))  # repo root
import sys, warnings, numpy as np, pandas as pd
sys.path.insert(0,(_ROOT+'/bdc/scripts'))
from ai_rev2_lib import fit, se, _ss
from ai_prelim_fe import _codes
warnings.filterwarnings('ignore')
V=(_ROOT+'/bdc/v2'); F=(_ROOT+'/bdc/industry_llm/full')
import os
CG=os.environ.get('H3_COSTGUARD','0')=='1'   # 1 = drop later-quarter positions with cost < 50% of principal (see below)
KEYS=os.environ.get('H3_KEYS','frozen')   # frozen = only position keys held at Sep 2025; all = every position of the manager in the loan
WT=os.environ.get('H3_W','mean')          # mean = simple mean over the manager's positions; pw = principal-weighted
TAG=('_guarded' if CG else '_noguard')+f'_{KEYS}_{WT}'
OUT=f'{V}/results_h3_registered{TAG}_2026-11-13.txt'
lines=[]
def say(*a):
    s=' '.join(str(x) for x in a); print(s,flush=True); lines.append(s)

p=pd.read_parquet(f'{V}/panel.parquet')
assert p.ddate.max()<=pd.Timestamp('2026-06-30'), 'hold-out lock: panel must end at 30 June 2026'
pos=pd.read_parquet(f'{V}/positions.parquet',columns=['pos_id','ident','fv','form','cik','ddate'])
ident=pos.set_index('pos_id').ident.str.lower()
p['rev']=p.pos_id.map(ident.str.contains(r'revolv|\brcf\b',regex=True))
p['dd']=p.pos_id.map(ident.str.contains(r'delayed|\bddtl\b',regex=True))
p['l2']=p.pos_id.map(ident.str.contains(r'second lien|2nd lien',regex=True))
fv=pos.set_index('pos_id').fv
p['res']=np.where(p.pos_id.map(fv%1000==0),100*500/p.principal,100*0.5/p.principal)   # mark resolution from FV rounding
L=pd.read_parquet(f'{F}/labels.parquet')[['bid','group']]
gl=pd.read_parquet(f'{F}/position_borrower.parquet').merge(L,on='bid').merge(p[['pos_id','loan']],on='pos_id').drop_duplicates('loan').set_index('loan').group
p['grp']=p.loan.map(gl)
k=pos[pos.form.str.startswith('10-K',na=False)].groupby('cik').ddate.agg(lambda s: s.dt.month.mode().iloc[0])
p['fye']=p.cik.astype('float').map(k.rename(index=lambda i: float(i)))

base=p[(p.ddate=='2025-09-30')&p.ok&p.grp.notna()&~p.grp.isin(['UNKN','FUND'])].copy()
def build(base, tranche_filter=True):
    x=base.copy()
    if tranche_filter:
        x=x[~(x.rev|x.dd)]
        maj=x.groupby('loan').apply(lambda g: np.average(g.l2,weights=g.principal)>0.5)
        x=x[x.l2.values==x.loan.map(maj).values]
    m=x.groupby(['loan','family']).agg(mark=('mark','mean'),prin=('principal','sum'),grp=('grp','first')).reset_index()
    n=m.groupby('loan').family.size(); mn=m.groupby('loan').mark.min()
    keep=n[(n>=3)&(mn.reindex(n.index)>=98)].index
    m=m[m.loan.isin(keep)].sort_values(['loan','prin','family'],ascending=[True,False,True])
    m['rk']=m.groupby('loan').cumcount()
    return x[x.loan.isin(keep)], m
def panel(x, m, quarters):
    # Each quarter, a manager's mark = mean over ALL its positions in the loan that quarter (registered: "average across its
    # funds"), with the same tranche filter (no revolver/delayed draw, same lien as the loan's Sep 2025 majority lien), mark
    # 0-110 and principal >= $250k as in the frozen-cohort rule. Managers and top-two roles stay fixed at Sep 2025.
    lien=x.groupby('loan').l2.agg(lambda s: s.mean()>0.5)
    q=p[p.loan.isin(m.loan.unique())&p.ddate.isin(quarters)&p.mark.between(0,110)&(p.principal>=250000)&~(p.rev|p.dd)]
    q=q[q.l2.values==q.loan.map(lien).values].merge(m[['loan','family']],on=['loan','family'])
    # Hand-check 13 Nov 2026: the extreme gaps on non-software loans came from partly funded delayed-draw/revolver tranches
    # (cost and FV on the funded part, principal = full commitment, marks 6-38) that the stable key linked to a Sep 2025 term
    # loan (mostly GS funds, whose descriptions are identical across tranches). A markdown does not change cost, so a position
    # with cost below 50% of principal is a funding artefact, not a mark. CG=1 drops those rows.
    # Second guard (same hand-check): affiliated funds of one manager share one mark, so a position more than 10 points from
    # its manager's median mark on that loan and date is a key mix-up or a 255-character merged row (e.g. GS Guidepoint 6.49
    # next to 99.50 in sibling funds), not a mark.
    if CG:
        n0=len(q); q=q[q.cost>=0.5*q.principal]; n1=len(q)
        med=q.groupby(['loan','family','ddate']).mark.transform('median'); q=q[(q.mark-med).abs()<=10]
        say(f'  guards: dropped {n0-n1} rows with cost < 50% of principal and {n1-len(q)} rows > 10 points from the manager median, of {n0}')
    # frozen keys applied after the guards, so the manager median uses all its same-tranche positions on that loan and date
    if KEYS=='frozen': q=q[q.pkey.isin(set(x.pkey))]
    q=q.assign(mp=q.mark*q.principal)
    mq=q.groupby(['loan','family','ddate']).agg(mark=('mark','mean'),mp=('mp','sum'),pr=('principal','sum'),res=('res','max')).reset_index()
    if WT=='pw': mq['mark']=mq.mp/mq.pr
    mq=mq.merge(m[['loan','family','rk']],on=['loan','family'])
    t1=mq[mq.rk==0].set_index(['loan','ddate']); t2=mq[mq.rk==1].set_index(['loan','ddate'])
    oth=mq[mq.rk>=2].groupby(['loan','ddate']).agg(om=('mark','mean'),no=('mark','size'))
    rng=mq.groupby(['loan','ddate']).agg(rng=('mark',lambda s:s.max()-s.min()),nh=('mark','size'))
    g=t1[['mark','res','family']].join(t2[['mark','res','family']],lsuffix='1',rsuffix='2',how='inner').join(oth).join(rng).reset_index()
    g['gap']=(g.mark1-g.mark2).abs(); g['gap_r']=np.where(g.gap<g.res1+g.res2,0.0,g.gap)
    g['fa']=np.minimum(g.family1,g.family2); g['fc']=np.maximum(g.family1,g.family2)
    info=m.groupby('loan').agg(grp=('grp','first'),nm=('family','size'),sepmean=('mark','mean'))
    g=g.merge(info,left_on='loan',right_index=True)
    g['q']=g.ddate.dt.strftime('%Y%m'); g['sw']=(g.grp=='SOFT').astype(float)
    g['dec']=(g.q=='202512').astype(float); g['post']=(g.ddate>='2026-03-31').astype(float)
    g['sep_bin']=pd.cut(g.sepmean,[-1,99,99.5,100,1e9]).astype(str)
    g['oth_bin']=pd.cut(g.om,[-1,80,90,95,97,98,99,99.5,100,1e9]).astype(str).where(g.om.notna(),'none')
    g['oth_bin_f']=pd.cut(g.om,[-1,70,80,85,90,92,94,95,96,97,97.5,98,98.5,99,99.25,99.5,99.75,100,100.5,1e9]).astype(str).where(g.om.notna(),'none')
    return g

def se_dyad(d, sc, Ainv, a='fa', b='fc'):
    two=np.vstack([sc,sc]); A=d[a].values; B=d[b].values
    M=_ss(two,_codes(pd.Series(np.concatenate([A,B]))),False)-_ss(sc,_codes(pd.Series(A+'|'+B)),False)
    return np.sqrt(np.clip(np.diag(Ainv@M@Ainv),0,None))
def reg(g, y, fe, ind='sw', label='', show=True):
    c=g.copy(); c['x_dec']=c[ind]*c.dec; c['x_post']=c[ind]*c.post
    d,b,sc,Ainv=fit(c,y,['x_dec','x_post'],fe)
    sd=se_dyad(d,sc,Ainv); sl=se(d,sc,Ainv,'loan')
    if show: say(f'  {label}: soft x Dec25 {b.x_dec:+.3f} (dyadic SE {sd[0]:.3f}; loan {sl[0]:.3f}) | soft x post {b.x_post:+.3f} (dyadic SE {sd[1]:.3f}; loan {sl[1]:.3f}) | n {len(d)}, loans {d.loan.nunique()}')
    return b.x_post, sd[1]

Q=[pd.Timestamp(s) for s in ['2025-09-30','2025-12-31','2026-03-31','2026-06-30']]
QPRE=[pd.Timestamp(s) for s in ['2024-12-31','2025-03-31','2025-06-30']]+Q

say('Registered H3 (PAP v5 section 7), run 13 Nov 2026 on quarters through 30 June 2026. Not confirmatory. Sep 2026 hold-out not used.')
say('Version:', f'keys={KEYS}, weights={WT},', 'GUARDED (cost < 50% of principal dropped; positions > 10 points from their manager median on the loan-date dropped)' if CG else 'AS REGISTERED (no cost guard)')
for tf,lab in [(False,'no tranche filter (how the registered sample count was made)'),(True,'tranche filter as registered')]:
    x,m=build(base,tf); n=m.groupby('loan').size(); gg=m.groupby('loan').grp.first()
    say(f'Sample, {lab}: loans {len(n)}, software {(gg=="SOFT").sum()}, exactly 3 managers {(n==3).sum()}, distinct top-two managers {m[m.rk<=1].family.nunique()}')
x,m=build(base,True)
g=panel(x,m,Q); ge=panel(x,m,QPRE)
say(f'Loan-quarters with both top-two holders present: {len(g)} ({g.loan.nunique()} loans); with other holders present {g.om.notna().mean():.1%}')
say('\nMean top-two gap by quarter (software / other) and other holders\' mean mark:')
t=g.groupby(['q','sw']).agg(gap=('gap','mean'),om=('om','mean'),n=('gap','size')).unstack(); say(t.round(2).to_string())

say('\n=== Main rows (sample Sep 2025 to Jun 2026) ===')
r1=reg(g,'gap',['loan',g.columns[0]] if False else ['loan','q'],label='Row 0 (loan + quarter FE, for reference)')
g['qsep']=g.q+'|'+g.sep_bin; g['qoth']=g.q+'|'+g.oth_bin; g['qothf']=g.q+'|'+g.oth_bin_f
b1=reg(g,'gap',['loan','qsep'],label='Row 1 (quarter x Sep 2025 mark bin; keeps the mechanical part)')
b2=reg(g,'gap',['loan','qoth'],label='Row 2 PRIMARY (quarter x other holders\' mean-mark bin)')
reg(g,'gap',['loan','qothf'],label='Row 2, finer other-holder bins (check)')
g['gap1']=100*(g.gap>1); g['gapw']=g.gap.clip(upper=g.gap.quantile(.99))
reg(g,'gap1',['loan','qsep'],label='NOT REGISTERED, descriptive: 100 x 1[gap > 1], row 1 FE')
reg(g,'gap1',['loan','qoth'],label='NOT REGISTERED, descriptive: 100 x 1[gap > 1], row 2 FE')
reg(g,'gapw',['loan','qoth'],label='NOT REGISTERED: gap winsorised at its 99th percentile, row 2 FE')

say('\n=== Event time, row 2 FE, base Sep 2025 (pre quarters use loans where both top-two held then) ===')
c=ge.copy(); c['qoth']=c.q+'|'+c.oth_bin; cols=[]
for qq in sorted(c.q.unique()):
    if qq=='202509': continue
    c[f'e{qq}']=c.sw*(c.q==qq); cols.append(f'e{qq}')
d,b,sc,Ainv=fit(c,'gap',cols,['loan','qoth']); sd=se_dyad(d,sc,Ainv)
say('  '+'; '.join(f'{k[1:5]}-{k[5:]} {b[k]:+.3f} ({sd[i]:.3f})' for i,k in enumerate(cols)))

say('\n=== Row 3: decomposition with the pre-shock gap-to-level relation (non-software loans, Mar 2023 to Sep 2025) ===')
# Pre-shock relation: every non-software labelled loan with 3+ managers in a quarter, top two by principal in that quarter,
# gap against the other holders' mean-mark bin (same bins as row 2). No >=98 restriction, so the relation covers low levels.
pp=p[(p.ddate>='2023-03-31')&(p.ddate<='2025-09-30')&p.ok&p.grp.notna()&~p.grp.isin(['UNKN','FUND','SOFT'])&~(p.rev|p.dd)]
mm=pp.groupby(['loan','ddate','family']).agg(mark=('mark','mean'),prin=('principal','sum')).reset_index()
mm['nm']=mm.groupby(['loan','ddate']).family.transform('size'); mm=mm[mm.nm>=3].sort_values(['loan','ddate','prin'],ascending=[True,True,False])
mm['rk']=mm.groupby(['loan','ddate']).cumcount()
a1=mm[mm.rk==0].set_index(['loan','ddate']).mark; a2=mm[mm.rk==1].set_index(['loan','ddate']).mark
oo=mm[mm.rk>=2].groupby(['loan','ddate']).mark.mean()
rel=pd.DataFrame({'gap':(a1-a2).abs(),'om':oo}).dropna()
rel['bin']=pd.cut(rel.om,[-1,80,90,95,97,98,99,99.5,100,1e9]).astype(str)
f=rel.groupby('bin').gap.agg(['mean','size']); say('  pre-shock gap by other-holder bin (non-software): '+'; '.join(f'{i} {r["mean"]:.2f} (n {int(r["size"])})' for i,r in f.iterrows()))
g['pred']=g.oth_bin.map(f['mean'])
w=g[g.q.isin(['202509','202606'])&g.pred.notna()].pivot_table(index=['loan','sw'],columns='q',values=['gap','pred']).dropna(); w.columns=[f'{a}_{b}' for a,b in w.columns]; w=w.reset_index()
w['dg']=w.gap_202606-w.gap_202509; w['dp']=w.pred_202606-w.pred_202509
s=w.groupby('sw')[['dg','dp']].mean(); dd=s.loc[1.0]-s.loc[0.0]
say(f'  Sep 2025 -> Jun 2026, loans with both top-two in both quarters ({len(w)}): change in gap software {s.loc[1.0,"dg"]:+.2f}, other {s.loc[0.0,"dg"]:+.2f}; predicted from level software {s.loc[1.0,"dp"]:+.2f}, other {s.loc[0.0,"dp"]:+.2f}')
say(f'  difference software-other: actual {dd.dg:+.2f}, explained by level {dd.dp:+.2f} ({dd.dp/dd.dg:.0%}), unexplained {dd.dg-dd.dp:+.2f}')
# bootstrap by loan for the unexplained part
rng_=np.random.default_rng(20261113); bs=[]
for _ in range(2000):
    ww=w.iloc[rng_.integers(0,len(w),len(w))]; ss=ww.groupby('sw')[['dg','dp']].mean()
    if len(ss)==2: bs.append((ss.loc[1.0,'dg']-ss.loc[1.0,'dp'])-(ss.loc[0.0,'dg']-ss.loc[0.0,'dp']))
say(f'  unexplained part: loan bootstrap SE {np.std(bs):.2f}, 95% interval [{np.percentile(bs,2.5):+.2f}, {np.percentile(bs,97.5):+.2f}]')

say('\n=== Sector placebos, row 2 spec (software excluded from every placebo\'s comparison loans), groups with 10+ loans ===')
grp_n=m.groupby('loan').grp.first().value_counts(); res=[]
for grp in grp_n[grp_n>=10].index:
    if grp=='SOFT': c=g.copy(); ind='sw'
    else: c=g[g.grp!='SOFT'].copy(); c['ind']=(c.grp==grp).astype(float); ind='ind'
    bb,ss=reg(c,'gap',['loan','qoth'],ind=ind,show=False); res.append((grp,grp_n[grp],bb,ss,bb/ss))
R=pd.DataFrame(res,columns=['group','loans','b_post','dyadic_se','t']).sort_values('t',ascending=False).reset_index(drop=True)
R.index+=1; say(R.round(3).to_string()); say(f'  software rank on t (widening first): {int(R.index[R.group=="SOFT"][0])} of {len(R)}')

say('\n=== Checks (row 2 spec unless stated) ===')
reg(g[g.nm>=4],'gap',['loan','qoth'],label='Loans with 4+ managers')
reg(g[g.nm==3],'gap',['loan','qoth'],label='Loans with exactly 3 managers')
reg(g,'gap_r',['loan','qoth'],label='Gaps below FV rounding resolution set to zero')
gd=g[g.q!='202512'].copy(); c=gd.copy(); c['x_post']=c.sw*c.post
d,b,sc,Ainv=fit(c,'gap',['x_post'],['loan','qoth']); say(f'  December 2025 dropped: soft x post {b.x_post:+.3f} (dyadic SE {se_dyad(d,sc,Ainv)[0]:.3f}), n {len(d)}')
mf=p[(p.ddate=='2025-09-30')].groupby('family').fye.agg(lambda s: set(s.dropna().astype(int)))
decfam=set(mf[mf.apply(lambda s: s=={12})].index)
reg(g[g.fa.isin(decfam)&g.fc.isin(decfam)],'gap',['loan','qoth'],label='Both top-two managers December fiscal year-end only')
reg(g,'gap',['loan','qsep'],label='(row 1 for comparison)',show=False)
say('  Range across all frozen holders (row 1 FE), by number of holders at Sep 2025:')
for lab,cond in [('3',g.nm==3),('4-5',g.nm.between(4,5)),('6+',g.nm>=6),('all',g.nm>=3)]:
    reg(g[cond],'rng',['loan','qsep'],label=f'range, {lab} holders')

say('\n=== Persistence of gaps (row 2 sample) ===')
w2=g[g.q.isin(['202603','202606'])].pivot_table(index=['loan','sw'],columns='q',values='gap').dropna().reset_index()
for s_,nm in [(0.0,'other'),(1.0,'software')]:
    z=w2[(w2.sw==s_)&(w2['202603']>=1)]
    say(f'  {nm}: {len(z)} loans 1+ apart in Mar 2026; within 0.5 by Jun {np.mean(z["202606"]<0.5):.0%}; mean gap Mar {z["202603"].mean():.2f}, Jun {z["202606"].mean():.2f}')

# hand-check file: software loans with the largest gap at Jun 2026, both managers' Jun 2026 descriptions
jun=g[(g.q=='202606')&(g.sw==1)].sort_values('gap',ascending=False).head(40)
keys=x[['pkey','family','loan']].drop_duplicates()
q=p[p.pkey.isin(keys.pkey)&(p.ddate=='2026-06-30')].merge(keys,on=['pkey','family','loan'])
q['ident']=q.pos_id.map(ident)
rows=[]
for _,r in jun.iterrows():
    for fam,mk in [(r.family1,r.mark1),(r.family2,r.mark2)]:
        z=q[(q.loan==r.loan)&(q.family==fam)]
        for _,y in z.iterrows(): rows.append({'loan':r.loan,'gap':round(r.gap,2),'family':fam,'mgr_mark':round(mk,2),'pos_mark':round(y.mark,2),'principal':y.principal,'ident':y.ident})
pd.DataFrame(rows).to_csv(f'{V}/handcheck/h3_registered_top40_software_jun26{TAG}.csv',index=False)
g.to_parquet(f'{V}/h3_registered_panel{TAG}_2026-11-13.parquet')
open(OUT,'w').write('\n'.join(lines)+'\n')
