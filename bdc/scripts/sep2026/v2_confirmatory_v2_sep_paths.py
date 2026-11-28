# Registered confirmatory statistics C1, C1b and C2 (PAP v5 section 5, OSF 10.17605/OSF.IO/PSZ5D) for ONE quarterly change.
# v2 (16 Nov 2026, evening): fund label from filings to 30 Sep 2025 only; otherwise identical to v2_confirmatory_2026-11-16.py.
# Written 16 Nov 2026 so the September 2026 run is one command once the panel holds 30 Sep 2026 rows.
#
#   python3 v2_confirmatory_v2_2026-11-16.py            -> uses the last two quarter-ends in panel.parquet
#   python3 v2_confirmatory_v2_2026-11-16.py 2026-03-31 2026-06-30   -> dry run on quarters already seen
#
# This script does NOT open the hold-out lock and cannot: it only reads panel.parquet and nport_matches_coupon.parquet.
# September 2026 rows get into those files only through v2_load.py --allow-sep2026 with the OSF-POSTED.txt receipt.
# If the change ends on or before 30 June 2026 the output is labelled DRY RUN (exploratory, not confirmatory).
#
# C1 / C1b: c_soft,t = software's change in the mark / in Below99 from t0 to t1 relative to other loans in the same fund
#   (regression of the change on a software dummy with fund effects; positions held at both dates whose borrower has a
#   usable model label, excluding ITSV, DATA, HARD, UNKN, FUND; borrower-clustered).
# C2: dMark = a + b Soft + lambda G + dlambda Soft x G, G = P_t1 - Mark_t0, N-PORT match set fixed at Sep 2025 (verified
#   Level 2), pairs with a Level 2 price at both dates and |P_t0 - Mark_t0| <= 10; one-sided 5%, borrower-clustered.
# Fund samples: confirmatory (not on early-filer list A, no September fiscal year-end), all funds, confirmatory + BCRED,
#   confirmatory without list B. Label versions: model (confirmatory), funds' own, both agree.
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..','..')))  # repo root
import sys, numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy.stats import norm
V=(_ROOT+'/bdc/v2'); F=(_ROOT+'/bdc/industry_llm/full')
EF=(_ROOT+'/raw-sec/edgar-pull/early-filers-2026-11-08/early_filers_2026-11-08.csv')
import os; VD=os.environ['VD']; p=pd.read_parquet(f'{VD}/panel.parquet')
# 16 Nov 2026 (code audit): the funds' own label must use filings up to 30 Sep 2025 only (PAP v5). panel.sector is voted over
# all dates, so it is replaced here by the pre-shock vote built by 4-comments-and-reviews/2026-11-16_code-audit/scripts/a04_fund_label_pre.py.
_fl=pd.read_parquet(f'{V}/fund_label_pre_2026-11-16.parquet').set_index('pkey')
# 21 Nov 2026: fund_label_pre is keyed by the June panel's pkey; adding a quarter renumbers pkey. Map through pos_id (identical for
# rows to Jun 2026) to the old pkey, then carry the label along the new pkey chain to the Sep 2026 rows. The label itself is unchanged.
_old=pd.read_parquet(os.environ['OLDPANEL'],columns=['pos_id','pkey']).set_index('pos_id').pkey
_o=p.pos_id.map(_old); _ns=_o.map(_fl.sector_pre); _na=_o.map(_fl.agree_pre)
_t=pd.DataFrame({'pkey':p.pkey,'d':p.ddate,'s':_ns,'a':_na,'o':_o}).dropna(subset=['o']).sort_values('d').drop_duplicates('pkey',keep='last').set_index('pkey')
p['sector']=p.pkey.map(_t.s); p['agree']=p.pkey.map(_t.a)
qs=sorted(p.ddate.unique())
t0,t1=(pd.Timestamp(sys.argv[1]),pd.Timestamp(sys.argv[2])) if len(sys.argv)>2 else (pd.Timestamp(qs[-2]),pd.Timestamp(qs[-1]))
DRY=t1<=pd.Timestamp('2026-06-30'); base=pd.Timestamp('2025-09-30')
tag=f'{t0:%Y%m}_{t1:%Y%m}'; out=[]
def say(s): out.append(s); print(s,flush=True)
say(f'Registered statistics C1, C1b, C2 for the change {t0:%d %b %Y} -> {t1:%d %b %Y}. '+('DRY RUN on quarters already seen: exploratory, NOT confirmatory.' if DRY else 'CONFIRMATORY RUN (hold-out quarter).'))
L=pd.read_parquet(f'{F}/labels.parquet')[['bid','group']]
pb=pd.read_parquet(os.environ.get('PB',f'{F}/position_borrower.parquet')).merge(L,on='bid').merge(p[['pos_id','pkey']],on='pos_id').drop_duplicates('pkey').set_index('pkey')
p['grp']=p.pkey.map(pb.group); p['bid']=p.pkey.map(pb.bid); p['b99']=100*(p.mark<99)
# fund lists
d=pd.read_csv(EF,dtype=str); MANUAL_A={1803498}   # Blackstone Private Credit Fund, moved to list A by hand (PAP v5)
A={int(x) for x in d[d.reason.str.startswith('A')].cik}|MANUAL_A; B={int(x) for x in d[d.reason.str.startswith('B')].cik}-A
pos=pd.read_parquet(f'{VD}/positions.parquet',columns=['cik','ddate','form'])
fye=pos[pos.form.str.startswith('10-K',na=False)].groupby('cik').ddate.agg(lambda s:s.dt.month.mode().iloc[0])
sepfye={int(c) for c in fye[fye==9].index}
cik=p.cik.astype(int)
samples={'confirmatory (not list A, no Sep FYE)':~cik.isin(A|sepfye),'all funds':pd.Series(True,index=p.index),
         'confirmatory + Blackstone Private Credit Fund':~cik.isin((A-{1803498})|sepfye),'confirmatory without list B':~cik.isin(A|B|sepfye)}
ever_ok=set(p.loc[p.ok,'pkey'])
p['valid']=np.where(p.ddate<=base,p.ok,p.mark.between(0,110)&(p.principal>=250e3)&p.pkey.isin(ever_ok))
labels={'model labels':(p.grp.notna()&~p.grp.isin(['UNKN','FUND','ITSV','DATA','HARD']),p.grp=='SOFT'),
        'fund labels':(p.sector.notna()&~p.sector.isin(['it_services','tech_other']),p.sector=='software')}
labels['both labels agree']=(labels['model labels'][0]&labels['fund labels'][0]&(labels['model labels'][1]==labels['fund labels'][1]),p.grp=='SOFT')
p['clu']=np.where(p.bid.notna(),'b'+p.bid.astype(str),'l'+p.loan.astype(str))
def change(mask):
    x=p[p.valid&mask&p.ddate.isin([t0,t1])]
    w=x.pivot_table(index='pkey',columns='ddate',values=['mark','b99']); w=w.dropna()
    r=pd.DataFrame({'dmark':w[('mark',t1)]-w[('mark',t0)],'db99':w[('b99',t1)]-w[('b99',t0)]})
    return r.join(x[x.ddate==t0].set_index('pkey')[['cik','clu']])
def c_soft(r,y):
    r=r.copy();
    for c in [y,'sw']: r[c+'_d']=r[c]-r.groupby('cik')[c].transform('mean')
    m=smf.ols(f'{y}_d~sw_d-1',r).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(r.clu)[0]})
    return m.params.iloc[0],m.bse.iloc[0]
def region(v,lo,hi,up=True):
    # C1: supports if below +0.5, against if above +1.0. C1b: supports if above -5, against if below -10.
    if up: return 'SUPPORTS' if v<lo else ('COUNTS AGAINST' if v>hi else 'INCONCLUSIVE')
    return 'SUPPORTS' if v>hi else ('COUNTS AGAINST' if v<lo else 'INCONCLUSIVE')
say('\nC1 (mark) and C1b (Below99): software change relative to other loans in the same fund')
main={}
for ln,(lab,sw) in labels.items():
    for sn,sm in samples.items():
        r=change(lab&sm); r['sw']=sw.reindex(p.index)[p.pkey.drop_duplicates().index].set_axis(p.pkey.drop_duplicates().values).reindex(r.index).astype(float)
        r=r.dropna(subset=['sw'])
        if r.sw.sum()<10: say(f'  [{ln}] {sn}: too few software positions ({int(r.sw.sum())})'); continue
        b1,s1=c_soft(r,'dmark'); b2,s2=c_soft(r,'db99')
        say(f'  [{ln}] {sn}: n {len(r)}, software {int(r.sw.sum())}, funds {r.cik.nunique()} | C1 {b1:+.2f} (SE {s1:.2f}; 90% CI {b1-1.645*s1:+.2f}, {b1+1.645*s1:+.2f}) {region(b1,0.5,1.0)} | C1b {b2:+.1f} pp (SE {s2:.1f}; 90% CI {b2-1.645*s2:+.1f}, {b2+1.645*s2:+.1f}) {region(b2,-10,-5,up=False)}')
        if ln=='model labels' and sn.startswith('confirmatory (not'): main['C1']=b1; main['C1b']=b2
say('\nC2: relative partial adjustment on the N-PORT match set (fixed at Sep 2025, verified Level 2)')
M=pd.read_parquet(f'{V}/nport_matches_coupon.parquet'); M=M[M.verified&(M.lvl=='2')]; px=M.groupby('pos_id').price_median.median().rename('px')
lab,sw=labels['model labels']
keys=set(p.loc[(p.ddate==base)&p.ok&lab,'pkey'])
q=p[p.pkey.isin(keys)].join(px,on='pos_id'); q=q[q.mark.between(0,110)]
bs=set(q.loc[(q.ddate==base)&q.px.notna(),'pkey']); q=q[q.pkey.isin(bs)]
for sn,sm in samples.items():
    x=q[sm.reindex(q.index)&q.ddate.isin([t0,t1])]
    w=x.pivot_table(index='pkey',columns='ddate',values=['mark','px'])
    if (('px',t1) not in w.columns) or w[('px',t1)].notna().sum()==0: say(f'  {sn}: no N-PORT Level 2 prices at {t1:%Y-%m-%d} in the match file yet'); continue
    z=pd.DataFrame({'dm':w[('mark',t1)]-w[('mark',t0)],'g0':w[('px',t0)]-w[('mark',t0)],'gap':w[('px',t1)]-w[('mark',t0)],'same':(w[('px',t1)]-w[('mark',t1)]).abs()<0.01}).dropna()
    z=z.join(x.drop_duplicates('pkey').set_index('pkey')[['grp','bid']]); z['soft']=(z.grp=='SOFT').astype(int)
    for tn,zz in [('10-point trim (registered)',z[z.g0.abs()<=10]),('identical pairs dropped',z[(z.g0.abs()<=10)&~z.same]),('5-point trim',z[z.g0.abs()<=5]),('no trim',z)]:
        if zz.soft.sum()<10: say(f'  {sn} [{tn}]: too few software pairs ({int(zz.soft.sum())})'); continue
        m=smf.ols('dm~gap*soft',zz).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(zz.bid)[0]})
        b,s=m.params['gap:soft'],m.bse['gap:soft']; p1=norm.cdf(b/s)
        verdict='SUPPORTS' if (b<0 and p1<0.05) else ('INCONCLUSIVE' if b<0 else 'COUNTS AGAINST')
        say(f'  {sn} [{tn}]: software pairs {int(zz.soft.sum())} ({zz[zz.soft==1].bid.nunique()} borrowers), other {int((1-zz.soft).sum())} | lambda other {m.params.gap:.2f} | dlambda {b:+.2f} (SE {s:.2f}; one-sided p {p1:.3f}; 90% CI {b-1.645*s:+.2f}, {b+1.645*s:+.2f}) {verdict}')
        if sn.startswith('confirmatory (not') and tn.startswith('10'): main['C2']=verdict; main['C2n']=int(zz[zz.soft==1].grp.count())
say('\nReading (PAP v5): H2 is supported if C2 supports it and C1 does not count against it. If C1 and C1b fall in different columns, C1 decides. '
    'C2 needs at least 50 software loans with a Level 2 price at both dates; otherwise it is reported later as a dated deviation.')
if any(x in ('SUPPORTS','COUNTS AGAINST','INCONCLUSIVE') for x in main.values()) or main: say('Confirmatory-sample values: '+', '.join(f'{k} {v:+.2f}' if isinstance(v,float) else f'{k} {v}' for k,v in main.items()))
open(f'{VD}/results_confirmatory_{tag}_{"dryrun_" if DRY else ""}v2_2026-11-16.txt','w').write('\n'.join(out)+'\n')
