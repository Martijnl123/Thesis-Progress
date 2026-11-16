# Traded vs untraded software loans (13 Nov 2026). Data through 30 Jun 2026 only; no Sep 2026 data is read.
# Question: traded software loans (Level 2 N-PORT price at Sep 2025) fell ~10 points by Jun 2026, untraded ones ~4.
# Is that selection (different loans) or slower marks on untraded loans?
# Registered part (PAP v5 sec. 5, benchmark (a)): traded vs untraded software compared on spread, PIK, maturity at Sep 2025.
# Everything else here is exploratory.
# Fixes vs Fig 5b: (1) "untraded" only among positions that entered the N-PORT match universe (fund sector not IT services /
# other tech); before, never-matched positions counted as untraded. (2) traded is defined at LOAN level (any BDC position of
# the loan has a verified Level 2 match), so a failed name match at one BDC does not make the loan "untraded".
# (3) uses nport_matches_coupon.parquet (coupon read from the description, registered pre-step of 8 Nov).
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..')))  # repo root
import sys, warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
V=(_ROOT+'/bdc/v2'); F=(_ROOT+'/bdc/industry_llm/full')
OUT=f'{V}/results_traded_untraded_2026-11-13.txt'
out=[]
def say(*a):
    s=' '.join(str(x) for x in a); print(s, flush=True); out.append(s)
p=pd.read_parquet(f'{V}/panel.parquet'); assert p.ddate.max()<=pd.Timestamp('2026-06-30')
P=pd.read_parquet(f'{V}/positions.parquet',columns=['pos_id','maturity','first_lien','second_lien','nonaccrual','ident'])
p=p.merge(P,on='pos_id',how='left')
L=pd.read_parquet(f'{F}/labels.parquet')[['bid','group']]
pb=pd.read_parquet(f'{F}/position_borrower.parquet').merge(L,on='bid').merge(p[['pos_id','pkey']],on='pos_id').drop_duplicates('pkey').set_index('pkey')
p['grp']=p.pkey.map(pb.group); p['bid']=p.pkey.map(pb.bid)
lab=p.grp.notna()&~p.grp.isin(['UNKN','FUND'])
keys=set(p.loc[(p.ddate=='2025-09-30')&p.ok&lab,'pkey'])
fz=p[p.pkey.isin(keys)&lab]; post=fz.ddate>'2025-09-30'
d=fz[~post&fz.ok | post&fz.mark.between(0,110)&(fz.principal>=250e3)].copy()
d['sw']=d.grp.eq('SOFT').astype(int); d['q']=d.ddate.dt.strftime('%Y-%m')
# ---------- traded status at Sep 2025 ----------
M=pd.read_parquet(f'{V}/nport_matches_coupon.parquet'); assert M.ddate.max()<=pd.Timestamp('2026-06-30')
M=M[M.verified&(M.ddate=='2025-09-30')]
s=d[d.q=='2025-09'].copy()
s['univ']=s.ok&s.sector.notna()&~s.sector.isin(['it_services','tech_other'])
s['l2pos']=s.pos_id.isin(set(M.loc[M.lvl=='2','pos_id'])); s['anypos']=s.pos_id.isin(set(M.pos_id))
lt=s.groupby('loan').agg(l2=('l2pos','any'),anyf=('anypos','any'),univ=('univ','any'))
s['T']=s.loan.map(lt.l2); s['anyf']=s.loan.map(lt.anyf); s['luniv']=s.loan.map(lt.univ)
s['cls']=np.select([s['T'],s.anyf,s.luniv],['traded','L3 only','untraded'],'not matched')
# Borrower-level fix (hand-check 13 Nov, v2/handcheck/traded_untraded_borrower_verdicts_2026-11-13.csv): an "untraded" loan
# whose borrower has a traded loan (automatic: any loan of the borrower is traded; by hand: name search of N-PORT Level 2
# issuers for every private software borrower) is put in its own class. "untraded" from here on = private borrower.
HV=pd.read_csv(f'{V}/handcheck/traded_untraded_borrower_verdicts_2026-11-13.csv')
tb=set(s.loc[s['T'],'bid'])|set(HV.loc[HV.verdict=='traded_borrower','bid']); unc=set(HV.loc[HV.verdict=='unclear','bid'])
s.loc[(s.cls=='untraded')&s.bid.isin(tb),'cls']='traded borrower'; s.loc[(s.cls=='untraded')&s.bid.isin(unc),'cls']='unclear'
say('Traded vs untraded software loans, frozen Sep 2025 cohort, LLM labels, data to Jun 2026. Script v2_traded_untraded_2026-11-13.py')
say('\nA. Classes at Sep 2025 (loan level: traded = any position of the loan has a verified Level 2 N-PORT price)')
say(pd.crosstab(s.cls,s.sw.map({1:'software',0:'other'}),margins=True).to_string())
say('Positions whose own match failed but loan is traded (loan-level fix):', int((s['T']&~s.l2pos).sum()))
say('Old Fig 5b "untraded" software positions outside the match universe (never matched):',int(((s.sw==1)&~s.anypos&~s.univ).sum()), 'of', int(((s.sw==1)&~s.anypos).sum()))
# ---------- covariates at Sep 2025 ----------
s['ttm']=(s.maturity-pd.Timestamp('2025-09-30')).dt.days/365.25
s['lnprin']=np.log(s.principal.clip(lower=1))
bsz=s.groupby('bid').principal.sum(); s['ln_bsize']=np.log(s.bid.map(bsz))
s['nfam']=s.bid.map(s.groupby('bid').family.nunique())
s['spr']=s.spread_pre*100; s['pikp']=s.pik_pre.fillna(0)
s['par']=s.mark.between(99,101).astype(int); s['b99']=(s.mark<99).astype(int)
s['fl']=s.first_lien.fillna(0); s['sl']=s.second_lien.fillna(0); s['na']=s.nonaccrual.fillna(0)
s['listed']=s.vtype.eq('listed').astype(int); s['perp']=s.vtype.eq('perpetual').astype(int)
s['ttm_m']=s.ttm.isna().astype(int); s['ttm_f']=s.ttm.fillna(s.ttm.median())
s['spr_m']=s.spr.isna().astype(int); s['spr_f']=s.spr.fillna(s.spr.median())
cov=['mark','b99','par','spr','pikp','ttm','lnprin','ln_bsize','nfam','fl','sl','na','listed','perp']
say('\nB. Sep 2025 characteristics, software positions (registered: spread, PIK, maturity). Means; n non-missing')
tab=[]
for c in cov:
    r={'var':c}
    for k in ['traded','untraded','traded borrower','L3 only','not matched']:
        x=s.loc[(s.sw==1)&(s.cls==k),c]; r[k]=f'{x.mean():.3f} ({x.notna().sum()})'
    a=s.loc[(s.sw==1)&(s.cls=='traded'),c]; b=s.loc[(s.sw==1)&(s.cls=='untraded'),c]
    r['std diff T-U']=f'{(a.mean()-b.mean())/np.sqrt((a.var()+b.var())/2):+.2f}'
    tab.append(r)
say(pd.DataFrame(tab).to_string(index=False))
say('Same for other loans (traded vs untraded), standardised difference only:')
say('  '+', '.join(f"{c} {((s.loc[(s.sw==0)&(s.cls=='traded'),c].mean()-s.loc[(s.sw==0)&(s.cls=='untraded'),c].mean())/np.sqrt((s.loc[(s.sw==0)&(s.cls=='traded'),c].var()+s.loc[(s.sw==0)&(s.cls=='untraded'),c].var())/2)):+.2f}" for c in cov))
# ---------- outcomes ----------
w=d.pivot_table(index='pkey',columns='q',values='mark')
for q in ['2025-03','2025-06','2025-12','2026-03','2026-06']:
    s[f'dm_{q}']=s.pkey.map(w[q]-w['2025-09']) if q>'2025-09' else s.pkey.map(w['2025-09']-w[q])
    s[f'db_{q}']=100*(s.pkey.map((w[q]<99).astype(float).where(w[q].notna()))-s.b99) if q>'2025-09' else np.nan
s['held_jun']=s.pkey.map(w['2026-06']).notna()
s['T_']=s['T'].astype(int); s['cik']=s.cik.astype(str)
u=s[s.cls.isin(['traded','untraded'])].copy(); u3=s[s.cls.isin(['traded','untraded','traded borrower'])]
say('\nC. Raw mark change since Sep 2025 (positions held at the date). Pre-period rows: change TO Sep 2025.')
for q in ['2025-03','2025-06','2025-12','2026-03','2026-06']:
    g=u3.groupby(['sw','cls'])[f'dm_{q}'].agg(['mean','count']).round(2)
    say(f'  {q}: '+'; '.join(f"{'soft' if a else 'other'} {b} {r['mean']:+.2f} (n={int(r['count'])})" for (a,b),r in g.iterrows()))
say('  Medians, Jun 2026: '+'; '.join(f"{'soft' if a else 'other'} {b} {v:+.2f}" for (a,b),v in u3.groupby(['sw','cls'])['dm_2026-06'].median().items()))
say('  Share fallen more than 5 points by Jun 2026: '+'; '.join(f"{'soft' if a else 'other'} {b} {v:.3f}" for (a,b),v in u3.dropna(subset=['dm_2026-06']).groupby(['sw','cls'])['dm_2026-06'].apply(lambda v:(v<-5).mean()).items()))
say('  Share still held at Jun 2026: '+'; '.join(f"{'soft' if a else 'other'} {b} {v:.3f}" for (a,b),v in u.groupby(['sw','cls']).held_jun.mean().items()))
say('  Below99 change to Jun 2026 (pp): '+'; '.join(f"{'soft' if a else 'other'} {b} {v:+.1f}" for (a,b),v in u.groupby(['sw','cls'])['db_2026-06'].mean().items()))
ctrl='spr_f + spr_m + pikp + ttm_f + ttm_m + lnprin + ln_bsize + nfam + fl + sl + mark'
def fit(f,data):
    import re
    used=[c for c in data.columns if re.search(r'(?<![A-Za-z_])'+re.escape(c)+r'(?![A-Za-z_0-9])',f)]
    data=data.dropna(subset=used)
    r=smf.ols(f,data).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(data.bid)[0]})
    return r,len(data)
def row(r,n,terms):
    return '; '.join(f'{t} {r.params[t]:+.2f} ({r.bse[t]:.2f})' for t in terms)+f'; n={n}, borrowers clustered'
for c in ['spr_f','spr_m','pikp','ttm_f','ttm_m','lnprin','ln_bsize','nfam','fl','sl','mark']: u[c]=u[c]-u[c].mean()
say('\nD0. Software only, mark change to Jun 2026: raw gap, then vehicle type FE, then fund FE')
x=u[(u.sw==1)].dropna(subset=['dm_2026-06'])
for fe in ['1','C(vtype)','C(cik)']:
    r,n=fit(f'Q("dm_2026-06") ~ T_ + {fe}',x); say(f'  FE {fe}: '+row(r,n,['T_']))
say('  Raw gap by vehicle type: '+'; '.join(f"{v}: T {x[(x.vtype==v)&(x.T_==1)]['dm_2026-06'].mean():+.2f} (n={((x.vtype==v)&(x.T_==1)).sum()}) U {x[(x.vtype==v)&(x.T_==0)]['dm_2026-06'].mean():+.2f} (n={((x.vtype==v)&(x.T_==0)).sum()})" for v in ['listed','perpetual','private']))
say('\nD. Regressions, outcome = mark change Sep 2025 -> Jun 2026, positions held at Jun 2026, fund FE, SE by borrower')
for y,lab_ in [('dm_2026-06','mark'),('dm_2026-03','mark to Mar 2026'),('db_2026-06','Below99 pp')]:
    x=u.dropna(subset=[y]); sw=x[x.sw==1]
    r,n=fit(f'Q("{y}") ~ T_ + C(cik)',sw); say(f'  [{lab_}] software, raw:            '+row(r,n,['T_']))
    r,n=fit(f'Q("{y}") ~ T_ + {ctrl} + C(cik)',sw); say(f'  [{lab_}] software, controls:       '+row(r,n,['T_']))
    r,n=fit(f'Q("{y}") ~ T_*sw + C(cik)',x); say(f'  [{lab_}] all, T x software:        '+row(r,n,['T_','sw','T_:sw']))
    r,n=fit(f'Q("{y}") ~ T_*sw + ({ctrl})*sw + C(cik)',x); say(f'  [{lab_}] all, + controls x sw:     '+row(r,n,['T_','sw','T_:sw']))
say('\nD1. Fund x sector FE (traded vs private within the same fund and sector group; T x soft = software-specific gap). Outcome variants')
u['fs']=u.cik+'_'+u.sw.astype(str)
for q in ['2026-03','2026-06']:
    y=f'dm_{q}'; x=u.dropna(subset=[y]).copy(); lo,hi=x[y].quantile([.05,.95]); x['yw5']=x[y].clip(lo,hi); x['f5']=100*(x[y]<-5)
    for yy,nm in [(y,'mark'),('yw5','mark winsorised 5/95'),('f5','fell >5 pts (pp)')]:
        r,n=fit(f'Q("{yy}") ~ T_ + T_:sw + C(fs)',x); say(f'  {q} {nm}: '+row(r,n,['T_','T_:sw']))
say('\nE. Pre-trend: same regressions on the change TO Sep 2025 (software, fund FE, controls)')
for y in ['dm_2025-03','dm_2025-06']:
    sw=u[(u.sw==1)].dropna(subset=[y]); r,n=fit(f'Q("{y}") ~ T_ + {ctrl} + C(cik)',sw); say(f'  {y}: '+row(r,n,['T_']))
# ---------- reweighting ----------
say('\nF. Untraded software reweighted to look like traded software on Sep 2025 observables (logit propensity, ATT weights)')
sw=u[(u.sw==1)].dropna(subset=['dm_2026-06']).copy()
lg=smf.logit(f'T_ ~ {ctrl} + listed + perp',sw).fit(disp=0); ps=lg.predict(sw).clip(0.01,0.99)
sw['wt']=np.where(sw.T_==1,1,ps/(1-ps))
for k,nm in [(1,'traded'),(0,'untraded')]:
    x=sw[sw.T_==k]; say(f'  {nm}: unweighted {x["dm_2026-06"].mean():+.2f}, weighted {np.average(x["dm_2026-06"],weights=x.wt):+.2f}, n={len(x)}, eff. n={x.wt.sum()**2/(x.wt**2).sum():.0f}')
say('  Overlap: propensity range traded '+f'{ps[sw.T_==1].min():.2f}-{ps[sw.T_==1].max():.2f}, untraded {ps[sw.T_==0].min():.2f}-{ps[sw.T_==0].max():.2f}; pseudo-R2 {lg.prsquared:.2f}')
x=sw[sw.T_==0]; say('  Weighted untraded balance (std diff vs traded): '+', '.join(f"{c} {(np.average(x[c],weights=x.wt)-sw.loc[sw.T_==1,c].mean())/sw[c].std():+.2f}" for c in ['mark','spr_f','pikp','ttm_f','lnprin','ln_bsize','sl']))
# ---------- overlap: compare within bins of the variables that separate the groups ----------
say('\nG. Software, mark change to Jun 2026, traded vs private within bins (raw means; n)')
x=s[(s.sw==1)&s.cls.isin(['traded','untraded'])].dropna(subset=['dm_2026-06']).copy()
x['nfam_b']=pd.cut(x.nfam,[0,1,3,6,100],labels=['1','2-3','4-6','7+'])
x['size_b']=pd.qcut(x.ln_bsize,3,labels=['small','mid','large'])
x['spr_b']=pd.cut(x.spr,[0,4,5,6,20],labels=['<4','4-5','5-6','6+'])
x['mk_b']=pd.cut(x.mark,[0,97,99,101,200],labels=['<97','97-99','99-101','>101'])
for b in ['nfam_b','size_b','spr_b','mk_b','vtype']:
    g=x.groupby([b,'cls'])['dm_2026-06'].agg(['mean','median','size']).unstack('cls')
    say(f'  by {b}: '+'; '.join(f"{lv}: T {g.loc[lv,('mean','traded')]:+.2f}/{g.loc[lv,('median','traded')]:+.2f} (n={int(g.loc[lv,('size','traded')]) if g.loc[lv,('size','traded')]==g.loc[lv,('size','traded')] else 0}) U {g.loc[lv,('mean','untraded')]:+.2f}/{g.loc[lv,('median','untraded')]:+.2f} (n={int(g.loc[lv,('size','untraded')]) if g.loc[lv,('size','untraded')]==g.loc[lv,('size','untraded')] else 0})" for lv in g.index)+'   [mean/median]')
say('\nH. Borrower-level means (each borrower weighted once), software, Jun 2026: '+'; '.join(f"{k} {v['mean']:+.2f} median {v['median']:+.2f} (borrowers {int(v['size'])})" for k,v in x.groupby(['cls','bid'])['dm_2026-06'].mean().groupby('cls').agg(['mean','median','size']).iterrows()))
say('Catch-up check, software, change Mar -> Jun 2026 on positions held at both: '+'; '.join(f"{k} {v:+.2f}" for k,v in x.dropna(subset=['dm_2026-03']).assign(dq=lambda z:z['dm_2026-06']-z['dm_2026-03']).groupby('cls').dq.mean().items()))
pr=x[x.cls=='untraded']; r,n=fit(f'Q("dm_2026-06") ~ pikp + spr_f + spr_m + mark + nfam + ln_bsize + C(cik)',pr)
say('Within private software loans (fund FE): '+row(r,n,['pikp','spr_f','mark','nfam','ln_bsize']))
# AI exposure (model label, 0-3, judged from the business model in mid-2025) and whether the model knew the firm
LL=pd.read_parquet(f'{F}/labels.parquet').drop_duplicates('bid').set_index('bid')
x['aix']=x.bid.map(LL.ai_exposure); x['known']=x.bid.map(LL.known).astype(str)
say('\nI. AI exposure label: share 3 (very high) traded '+f"{(x[x.cls=='traded'].aix==3).mean():.2f} vs private {(x[x.cls=='untraded'].aix==3).mean():.2f}; mean {x[x.cls=='traded'].aix.mean():.2f} vs {x[x.cls=='untraded'].aix.mean():.2f}")
g=x.groupby(['aix','cls'])['dm_2026-06'].agg(['mean','size']).unstack('cls')
say('  by AI exposure: '+'; '.join(f"{lv}: T {g.loc[lv,('mean','traded')]:+.2f} (n={g.loc[lv,('size','traded')]:.0f}) U {g.loc[lv,('mean','untraded')]:+.2f} (n={g.loc[lv,('size','untraded')]:.0f})" for lv in g.index))
x['cik']=x.cik.astype(str)
r,n=fit(f'Q("dm_2026-06") ~ T_ + C(aix) + C(known) + {ctrl} + C(cik)',x); say('  fund FE + controls + AI exposure + known dummies: '+row(r,n,['T_']))
s.to_parquet(f'{V}/traded_untraded_sep25_2026-11-13.parquet',index=False)
open(OUT,'w').write('\n'.join(out)+'\n'); print('written',OUT)
