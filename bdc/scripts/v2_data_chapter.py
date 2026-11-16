# Every number in the data chapter (draft 6 Nov 2026, chapters v4), regenerated in one place.
# Output: bdc/v2/results_data_chapter_2026-11-06.txt and LaTeX tables in 3-writing/chapters/chapters-draft-2026-11-06-v4/tables/.
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..')))  # repo root
import os, re, numpy as np, pandas as pd
V=(_ROOT+'/bdc/v2'); F=(_ROOT+'/bdc/industry_llm/full')
T=(_ROOT+'/outputs/chapters-draft-2026-11-06-v4/tables'); os.makedirs(T,exist_ok=True)
out=[]
def say(*a):
    s=' '.join(str(x) for x in a); out.append(s); print(s, flush=True)
pos=pd.read_parquet(f'{V}/positions.parquet',columns=['pos_id','adsh','cik','ddate','form','filed','ident','maturity','principal','cost','fv','mark','spread','pik','acqdate','family'])
p=pd.read_parquet(f'{V}/panel.parquet')
# ---------- A. raw coverage ----------
say('A. SEC BDC data sets: debt rows', len(pos), '| filings', pos.adsh.nunique(), '| funds (CIK)', pos.cik.nunique(), '| managers', pos.family.nunique(),
    '| quarter-ends', pos.ddate.min().date(), 'to', pos.ddate.max().date(), '| quarter-ends used', pos.ddate.dt.month.isin([3,6,9,12]).mean().round(4))
# ---------- B. filter ----------
r=pos.cost/pos.principal
s1=r.between(0.90,1.05); s2=s1&pos.mark.between(20,110); s3=s2&(pos.principal>=250e3)
say('B. filter: all', len(pos), '| cost/principal 0.90-1.05', int(s1.sum()), '| and mark 20-110', int(s2.sum()), '| and principal>=250k', int(s3.sum()), '| panel ok', int(p.ok.sum()))
# ---------- C. labels and frozen cohort ----------
L=pd.read_parquet(f'{F}/labels.parquet')[['bid','group']]
pb=pd.read_parquet(f'{F}/position_borrower.parquet').merge(L,on='bid').merge(p[['pos_id','pkey']],on='pos_id').drop_duplicates('pkey').set_index('pkey')
p['grp']=p.pkey.map(pb.group); p['bid']=p.pkey.map(pb.bid)
sep=p[(p.ddate=='2025-09-30')&p.ok]
say('C. Sep 2025 clean positions', len(sep), '| with an LLM label', int(sep.grp.notna().sum()), '| labelled UNKN', int((sep.grp=='UNKN').sum()), '| FUND', int((sep.grp=='FUND').sum()),
    '| no label (borrower outside the 5,686 run)', int(sep.grp.isna().sum()))
lab=p.grp.notna()&~p.grp.isin(['UNKN','FUND'])
keys=set(p.loc[(p.ddate=='2025-09-30')&p.ok&lab,'pkey'])
c=p[(p.ddate=='2025-09-30')&p.pkey.isin(keys)].merge(pos[['pos_id','ident','maturity','fv','acqdate']],on='pos_id')
c['sw']=c.grp.eq('SOFT')
say('C. frozen cohort: positions', len(c), '| borrowers', c.bid.nunique(), '| funds', c.cik.nunique(), '| managers', c.family.nunique(), '| principal $bn', round(c.principal.sum()/1e9,1),
    '| share of all Sep 2025 clean principal', round(c.principal.sum()/sep.principal.sum(),3))
say('C. software: positions', int(c.sw.sum()), f'({c.sw.mean():.1%})', '| borrowers', c[c.sw].bid.nunique(), '| principal share', round(c[c.sw].principal.sum()/c.principal.sum(),3),
    '| fund-label software share of positions', round((c.sector=='software').mean(),3))
# ---------- D. fiscal year-ends (month of 10-K period) ----------
k=pos[pos.form.str.startswith('10-K',na=False)].groupby('cik').ddate.agg(lambda s: s.dt.month.mode().iloc[0])
fy=pd.Series(c.cik.unique()).map(k).fillna(0).astype(int).value_counts().rename({12:'Dec',9:'Sep',6:'Jun',3:'Mar',0:'no 10-K'})
say('D. cohort funds by fiscal year-end month:', fy.to_dict())
dec=pos[(pos.ddate=='2025-12-31')&pos.form.str.startswith('10-K',na=False)].drop_duplicates('cik')
say('D. Dec 2025 10-Ks: filed p10', dec.filed.quantile(.1).date(), 'median', dec.filed.median().date(), 'p90', dec.filed.quantile(.9).date(), '| n', len(dec))
# ---------- E. variables ----------
say('E. maturity tagged: share of cohort positions', round(c.maturity.notna().mean(),3), '| principal-weighted', round((c.principal*c.maturity.notna()).sum()/c.principal.sum(),3))
say('E. acquisition date present', round(c.acqdate.notna().mean(),3), '| spread present', round(c.spread.notna().mean(),3), '| PIK share', round(c.pik.mean(),3))
fvk=(c.fv%1000==0)
err=50000/c.principal   # max rounding error in points of par when fair value is rounded to $1,000
say('E. fair value a multiple of $1,000:', round(fvk.mean(),3), '| positions where rounding to $1,000 can move the mark by 0.1+ points:', round(((err>=0.1)&fvk).mean(),3),
    '| median principal $m', round(c.principal.median()/1e6,1))
# ---------- F. linking within fund over time ----------
nxt=set(p.loc[p.ddate=='2025-12-31','pkey']); filed_dec=set(p.loc[p.ddate=='2025-12-31','cik'])
cc=c[c.cik.isin(filed_dec)]
say('F. stable key: share of Sep 2025 cohort positions (funds that reported Dec 2025) still found at Dec 2025', round(cc.pkey.isin(nxt).mean(),3))
# ---------- G. cross-fund ----------
lm=p[p.ddate=='2025-09-30'].groupby('loan').family.nunique()
c['nman']=c.loan.map(lm)
say('G. cohort positions in loans held by 2+ managers', round((c.nman>=2).mean(),3), '| software', round((c[c.sw].nman>=2).mean(),3), '| other', round((c[~c.sw].nman>=2).mean(),3))
# H3 sample: loans with 3+ managers, every manager's (fund-averaged) mark >= 98 at Sep 2025, LLM labels
m=p[(p.ddate=='2025-09-30')&p.ok&lab].groupby(['loan','family']).agg(mark=('mark','mean'),grp=('grp','first')).reset_index()
g=m.groupby('loan').agg(n=('family','size'),mn=('mark','min'),grp=('grp','first'))
h3=g[(g.n>=3)&(g.mn>=98)]
say('G. H3 sample: loans', len(h3), '| software', int((h3.grp=='SOFT').sum()), '| exactly three managers', int((h3.n==3).sum()), '| software with three', int(((h3.n==3)&(h3.grp=='SOFT')).sum()))
# ---------- H. N-PORT ----------
M=pd.read_parquet(f'{V}/nport_matches.parquet'); Ms=M[M.ddate=='2025-09-30']
anym=set(Ms.pos_id); ver=set(Ms[Ms.verified].pos_id); l2=set(Ms[Ms.verified&(Ms.lvl=='2')].pos_id)
for nm,x in [('all',c),('software',c[c.sw]),('other',c[~c.sw])]:
    say(f'H. N-PORT at Sep 2025, cohort {nm}: any name match', round(x.pos_id.isin(anym).mean(),3), '| verified', round(x.pos_id.isin(ver).mean(),3), '| verified Level 2', round(x.pos_id.isin(l2).mean(),3))
s=c[c.sw&c.pos_id.isin(l2)]
say('H. software with verified Level 2 price at Sep 2025: positions', len(s), '| loans', s.loan.nunique(), '| borrowers', s.bid.nunique())
# ---------- I. attrition of the frozen cohort ----------
fz=p[p.pkey.isin(keys)]; post=fz.ddate>'2025-09-30'
d=fz[~post&fz.ok | post&fz.mark.between(0,110)&(fz.principal>=250e3)]
swk=set(c.loc[c.sw,'pkey'])
for q in ['2025-12-31','2026-03-31','2026-06-30']:
    rep=set(fz.loc[fz.ddate==q,'pkey']); ins=set(d.loc[d.ddate==q,'pkey'])
    say('I.',q[:7],'still reported', round(len(rep&keys)/len(keys),3), '| in H1 sample', round(len(ins&keys)/len(keys),3), '| software', round(len(ins&swk)/len(swk),3), '| other', round(len(ins&(keys-swk))/len(keys-swk),3))
# exits with a low last mark: share of cohort positions in the sample at t-1 that are gone at t with a last mark below 80
for q0,q1 in [('2025-09-30','2025-12-31'),('2025-12-31','2026-03-31'),('2026-03-31','2026-06-30')]:
    a=d[d.ddate==q0].set_index('pkey').mark; b=set(d.loc[d.ddate==q1,'pkey'])
    say('I. leave between',q0[:7],'and',q1[:7],': share of positions', round((~a.index.isin(b)).mean(),3), '| with last mark below 80', round(((~a.index.isin(b))&(a<80)).mean(),4))
# ---------- Table: cohort at Sep 2025 by group ----------
def col(x):
    return pd.Series({'Positions':len(x),'Borrowers':x.bid.nunique(),'Funds':x.cik.nunique(),'Principal (\\$bn)':x.principal.sum()/1e9,
        'Median principal (\\$m)':x.principal.median()/1e6,'Mark, mean':x.mark.mean(),'Below 99 (\\%)':100*(x.mark<99).mean(),
        'Spread (bp), mean':1e4*x.spread.mean(),'PIK (\\%)':100*x.pik.mean(),'Maturity tagged (\\%)':100*x.maturity.notna().mean(),
        'Held by 2+ managers (\\%)':100*(x.nman>=2).mean(),'Level 2 N-PORT price (\\%)':100*x.pos_id.isin(l2).mean()})
tab=pd.DataFrame({'Software':col(c[c.sw]),'Other':col(c[~c.sw]),'All':col(c)})
say('\nTable D1 (cohort at 30 Sep 2025):'); say(tab.round(2).to_string())
fmt=lambda v,r: f'{v:,.0f}' if r in ['Positions','Borrowers','Funds'] else f'{v:,.1f}'
lines=['\\begin{tabular}{lrrr}','\\toprule',' & Software & Other & All \\\\','\\midrule']
for r in tab.index: lines.append(f'{r} & '+' & '.join(fmt(tab.loc[r,k],r) for k in tab.columns)+' \\\\')
lines+=['\\bottomrule','\\end{tabular}']
open(f'{T}/tab_d1_cohort.tex','w').write('\n'.join(lines)+'\n')
open(f'{V}/results_data_chapter_2026-11-06.txt','w').write('\n'.join(out)+'\n')
