# Draft 1 tables (16 Nov 2026). Quarters to 30 June 2026 only (Sep 2026 hold-out lock untouched; exploratory).
# A. H1 headline event study (PAP v5 section 4 spec) under label versions and robustness variants -> one table.
# B. Scaled sector rank of PAP v5 section 4 through June 2026 (persistence check) and Fig 4.
# C. H2: event-time gap between BDC mark and N-PORT Level 2 price (theta_k), and delta-lambda without identical-price pairs.
# D. Traded vs private: share of marks left unchanged in the shock quarter, and the Level-3-only middle group.
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..')))  # repo root
import sys, warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,(_ROOT+'/bdc/scripts'))
from ai_rev2_lib import fit, se
warnings.filterwarnings('ignore')
V=(_ROOT+'/bdc/v2'); F=(_ROOT+'/bdc/industry_llm/full')
FIG=(_ROOT+'/outputs/figures-2026-11-16')
import os; os.makedirs(FIG,exist_ok=True)
base=pd.Timestamp('2025-09-30')
p=pd.read_parquet(f'{V}/panel.parquet'); assert p.ddate.max()<=pd.Timestamp('2026-06-30')
pos=pd.read_parquet(f'{V}/positions.parquet',columns=['pos_id','ident','maturity']); p=p.merge(pos,on='pos_id',how='left')
p['fq']=p.cik.astype(str)+'_'+p.ddate.dt.strftime('%Y%m'); p['mq']=p.family.astype(str)+'_'+p.ddate.dt.strftime('%Y%m')
p['vint']=p.pkey.map(p.groupby('pkey').ddate.min()).dt.strftime('%Y%m'); p['vq']=p.vint+'|'+p.ddate.dt.strftime('%Y%m')
L=pd.read_parquet(f'{F}/labels.parquet')[['bid','group']]
pb=pd.read_parquet(f'{F}/position_borrower.parquet').merge(L,on='bid').merge(p[['pos_id','pkey']],on='pos_id').drop_duplicates('pkey').set_index('pkey')
p['grp']=p.pkey.map(pb.group); p['bid']=p.pkey.map(pb.bid)
p['clu']=np.where(p.bid.notna(),'b'+p.bid.astype(str),'l'+p.loan.astype(str))
p['b99']=100*(p.mark<99)
s9all=p[(p.ddate==base)&p.ok].set_index('pkey')
p['mat9']=p.pkey.map(pd.to_datetime(s9all.maturity,errors='coerce')); p['ttm']=(p.mat9-base).dt.days/365.25
p.loc[~p.ttm.between(0,15),'ttm']=np.nan
p['prin9']=p.pkey.map(s9all.principal)
out=[]
def say(s): out.append(s); print(s,flush=True)

# ---------- A. H1 ----------
def cohort(mask_lab, sw):
    keys=set(p.loc[(p.ddate==base)&p.ok&mask_lab,'pkey'])
    fz=p[p.pkey.isin(keys)]; post=fz.ddate>base
    d=fz[(~post&fz.ok)|(post&fz.mark.between(0,110)&(fz.principal>=250e3))].copy(); d['sw']=sw.reindex(d.index).astype(float); return d
def es(d,fe=('pkey','fq','vq'),clu='clu',w=None,extra=None):
    d=d.copy(); qs=sorted(d.ddate.unique()); ev=[q for q in qs if q!=base]; X=[]
    for q in ev:
        k=q.strftime('%Y%m'); dq=(d.ddate==q).astype(float)
        d[f'sw{k}']=d.sw*dq; d[f'sp{k}']=d.spread_pre.fillna(0)*100*dq; d[f'spm{k}']=d.spread_pre.isna()*dq; d[f'pk{k}']=d.pik_pre.fillna(0)*dq
        X+= [f'sp{k}',f'spm{k}',f'pk{k}']
        if extra:
            d[f'ex{k}']=d[extra]*dq; X.append(f'ex{k}')
    SWC=[f'sw{q.strftime("%Y%m")}' for q in ev]; r={}
    for y in ['mark','b99']:
        dd,b,sc,Ainv=fit(d,y,SWC+X,list(fe),w=w); s=se(dd,sc,Ainv,clu)[:len(SWC)]
        r[y]=pd.DataFrame({'b':b[SWC].values,'se':s},index=[q.strftime('%Y-%m') for q in ev]); r['n']=(dd.pkey.nunique(),int(dd.drop_duplicates('pkey').sw.sum()),dd[clu if isinstance(clu,str) else clu[0]].nunique())
    return r
def row(nm,r):
    f=lambda y,q: f'{r[y].b[q]:+.2f} ({r[y].se[q]:.2f})'
    say(f'{nm} | pos {r["n"][0]} sw {r["n"][1]} clusters {r["n"][2]} | mark Dec {f("mark","2025-12")} Mar {f("mark","2026-03")} Jun {f("mark","2026-06")} | b99 Dec {f("b99","2025-12")} Mar {f("b99","2026-03")} Jun {f("b99","2026-06")}')
    return r
say('A. H1 event study, PAP v5 headline spec (position, fund x quarter, vintage x quarter FE; spread and PIK x quarter); SE by borrower unless stated')
labm=p.grp.notna()&~p.grp.isin(['UNKN','FUND']); swm=(p.grp=='SOFT')
d0=cohort(labm,swm); R={}
R['model']=row('1 model labels (headline)',es(d0))
labf=p.sector.notna()&~p.sector.isin(['it_services','tech_other']); swf=(p.sector=='software')
R['fund']=row('2 fund labels',es(cohort(labf,swf)))
ag=labm&labf&((swm&swf)|(~swm&~swf)); R['agree']=row('3 both labels agree',es(cohort(ag,swm)))
adj=labm&~p.grp.isin(['ITSV','DATA','HARD']); R['noadj']=row('4 model labels, IT services/data/hardware out of controls',es(cohort(adj,swm)))
dm=d0[d0.ttm.notna()]; R['mat0']=row('5a maturity-tagged subsample, no maturity control',es(dm)); R['mat1']=row('5b maturity-tagged subsample, time to maturity x quarter',es(dm,extra='ttm'))
d0['w']=d0.prin9; R['pw']=row('6 principal-weighted (Sep 2025 principal)',es(d0,w='w'))
R['mq']=row('7 manager x quarter instead of fund x quarter FE',es(d0,fe=('pkey','mq','vq')))
R['2w']=row('8 SE two-way borrower and manager',es(d0,clu=['clu','family']))
mx=d0[d0.ddate>base].groupby('pkey').principal.max(); grown=set(mx[mx>1.005*d0.drop_duplicates('pkey').set_index('pkey').prin9.reindex(mx.index)].index)
R['nopik']=row('9 principal not grown since Sep 2025 (PIK bound)',es(d0[~d0.pkey.isin(grown)]))
R['novq']=row('10 no vintage x quarter FE',es(d0,fe=('pkey','fq')))
bal=set(d0[d0.ddate==pd.Timestamp('2026-06-30')].pkey); R['bal']=row('11 positions still held in June 2026 only',es(d0[d0.pkey.isin(bal)]))
cls=pd.read_parquet(f'{V}/traded_untraded_sep25_2026-11-13.parquet').set_index('pkey').cls; d0['cls']=d0.pkey.map(cls)
R['trd']=row('12 loans with a Level 2 market price in Sep 2025 (traded) only',es(d0[d0.cls=='traded']))
R['ntr']=row('13 all other loans (no Level 2 price: private, Level 3 only, not matched)',es(d0[d0.cls.isin(['untraded','L3 only','not matched'])]))
R['prv']=row('14 private only (in match universe, borrower has no traded debt)',es(d0[d0.cls=='untraded']))
R['l3']=row('15 Level 3 only (a registered fund holds it, no Level 2 price)',es(d0[d0.cls=='L3 only']))
say('   full event path, model labels: mark '+'; '.join(f'{q} {x.b:+.2f} ({x.se:.2f})' for q,x in R['model']['mark'].iterrows()))
say('   full event path, model labels: b99 '+'; '.join(f'{q} {x.b:+.2f} ({x.se:.2f})' for q,x in R['model']['b99'].iterrows()))
say('   full event path, fund labels: mark '+'; '.join(f'{q} {x.b:+.2f} ({x.se:.2f})' for q,x in R['fund']['mark'].iterrows()))
say('   full event path, fund labels: b99 '+'; '.join(f'{q} {x.b:+.2f} ({x.se:.2f})' for q,x in R['fund']['b99'].iterrows()))
# raw means for the text
for nm,m in [('software',d0.sw==1),('other',d0.sw==0)]:
    g=d0[m].groupby(d0.ddate.dt.strftime('%Y-%m')); say(f'   raw {nm}: mark '+'; '.join(f'{q} {v:.2f}' for q,v in g.mark.mean().items() if q>='2025-06')+' | b99 '+'; '.join(f'{q} {v:.1f}' for q,v in g.b99.mean().items() if q>='2025-06'))
# within-position raw change Sep25->Mar26/Jun26
w=d0.pivot_table(index='pkey',columns=d0.ddate.dt.strftime('%Y%m'),values='mark'); sw0=d0.drop_duplicates('pkey').set_index('pkey').sw
for q in ['202603','202606']:
    ch=(w[q]-w['202509']).dropna(); say(f'   within-position mean change Sep25->{q}: software {ch[sw0.reindex(ch.index)==1].mean():+.2f} (median {ch[sw0.reindex(ch.index)==1].median():+.2f}, n {int((sw0.reindex(ch.index)==1).sum())}); other {ch[sw0.reindex(ch.index)==0].mean():+.2f} (median {ch[sw0.reindex(ch.index)==0].median():+.2f}, n {int((sw0.reindex(ch.index)==0).sum())})')

# ---------- B. scaled rank ----------
say('\nB. Scaled sector rank (PAP v5 section 4; persistence check), chain-linked changes, 23 groups, sum Dec 2025 to Jun 2026 over sd of 11 pre-shock changes')
ever_ok=set(p.loc[p.ok,'pkey'])
p['valid']=np.where(p.ddate<=base,p.ok,p.mark.between(0,110)&(p.principal>=250e3)&p.pkey.isin(ever_ok))
v=p[p.valid].sort_values(['pkey','ddate']).copy(); g=v.groupby('pkey'); v['dmark']=g.mark.diff(); v['db99']=g.b99.diff(); v['gapd']=g.ddate.diff().dt.days
ch=v[v.gapd.between(80,100)&v.grp.notna()&~v.grp.isin(['UNKN','FUND','ITSV','DATA','HARD'])].copy()
qs=sorted(ch.ddate.unique()); pre=[q for q in qs if q<=base]; postq=[q for q in qs if q>base]
assert len(pre)==11, len(pre)
def c_t(x,y,ind):
    s=ind-ind.groupby(x.cik).transform('mean'); e=x[y]-x.groupby('cik')[y].transform('mean'); den=(s*s).sum(); return (s*e).sum()/den if den>0 else np.nan
groups=sorted(ch.grp.unique()); rows=[]
for gname in groups:
    x=ch if gname=='SOFT' else ch[ch.grp!='SOFT']
    for y in ['dmark','db99']:
        c=pd.Series({q:c_t(x[x.ddate==q],y,(x[x.ddate==q].grp==gname).astype(float)) for q in qs})
        sd=c[pre].std(); rows.append(dict(grp=gname,y=y,sd=sd,post=c[postq].sum(),T=c[postq].sum()/sd,T_mar=c[postq[1:]].sum()/sd,T8=c[postq].sum()/c[pre[-8:]].std(),n=int((ch[ch.ddate==base].grp==gname).sum()),**{q.strftime('c%Y%m'):c[q] for q in postq}))
S=pd.DataFrame(rows); S.to_csv(f'{V}/scaled_rank_2026-11-16.csv',index=False)
for y,asc,lab in [('db99',False,'Below99 (largest first)'),('dmark',True,'mark (smallest first)')]:
    t=S[S.y==y].sort_values('T',ascending=asc).reset_index(drop=True); rk=int(t.index[t.grp=='SOFT'][0])+1
    say(f'  {lab}: software rank {rk} of {len(t)} (p = {rk/len(t):.3f}); T software {t["T"][t.grp=="SOFT"].iloc[0]:+.1f}; next {t.grp[1 if rk==1 else 0]} {t["T"][1 if rk==1 else 0]:+.1f}; software post sum {t.post[t.grp=="SOFT"].iloc[0]:+.2f}, sd {t.sd[t.grp=="SOFT"].iloc[0]:.2f}')
    say('     top 5: '+'; '.join(f'{r.grp} {r["T"]:+.1f}' for _,r in t.head(5).iterrows())+' | bottom 3: '+'; '.join(f'{r.grp} {r["T"]:+.1f}' for _,r in t.tail(3).iterrows()))
    for alt in ['T_mar','T8']:
        t2=S[S.y==y].sort_values(alt,ascending=asc).reset_index(drop=True); say(f'     {alt}: software rank {int(t2.index[t2.grp=="SOFT"][0])+1} of {len(t2)}')
    t3=S[S.y==y].assign(a=lambda z:z['T'].abs()).sort_values('a',ascending=False).reset_index(drop=True); say(f'     two-sided |T|: software rank {int(t3.index[t3.grp=="SOFT"][0])+1}; top 3 '+'; '.join(f'{r.grp} {r["T"]:+.1f}' for _,r in t3.head(3).iterrows()))
# Fig 4
SW,INK,MUTED,GRID,AXIS,OT='#eb6834','#0b0b0b','#898781','#e1e0d9','#c3c2b7','#3a6ea5'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.edgecolor':AXIS,'xtick.color':MUTED,'ytick.color':MUTED,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':GRID,'grid.linewidth':0.6,'axes.axisbelow':True})
NAMES={'SOFT':'Software','TELE':'Telecom','MEDI':'Media','MKTG':'Marketing','HCPS':'Health care services','HCEQ':'Health care equipment','PHAR':'Pharma and biotech','INSU':'Insurance','FINS':'Financial services','PROF':'Professional services','COMM':'Commercial services','EDUC':'Education','CONS':'Consumer services','CPRD':'Consumer products','FOOD':'Food and beverage','CAPG':'Capital goods','CONB':'Construction services','TRAN':'Transport','AUTO':'Automotive','CHEM':'Chemicals and materials','ENER':'Energy','DIST':'Distribution','REAL':'Real estate'}
fig,axs=plt.subplots(1,2,figsize=(7.4,4.6))
for ax,(y,ttl,asc) in zip(axs,[('db99','(a) Share below 99',True),('dmark','(b) Mark',False)]):
    t=S[S.y==y].sort_values('T',ascending=asc).reset_index(drop=True)
    ax.axvline(0,color=AXIS,lw=1); ax.scatter(t['T'],range(len(t)),s=22,color=[SW if gname=='SOFT' else OT for gname in t.grp],zorder=3)
    ax.set_yticks(range(len(t))); ax.set_yticklabels(t.grp.map(NAMES),fontsize=7); ax.set_title(ttl,fontsize=9,color=INK,loc='left'); ax.set_xlabel('change since Sep 2025 / sd of pre-shock quarterly changes',fontsize=7.5); ax.grid(axis='y',visible=False)
fig.tight_layout(); fig.savefig(f'{FIG}/fig4_scaled_rank.pdf',bbox_inches='tight'); fig.savefig(f'{FIG}/fig4_scaled_rank.png',dpi=200,bbox_inches='tight')

# ---------- C. H2 ----------
say('\nC. H2: BDC mark minus N-PORT Level 2 price of the same loan, positions with a verified Level 2 match at Sep 2025 (coupon-fixed matches)')
M=pd.read_parquet(f'{V}/nport_matches_coupon.parquet'); M=M[M.verified&(M.lvl=='2')]; px=M.groupby('pos_id').price_median.median().rename('px')
lab2=p.grp.notna()&~p.grp.isin(['UNKN','FUND','ITSV','DATA','HARD'])
keys=set(p.loc[(p.ddate==base)&p.ok&lab2,'pkey'])
q=p[p.pkey.isin(keys)].join(px,on='pos_id'); q=q[q.mark.between(0,110)]
bs=set(q.loc[(q.ddate==base)&q.px.notna(),'pkey']); q=q[q.pkey.isin(bs)&(q.ddate>='2025-06-30')].copy(); q['sw']=(q.grp=='SOFT').astype(float); q['gap']=q.mark-q.px
say(f'  match set at Sep 2025: {len(bs)} positions, {int(q[q.ddate==base].sw.sum())} software, {q[(q.ddate==base)&(q.sw==1)].bid.nunique()} software borrowers')
for nm,m in [('software',q.sw==1),('other',q.sw==0)]:
    gq=q[m&q.px.notna()].groupby(q.ddate.dt.strftime('%Y-%m')); say(f'  {nm}: mean mark '+'; '.join(f'{k} {x:.2f}' for k,x in gq.mark.mean().items())+' | mean price '+'; '.join(f'{k} {x:.2f}' for k,x in gq.px.mean().items())+' | n '+'; '.join(f'{k} {x}' for k,x in gq.size().items()))
def es2(d,y,fe):
    d=d.dropna(subset=[y]).copy(); ev=[x for x in sorted(d.ddate.unique()) if x!=base]; C=[]
    for x in ev: d['s'+x.strftime('%Y%m')]=d.sw*(d.ddate==x); C.append('s'+x.strftime('%Y%m'))
    dd,b,sc,Ainv=fit(d,y,C,fe); s=se(dd,sc,Ainv,'clu'); return '; '.join(f'{c[1:5]}-{c[5:]} {b[c]:+.2f} ({s[i]:.2f})' for i,c in enumerate(C))+f' | n {len(dd)}, positions {dd.pkey.nunique()}'
say('  theta_k (gap = mark - price; position + fund x quarter FE; SE borrower): '+es2(q,'gap',['pkey','fq']))
say('  theta_k, position + quarter FE: '+es2(q,'gap',['pkey','ddate']))
say('  same sample, outcome mark:  '+es2(q[q.px.notna()],'mark',['pkey','fq']))
say('  same sample, outcome price: '+es2(q,'px',['pkey','fq']))
qb=q[q.pkey.isin(set(q[q.px.notna()].groupby('pkey').ddate.nunique().loc[lambda z:z==5].index))]
say('  balanced (price at all five dates): '+es2(qb,'gap',['pkey','fq']))
say('  share of pairs with |mark - price| < 0.01: '+'; '.join(f'{k} soft {x[1.0]:.3f} other {x[0.0]:.3f}' for k,x in q[q.px.notna()].assign(same=lambda z:(z.gap.abs()<0.01)).groupby([q.ddate.dt.strftime('%Y-%m'),'sw']).same.mean().unstack().iterrows()))
say('  mean |mark - price|: '+'; '.join(f'{k} soft {x[1.0]:.2f} other {x[0.0]:.2f}' for k,x in q[q.px.notna()].assign(a=lambda z:z.gap.abs()).groupby([q.ddate.dt.strftime('%Y-%m'),'sw']).a.mean().unstack().iterrows()))
w=q.pivot_table(index='pkey',columns=q.ddate.dt.strftime('%Y%m'),values=['mark','px']); w.columns=[f'{a}_{b}' for a,b in w.columns]
w=w.join(q.drop_duplicates('pkey').set_index('pkey')[['sw','bid']])
for t0,t1 in [('202506','202509'),('202509','202512'),('202512','202603'),('202603','202606')]:
    x=pd.DataFrame({'dm':w[f'mark_{t1}']-w[f'mark_{t0}'],'g0':w[f'px_{t0}']-w[f'mark_{t0}'],'gap':w[f'px_{t1}']-w[f'mark_{t0}'],'soft':w.sw,'bid':w.bid,'same':(w[f'px_{t1}']-w[f'mark_{t1}']).abs()<0.01}).dropna()
    for nm,yy in [('trim 10',x[x.g0.abs()<=10]),('trim 10, identical pairs dropped',x[(x.g0.abs()<=10)&~x.same]),('trim 5',x[x.g0.abs()<=5]),('no trim',x)]:
        r=smf.ols('dm~gap*soft',yy).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(yy.bid)[0]})
        say(f'  dlambda {t0}->{t1} [{nm}]: n soft {int(yy.soft.sum())}, other {int((1-yy.soft).sum())} | lambda other {r.params.gap:.2f} ({r.bse.gap:.2f}) | dlambda {r.params["gap:soft"]:+.2f} ({r.bse["gap:soft"]:.2f})')
# pooled shock quarters
xs=[]
for t0,t1 in [('202509','202512'),('202512','202603'),('202603','202606')]:
    xs.append(pd.DataFrame({'dm':w[f'mark_{t1}']-w[f'mark_{t0}'],'g0':w[f'px_{t0}']-w[f'mark_{t0}'],'gap':w[f'px_{t1}']-w[f'mark_{t0}'],'soft':w.sw,'bid':w.bid,'t':t1}).dropna())
x=pd.concat(xs); x=x[x.g0.abs()<=10]; r=smf.ols('dm~gap*soft+C(t)',x).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(x.bid)[0]})
say(f'  pooled Sep25->Jun26, quarter dummies, trim 10: lambda other {r.params.gap:.2f} ({r.bse.gap:.2f}); dlambda {r.params["gap:soft"]:+.2f} ({r.bse["gap:soft"]:.2f}); n {len(x)}')

# ---------- D. traded vs private ----------
say('\nD. Traded vs private (classes from v2_traded_untraded_2026-11-13.py)')
T=pd.read_parquet(f'{V}/traded_untraded_sep25_2026-11-13.parquet')
wm=p[p.pkey.isin(set(T.pkey))&p.mark.between(0,110)].pivot_table(index='pkey',columns=p.ddate.dt.strftime('%Y%m'),values='mark')
T=T.set_index('pkey').join(wm.add_prefix('m_'))
for a,b in [('202509','202512'),('202512','202603'),('202603','202606')]:
    T[f'unch_{b}']=np.where(T[f'm_{a}'].notna()&T[f'm_{b}'].notna(),((T[f'm_{b}']-T[f'm_{a}']).abs()<0.05).astype(float),np.nan)
    T[f'd_{b}']=T[f'm_{b}']-T[f'm_{a}']
for cls in ['traded','untraded','L3 only','not matched']:
    for s,nm in [(1,'software'),(0,'other')]:
        x=T[(T.cls==cls)&(T.sw==s)]
        say(f'  {cls:12s} {nm:8s}: n {len(x)}; change Sep25->Jun26 {x["dm_2026-06"].mean():+.2f} (median {x["dm_2026-06"].median():+.2f}, n {x["dm_2026-06"].notna().sum()}); Mar26 {x["dm_2026-03"].mean():+.2f}; unchanged mark (|d|<0.05) Sep->Dec {x.unch_202512.mean():.2f}, Dec->Mar {x.unch_202603.mean():.2f}, Mar->Jun {x.unch_202606.mean():.2f}; sd of Dec->Mar change {x.d_202603.std():.2f}')
# borrower-level dispersion: do private software marks discriminate between borrowers?
for cls in ['traded','untraded']:
    x=T[(T.cls==cls)&(T.sw==1)&T['dm_2026-06'].notna()]; bm=x.groupby('bid')['dm_2026-06'].mean()
    say(f'  software {cls}: borrower-level change to Jun26: p10 {bm.quantile(.1):+.1f}, p25 {bm.quantile(.25):+.1f}, p50 {bm.median():+.1f}, p75 {bm.quantile(.75):+.1f}, p90 {bm.quantile(.9):+.1f}; borrowers {len(bm)}; share of borrowers down >10: {(bm<-10).mean():.2f}; share within 1 point of zero: {(bm.abs()<1).mean():.2f}')
# ---------- E. Merton illustration ----------
say('\nE. Merton (1974) illustration: change in loan value (points per 100 of loan value) when borrower asset value falls; r 4%, T 5y. Inputs are assumptions.')
from scipy.stats import norm
from scipy.optimize import brentq
def D(Vv,Fc,s,r=0.04,Tm=5.0):
    d1=(np.log(Vv/Fc)+(r+0.5*s*s)*Tm)/(s*np.sqrt(Tm)); d2=d1-s*np.sqrt(Tm); return Vv*norm.cdf(-d1)+Fc*np.exp(-r*Tm)*norm.cdf(d2)
for a in [0.15,0.30]:
    for s in [0.15,0.25,0.35]:
        parts=[]
        for Lv in [0.3,0.4,0.5,0.6]:
            Fc=brentq(lambda f:D(100,f,s)-100*Lv,1e-6,1e4); parts.append(f'LTV {int(Lv*100)}%: {100*(D(100*(1-a),Fc,s)-D(100,Fc,s))/(100*Lv):+.1f} (vol +10pt: {100*(D(100*(1-a),Fc,s+0.10)-D(100,Fc,s))/(100*Lv):+.1f})')
        say(f'  asset fall {int(a*100)}%, asset vol {int(s*100)}%: '+'; '.join(parts))
open(f'{V}/results_draft1_tables_2026-11-16.txt','w').write('\n'.join(out)+'\n')
