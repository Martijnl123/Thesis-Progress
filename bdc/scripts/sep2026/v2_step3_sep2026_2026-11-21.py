# Registered Step 3 prediction for 30 September 2026 (OSF Amendment 1, filed 18 Nov 2026, osf.io/psz5d).
# Written 21 Nov 2026, after the Sep 2026 pull, before any Sep 2026 statistic was computed.
#
#   python3 v2_step3_sep2026_2026-11-21.py <dir with panel.parquet built with Sep 2026 rows>
#
# Statistic (Amendment 1, verbatim in substance): change in the BDC mark from 30 Jun 2026 to 30 Sep 2026, software
# relative to other loans of the same class, fund fixed effects, borrower-clustered, positions of the same class held at
# both dates, in all funds of the June estimate whose report for 30 Sep 2026 is on EDGAR by 20 Nov 2026 (the EDGAR pull
# used --filed-by 2026-11-20, so every Sep 2026 row in the panel meets this). The same statistic on the 76 confirmatory
# funds of C1 is reported next to it. P = private estimate, Q = traded estimate, read on point estimates, 90% CI shown.
#   R1 Slow marks: P <= -0.5 and Q <= +1.0 | R2 Better loans: P > -0.5 and Q <= +1.0 | R3 Price overshoot: P > -0.5 and Q > +1.0
#   P <= -0.5 and Q > +1.0: fits none of the three.
# Classes: step3_classes_2026-11-17.parquet (SHA-256 b68346d6...), keyed here by the Sep 2025 pos_id, so the result does not
# depend on the panel's position key numbering. Funds' own labels (voted on filings to 30 Sep 2025) as robustness.
# The June numbers (-7.9 / -0.8 / -7.1) are recomputed first on the same panel as a check that the rebuild changed nothing.
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..','..')))  # repo root
import sys, hashlib, numpy as np, pandas as pd, statsmodels.formula.api as smf
VD=sys.argv[1]
V=(_ROOT+'/bdc/v2'); F=(_ROOT+'/bdc/industry_llm/full')
EF=(_ROOT+'/raw-sec/edgar-pull/early-filers-2026-11-08/early_filers_2026-11-08.csv')
CLS=f'{V}/step3_classes_2026-11-17.parquet'
assert hashlib.sha256(open(CLS,'rb').read()).hexdigest()=='b68346d65f2ee22bd0968835287683e3502829ef55244d97c06ec5ffdb552754'
BASE,JUN,SEP=pd.Timestamp('2025-09-30'),pd.Timestamp('2026-06-30'),pd.Timestamp('2026-09-30')
out=[]
def say(s): out.append(s); print(s,flush=True)
p=pd.read_parquet(f'{VD}/panel.parquet')
assert p.ddate.max()==SEP, 'panel has no 30 Sep 2026 rows'
CL=pd.read_parquet(CLS)
import os
if os.environ.get('POSMAP'):   # build A: June-panel pos_id -> this panel's pos_id (content match)
    _m=pd.read_parquet(os.environ['POSMAP']).set_index('pos_id_o').pos_id_n; n0=len(CL); CL=CL[CL.pos_id.isin(_m.index)].copy(); CL['pos_id']=CL.pos_id.map(_m); print(f'POSMAP: {n0-len(CL)} class rows have no content match in this build and are dropped')
# map classes to this panel's position key through the Sep 2025 pos_id; check the rows are the same positions
b=p[p.ddate==BASE].set_index('pos_id')
chk=b.reindex(CL.pos_id)
same=(chk.cik.values.astype(int)==CL.cik.values.astype(int))&np.isclose(chk.mark.values,CL.mark.values)&np.isclose(chk.principal.values,CL.principal.values)
assert same.all(), f'{(~same).sum()} class rows do not map to the same Sep 2025 position'
CL['pk']=chk.pkey.values
fl=pd.read_parquet(f'{V}/fund_label_pre_2026-11-16.parquet')   # keyed by the OLD pkey: map through the class file's old pkey
CL['fsec']=CL.pkey.map(fl.set_index('pkey').sector_pre)
fz=p[p.pkey.isin(set(CL.pk))]
fz=fz[(fz.ddate<=BASE)&fz.ok | (fz.ddate>BASE)&fz.mark.between(0,110)&(fz.principal>=250e3)]
w=fz.pivot_table(index='pkey',columns='ddate',values='mark')
s=CL.set_index('pk')
s['dJun']=(w[JUN]-w[BASE]).reindex(s.index); s['dSep']=(w[SEP]-w[JUN]).reindex(s.index)
s['clu']='b'+s.bid.astype(str); s['cik']=s.cik.astype(int)
# fund samples (same definitions as v2_confirmatory_v2_2026-11-16.py)
d=pd.read_csv(EF,dtype=str); A={int(x) for x in d[d.reason.str.startswith('A')].cik}|{1803498}
B_={int(x) for x in d[d.reason.str.startswith('B')].cik}-A
pos=pd.read_parquet(f'{VD}/positions.parquet',columns=['cik','ddate','form'])
fye=pos[pos.form.str.startswith('10-K',na=False)&(pos.ddate<=JUN)].groupby('cik').ddate.agg(lambda x:x.dt.month.mode().iloc[0])
sepfye={int(c) for c in fye[fye==9].index}
def ols(f,x): return smf.ols(f,x).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(x.clu)[0]})
def est(x,y,cls):
    z=x[(x.cls==cls)].dropna(subset=[y])
    r=ols(f'{y} ~ sw + C(cik)',z); return r.params['sw'],r.bse['sw'],len(z),int(z.sw.sum()),z.cik.nunique(),z[z.sw==1].bid.nunique()
def gap(x,y):
    u=x[x.cls.isin(['traded','private'])].dropna(subset=[y]).copy(); u['T_']=(u.cls=='traded').astype(int); u['Tsw']=u.T_*u.sw
    u['fs']=u.cik.astype(str)+'_'+u.sw.astype(str); r=ols(f'{y} ~ T_ + Tsw + C(fs)',u); return r.params['Tsw'],r.bse['Tsw'],len(u)
def fmt(v): b,se,n,ns,nf,nb=v; return f'{b:+.2f} (SE {se:.2f}; 90% CI {b-1.645*se:+.2f}, {b+1.645*se:+.2f}); n {n}, software {ns} ({nb} borrowers), funds {nf}'
def reading(P,Q):
    if P<=-0.5 and Q<=1.0: return 'R1 Slow marks'
    if P>-0.5 and Q<=1.0: return 'R2 Better loans'
    if P>-0.5 and Q>1.0: return 'R3 Price overshoot'
    return 'fits none of R1-R3'
say('Step 3 prediction for 30 Sep 2026 (OSF Amendment 1). CONFIRMATORY RUN of a registered exploratory-result test.')
say(f'Classes {CLS.split("/")[-1]} (hash verified); {len(CL)} positions mapped to this panel by Sep 2025 pos_id, all identical.')
say('\nCheck: June 2026 numbers on this panel (registered: traded -7.9 (1.3), private -0.8 (0.5), gap -7.1 (1.6))')
say('  traded  '+fmt(est(s,'dJun','traded'))); say('  private '+fmt(est(s,'dJun','private')))
g=gap(s,'dJun'); say(f'  gap, fund x sector FE {g[0]:+.2f} (SE {g[1]:.2f}), n {g[2]}')
june_funds=set(s.dropna(subset=['dJun']).loc[lambda x:x.cls.isin(['traded','private']),'cik'])
sep_funds=set(p.loc[p.ddate==SEP,'cik'].astype(int))
say(f'\nFunds in the June estimate: {len(june_funds)}; of these with a 30 Sep 2026 report filed by 20 Nov 2026 (rows in the pull): {len(june_funds&sep_funds)}')
say(f'  June-estimate funds without a Sep 2026 report in the pull: {len(june_funds-sep_funds)} (of which Sep fiscal year-end: {len((june_funds-sep_funds)&sepfye)})')
samples={'registered sample (June-estimate funds filed by 20 Nov)':june_funds&sep_funds,
         '76 confirmatory funds of C1 (not list A, no Sep FYE)':(june_funds&sep_funds)-A-sepfye,
         'confirmatory without list B':(june_funds&sep_funds)-A-B_-sepfye}
res={}
for lab in ['model labels','fund labels']:
    x=s.copy()
    if lab=='fund labels':
        x=x[x.fsec.notna()&~x.fsec.isin(['it_services','tech_other'])].copy(); x['sw']=(x.fsec=='software').astype(int)
    say(f'\n[{lab}] change 30 Jun 2026 -> 30 Sep 2026, software vs other loans of the same class, fund FE, borrower-clustered')
    for sn,fs in samples.items():
        xx=x[x.cik.isin(fs)]
        try:
            Q=est(xx,'dSep','traded'); P=est(xx,'dSep','private'); g=gap(xx,'dSep')
        except Exception as e:
            say(f'  {sn}: could not estimate ({e})'); continue
        say(f'  {sn}:'); say(f'    Q traded  '+fmt(Q)); say(f'    P private '+fmt(P))
        say(f'    gap traded minus private, fund x sector FE (context) {g[0]:+.2f} (SE {g[1]:.2f}), n {g[2]}')
        say(f'    reading: {reading(P[0],Q[0])}')
        res[(lab,sn)]=(P,Q)
    # descriptive: mean changes by class x sector, registered sample
    xx=x[x.cik.isin(samples['registered sample (June-estimate funds filed by 20 Nov)'])].dropna(subset=['dSep'])
    t=xx[xx.cls.isin(['traded','private'])].groupby(['cls','sw']).dSep.agg(['mean','median','size']).round(2)
    say('    mean / median change by class and software flag (registered sample):\n'+t.to_string())
P,Q=res[('model labels','registered sample (June-estimate funds filed by 20 Nov)')]
say(f'\nRegistered reading (model labels, registered sample): P {P[0]:+.2f}, Q {Q[0]:+.2f} -> {reading(P[0],Q[0])}')
say('Note: R2 is also what private marks that stay high for more than one quarter would show; one quarter cannot separate the two.')
say('The N-PORT price change on the traded loans (context next to Q) needs the Sep 2026 N-PORT pull (due ~30 Nov) and is not computed here.')
open(f'{VD}/results_step3_sep2026_2026-11-21.txt','w').write('\n'.join(out)+'\n')
