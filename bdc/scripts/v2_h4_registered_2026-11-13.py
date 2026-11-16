# Registered H4 (PAP v5, OSF 10.17605/OSF.IO/PSZ5D, section 7), run 13 Nov 2026 on quarters through June 2026 only.
# Sep 2026 hold-out lock stays closed (panel ends 30 June 2026; asserted below).
#
# Mark_mlt on S_m x Post_t x Soft_l, with S_m x Post and S_m x Soft (and S_m) separately, loan x date fixed effects,
# loans held by 2+ managers on a date. S_m = manager's share of BDC assets (fair value) in LISTED funds, fixed at 30 Sep 2025.
# Post = quarter-ends from 31 Mar 2026 (time zero Q1 2026); Dec 2025 gets its own term in a check.
# Unit = manager x loan x date, mark = principal-weighted mean across the manager's BDCs (as v2_h4_listed_share.py).
# Inference: SEs clustered by loan and two-way loan + manager; TOST against +-0.5; permutation of S_m across managers within
# Sep 2025 size terciles (counts). Pre-specified single check: fee rate, PIK share, audited Dec 2025 report (Sep 2025 values)
# x Post x Soft. Bank affiliation shown as estimates without a test.
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..')))  # repo root
import sys, warnings, numpy as np, pandas as pd
sys.path.insert(0,(_ROOT+'/bdc/scripts'))
from ai_rev2_lib import fit, se
warnings.filterwarnings('ignore')
V=(_ROOT+'/bdc/v2'); F=(_ROOT+'/bdc/industry_llm/full')
OUT=f'{V}/results_h4_registered_2026-11-13.txt'; lines=[]
def say(*a):
    s=' '.join(str(x) for x in a); print(s,flush=True); lines.append(s)

p=pd.read_parquet(f'{V}/panel.parquet')
assert p.ddate.max()<=pd.Timestamp('2026-06-30'), 'hold-out lock: panel must end at 30 June 2026'
pos=pd.read_parquet(f'{V}/positions.parquet',columns=['pos_id','form','cik','ddate'])
k=pos[pos.form.str.startswith('10-K',na=False)].groupby('cik').ddate.agg(lambda s: s.dt.month.mode().iloc[0])
p=p[p.ok & p.mark.between(0,110) & (p.principal>0)].copy()   # as v2_h4_listed_share.py (30 Oct run)
p['fv']=p.mark*p.principal/100

# ---- manager characteristics fixed at 30 Sep 2025 ----
s25=p[p.ddate=='2025-09-30']
M=s25.groupby('family').apply(lambda g: pd.Series({
    'S':g.fv[g.vtype=='listed'].sum()/g.fv.sum(),
    'S_perp':g.fv[g.vtype=='perpetual'].sum()/g.fv.sum(),
    'size':g.fv.sum(),
    'pik_share':(g.fv*g.pik).sum()/g.fv.sum(),               # PIK share of fair value (proxy for PIK income share)
    'dec_fye':np.average(g.cik.astype(float).map(k.rename(index=float)).fillna(0).eq(12),weights=g.fv),  # Dec FYE => Dec 2025 report audited
    'ptype':g.ptype.mode().iloc[0]})).reset_index()
M['S_mid']=M.S.between(1e-9,1-1e-9)
say('Registered H4 (PAP v5 section 7), run 13 Nov 2026 on quarters through 30 June 2026. Not confirmatory. Sep 2026 hold-out not used.')
say(f'Managers with a BDC at Sep 2025: {len(M)}; no listed BDC {int((M.S==0).sum())}; only listed {int((M.S>=1-1e-9).sum())}; in between {int(M.S_mid.sum())}')
say(f'  (PAP text: 83 managers, 45 / 16 / 22)')
M['tercile']=pd.qcut(M['size'].rank(method='first'),3,labels=False)

L=pd.read_parquet(f'{F}/labels.parquet')[['bid','group']]
gl=pd.read_parquet(f'{F}/position_borrower.parquet').merge(L,on='bid').merge(p[['pos_id','loan']],on='pos_id').dropna(subset=['loan']).drop_duplicates('loan').set_index('loan').group
p['grp']=p.loan.map(gl)
p['mp']=p.mark*p.principal
m=p.groupby(['loan','ddate','family']).agg(mp=('mp','sum'),pr=('principal','sum'),grp=('grp','first')).reset_index()
m['mark']=m.mp/m.pr
m=m.merge(M[['family','S','S_perp','tercile','pik_share','dec_fye','ptype']],on='family')    # managers present at Sep 2025 only
m=m[m.grp.notna() & ~m.grp.isin(['UNKN','FUND'])]
m['nfam']=m.groupby(['loan','ddate']).family.transform('size'); m=m[m.nfam>=2].copy()
m['sw']=(m.grp=='SOFT').astype(float); m['post']=(m.ddate>='2026-03-31').astype(float); m['dec']=(m.ddate=='2025-12-31').astype(float)
m['ld']=m.loan.astype(str)+'_'+m.ddate.dt.strftime('%Y%m')
say(f'Sample: manager-loan-date obs {len(m)}; loans {m.loan.nunique()}; managers {m.family.nunique()}; dates {m.ddate.min().date()} to {m.ddate.max().date()}; software share {m.sw.mean():.3f}')

def terms(d, var, extra_dec=False):
    d=d.copy(); d['x']=d[var]; d['x_post']=d.x*d.post; d['x_sw']=d.x*d.sw; d['x_post_sw']=d.x*d.post*d.sw
    X=['x','x_post','x_sw','x_post_sw']
    if extra_dec: d['x_dec']=d.x*d.dec; d['x_dec_sw']=d.x*d.dec*d.sw; X+=['x_dec','x_dec_sw']
    return d,X
def run(d, var, label, extra_dec=False, show=True):
    d,X=terms(d,var,extra_dec); dd,b,sc,Ainv=fit(d,'mark',X,['ld']); e1=se(dd,sc,Ainv,'loan'); e2=se(dd,sc,Ainv,('loan','family'))
    if show: say(f'  {label}: '+'; '.join(f'{c} {b[c]:+.3f} (SE loan {e1[i]:.3f}; loan+mgr {e2[i]:.3f})' for i,c in enumerate(X))+f' | n {len(dd)}, loans {dd.loan.nunique()}, managers {dd.family.nunique()}')
    return b, e2, dd

say('\n=== Main (listed share fixed at Sep 2025) ===')
b,e,dd=run(m,'S','Listed share S_m')
coef=b['x_post_sw']; sd=e[3]; lo,hi=coef-1.645*sd,coef+1.645*sd
say(f'  TOST against +-0.5 (two-way SE): 90% CI [{lo:+.3f}, {hi:+.3f}] -> {"equivalent to zero (|effect| < 0.5)" if (lo>-0.5 and hi<0.5) else "NOT shown to be within +-0.5"}')
from scipy.stats import norm
say(f'  one-sided p-values: H0 effect <= -0.5: {1-norm.cdf((coef+0.5)/sd):.3f}; H0 effect >= +0.5: {norm.cdf((coef-0.5)/sd):.3f}')
say(f'  Interpretation of size: a manager moving from 0 to 100% listed changes its relative software mark after the shock by {coef:+.2f} points')

say('\n=== Permutation of S_m across managers within Sep 2025 size terciles (2,000 draws, seed 20261113) ===')
dP,X=terms(m,'S'); dP=dP.copy()
from ai_prelim_fe import drop_singletons
dP=drop_singletons(dP.dropna(subset=['mark']),['ld'])
g=pd.factorize(dP.ld)[0]; cnt=np.bincount(g)
def dm(v): return v-(np.bincount(g,weights=v)/cnt)[g]
y=dm(dP.mark.to_numpy(float)); post=dP.post.to_numpy(float); sw=dP.sw.to_numpy(float)
fam=dP.family.to_numpy(); Sx=M.set_index('family').S; terc=M.set_index('family').tercile
def est(Smap):
    x=dP.family.map(Smap).to_numpy(float)
    Z=np.column_stack([dm(x),dm(x*post),dm(x*sw),dm(x*post*sw)])
    return np.linalg.lstsq(Z,y,rcond=None)[0][3]
b0=est(Sx); rng=np.random.default_rng(20261113); draws=[]
fams=M.family.values
for _ in range(2000):
    perm=Sx.copy()
    for t_ in range(3):
        idx=M.family[M.tercile==t_].values; perm.loc[idx]=rng.permutation(Sx.loc[idx].values)
    draws.append(est(perm))
draws=np.array(draws)
say(f'  observed {b0:+.3f}; draws with |b| >= |observed|: {int((np.abs(draws)>=abs(b0)).sum())} of 2000 (p = {(1+(np.abs(draws)>=abs(b0)).sum())/2001:.3f}); draws <= observed: {int((draws<=b0).sum())}')
say(f'  permutation distribution: 2.5% {np.percentile(draws,2.5):+.3f}, 97.5% {np.percentile(draws,97.5):+.3f}')

say('\n=== Checks ===')
run(m,'S','Dec 2025 own terms (post = Mar 2026 on)',extra_dec=True)
run(m[m.family.isin(M.family[M.S_mid])],'S','Only the managers with 0 < S < 1 (they identify the coefficient)')
run(m[m.ddate>='2024-09-30'],'S','Sample from Sep 2024')
run(m,'S_perp','Perpetual share fixed at Sep 2025 (Daniel-Fang-Nini channel, not registered as H4)')
# pre-specified single check: fee rate, PIK share, audit, each x Post x Soft, all together with S
d,X=terms(m,'S')
avail=[]
for c in ['pik_share','dec_fye']:
    d[f'{c}_post']=d[c]*d.post; d[f'{c}_sw']=d[c]*d.sw; d[f'{c}_psw']=d[c]*d.post*d.sw; avail+=[c,f'{c}_post',f'{c}_sw',f'{c}_psw']
dd,b,sc,Ainv=fit(d,'mark',X+avail,['ld']); e2=se(dd,sc,Ainv,('loan','family'))
cols=X+avail
say('  Pre-specified check (PIK share of FV and December-FYE share, Sep 2025 values, x Post x Soft; fee rates NOT in our data, left out): '
    +'; '.join(f'{c} {b[c]:+.3f} ({e2[i]:.3f})' for i,c in enumerate(cols) if c in ['x_post_sw','pik_share_psw','dec_fye_psw']))
say(f'  Dec-FYE share across managers: mean {M.dec_fye.mean():.2f}, managers below 1: {int((M.dec_fye<0.999).sum())}')

say('\n=== Bank affiliation (estimates only, no test) ===')
m['bank']=m.ptype.isin(['bank','bank_jv']).astype(float)
d=m.copy(); d['b_post']=d.bank*d.post; d['b_sw']=d.bank*d.sw; d['b_post_sw']=d.bank*d.post*d.sw
dd,b,sc,Ainv=fit(d,'mark',['bank','b_post','b_sw','b_post_sw'],['ld']); e2=se(dd,sc,Ainv,('loan','family'))
say(f'  bank-affiliated managers: {sorted(M.family[M.ptype.isin(["bank","bank_jv"])])}')
say('  '+'; '.join(f'{c} {b[c]:+.3f} ({e2[i]:.3f})' for i,c in enumerate(['bank','b_post','b_sw','b_post_sw'])))

say('\n=== Who identifies it: the in-between managers at Sep 2025 ===')
mid=M[M.S_mid].sort_values('size',ascending=False)
say(mid[['family','S','S_perp','size']].assign(size=lambda x:(x['size']/1e9).round(1)).round(2).to_string(index=False))
say('\n=== Leave one in-between manager out (main spec, coefficient on S x Post x Soft) ===')
out=[]
for fam in mid.family:
    b_,e_,_=run(m[m.family!=fam],'S','',show=False); out.append(f'{fam} {b_["x_post_sw"]:+.2f}')
say('  '+'; '.join(out))
open(OUT,'w').write('\n'.join(lines)+'\n')
