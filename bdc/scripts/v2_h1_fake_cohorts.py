# Fake cohorts for H1 (29 Oct): freeze the positions held at Sep 2023 and Sep 2024,
# follow them forward three quarters with a fake shock in Q1 of the next year (December stays pre), and compare with the real
# Sep 2025 cohort run the same way. Uses the funds' own industry labels: the LLM labels only cover borrowers still held in
# Sep 2025, so they would select earlier cohorts on survival. Pre-shock spread and PIK are taken at each cohort's base quarter.
# Event time (base quarter omitted) shows whether a backward-followed cohort drifts up before its base date (survivorship).
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..')))  # repo root
import sys, warnings, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0,(_ROOT+'/bdc/scripts'))
from ai_rev2_lib import fit, se, vcov
warnings.filterwarnings('ignore')
V=(_ROOT+'/bdc/v2')
p=pd.read_parquet(f'{V}/panel.parquet')
p['fq']=p.cik.astype(str)+'_'+p.ddate.dt.strftime('%Y%m')
lab=p.sector.notna()&~p.sector.isin(['it_services','tech_other'])
qs=sorted(p.ddate.unique())
C=['sp_post','spm_post','pik_post']
def cohort(bq,label):
    bq=pd.Timestamp(bq); i=qs.index(bq); after=qs[i+1:i+4]; shock=after[1]   # Dec = after[0] stays pre; fake/real shock from Mar
    base=p[(p.ddate==bq)&p.ok&lab]; keys=set(base.pkey)
    sp=base.set_index('pkey').spread; pk=base.set_index('pkey').pik
    fz=p[p.pkey.isin(keys)&lab&(p.ddate<=after[-1])]; post=fz.ddate>bq
    d=fz[~post&fz.ok | post&fz.mark.between(0,110)&(fz.principal>=250e3)].copy()
    d['sw']=(d.sector=='software').astype(float)
    d['post']=(d.ddate>=shock).astype(float)
    d['sp_b']=d.pkey.map(sp); d['pk_b']=d.pkey.map(pk).fillna(0)
    d['sp_post']=d.sp_b.fillna(0)*100*d.post; d['spm_post']=d.sp_b.isna()*d.post; d['pik_post']=d.pk_b*d.post
    d['sw_post']=d.sw*d.post; d['b99']=100*(d.mark<99)
    d['fvc']=d.mark*d.principal/d.cost; d.loc[~d.fvc.between(0,150),'fvc']=np.nan
    held=[d[d.ddate==q].pkey.nunique()/len(keys) for q in after]
    print(f'\n=== {label}: base {bq:%b %Y}, {len(keys)} positions ({base.cik.nunique()} funds), software share {base.sector.eq("software").mean():.3f}; '
          f'shock from {shock:%b %Y}; still held after 1/2/3 quarters: '+' / '.join(f'{h:.2f}' for h in held),flush=True)
    for y in ['mark','b99','fvc']:
        dd,b,sc,Ainv=fit(d,y,['sw_post']+C,['pkey','fq']); s1=se(dd,sc,Ainv,'loan')[0]; s2=se(dd,sc,Ainv,('loan','family'))[0]
        print(f'  {y:4s} software x post {b["sw_post"]:+.2f} (SE loan {s1:.2f}; loan+family {s2:.2f}; t {b["sw_post"]/s1:+.1f})',flush=True)
    ev=[]
    for q in sorted(d.ddate.unique()):
        if q==bq: continue
        c='e'+pd.Timestamp(q).strftime('%Y%m'); d[c]=d.sw*(d.ddate==q); ev.append(c)
    pre=[c for c in ev if c<'e'+bq.strftime('%Y%m')]; ix=[ev.index(c) for c in pre]
    for y in ['mark','b99']:
        dd,b,sc,Ainv=fit(d,y,ev+C,['pkey','fq']); Vc=vcov(dd,sc,Ainv,'loan'); s=np.sqrt(np.diag(Vc))
        line='; '.join(f'{c[1:5]}-{c[5:]} {b[c]:+.2f} ({s[k]:.2f})' for k,c in enumerate(ev))
        if pre:
            bp=b[pre].to_numpy(); Vp=Vc[np.ix_(ix,ix)]; W=float(bp@np.linalg.pinv(Vp)@bp)
            line+=f' | backward joint test chi2({len(pre)}) = {W:.1f}, p = {1-stats.chi2.cdf(W,len(pre)):.3f}'
        print(f'  event time {y}: '+line,flush=True)
    x=d[d.ddate==after[-1]]; x0=d[d.ddate==bq]
    print(f'  share below 99, software vs other: base {x0[x0.sw==1].b99.mean():.1f} vs {x0[x0.sw==0].b99.mean():.1f}; '
          f'{after[-1]:%b %Y} {x[x.sw==1].b99.mean():.1f} vs {x[x.sw==0].b99.mean():.1f}',flush=True)
cohort('2023-09-30','FAKE cohort 1')
cohort('2024-09-30','FAKE cohort 2')
cohort('2025-09-30','REAL cohort (same rules, funds\' own labels)')
