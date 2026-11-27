# Item 6 (draft 7), cheap extras:
# (a) Table A3 maturity split: is the larger markdown where maturity is not reported a manager effect? Rerun both rows with
#     manager x quarter FE (instead of fund x quarter), on managers that report maturity for some but not all positions
#     (within-manager split), and print which managers make up each half.
# (b) H4 with manager x Post and manager x Soft FE (d2_misc.h4, unchanged; recomputed here so the number has a log line).
# (c) Table A6 panel B: power of the Step 3 rule with a fund-level bootstrap of the March-June 2026 rehearsal that keeps the
#     correlation between P and Q (draft 6 multiplies two independent normal probabilities).
from r7_common import *
import d2_step3, d2_misc
from scipy.stats import norm
L=Log(f'{OUT}/log_06_extras.txt'); p=load()
keys=cohort_keys(p); d0=frozen(p,keys); d0['sw']=(d0.grp=='SOFT').astype(float)
# ---------------- (a)
s9=d0[d0.ddate==BASE].copy(); s9['hm']=s9.ttm.notna(); fm=s9.groupby('family').agg(h=('hm','mean'),pr=('principal','sum'),n=('pkey','size'))
hasm=set(d0.loc[d0.ttm.notna(),'pkey'])
mixed=set(fm[(fm.h>0.1)&(fm.h<0.9)].index)
for nm,lab,d,fe in [('Yes','maturity reported',d0[d0.pkey.isin(hasm)],('pkey','fq','vq')),('No','maturity not reported',d0[~d0.pkey.isin(hasm)],('pkey','fq','vq')),
                    ('YesMq','maturity reported, manager x quarter FE',d0[d0.pkey.isin(hasm)],('pkey','mq','vq')),('NoMq','maturity not reported, manager x quarter FE',d0[~d0.pkey.isin(hasm)],('pkey','mq','vq')),
                    ('YesMix','maturity reported, managers reporting for 10-90% of positions',d0[d0.pkey.isin(hasm)&d0.family.isin(mixed)],('pkey','fq','vq')),
                    ('NoMix','maturity not reported, same managers',d0[~d0.pkey.isin(hasm)&d0.family.isin(mixed)],('pkey','fq','vq'))]:
    r=es(d,fe=fe)
    for y,Y,dg in [('mark','Mark',2),('b99','Below',1)]: putse(f'Mat{nm}{Y}Jun',*get(r,y,'2026-06'),f'Step 1 June 2026, {lab}',d=dg)
    put(f'NMat{nm}',r['n']['pos'],'int',f'{lab}: positions'); put(f'NMat{nm}Mgr',r['n']['managers'],'int',f'{lab}: managers')
    L(f'[A3] {lab}: {r["n"]} | June mark {get(r,"mark","2026-06")[0]:+.2f} ({get(r,"mark","2026-06")[1]:.2f}), below99 {get(r,"b99","2026-06")[0]:+.1f} ({get(r,"b99","2026-06")[1]:.1f})')
# interaction test within manager x quarter: does the software effect differ by maturity reporting once managers are held fixed?
d=d0.copy(); d['nm']=(~d.pkey.isin(hasm)).astype(float); d['sw_nm']=d.sw*d.nm
r=es(d,swcols=('sw','sw_nm'),fe=('pkey','mq','vq'))
for y,Y,dg in [('mark','Mark',2),('b99','Below',1)]: putse(f'MatDiffMq{Y}Jun',*get(r,y,'2026-06','sw_nm'),f'June 2026: software x maturity-not-reported, manager x quarter FE (difference of the two rows)',d=dg)
L(f'[A3] difference (not reported minus reported), manager x quarter FE: mark {get(r,"mark","2026-06","sw_nm")[0]:+.2f} ({get(r,"mark","2026-06","sw_nm")[1]:.2f}), below99 {get(r,"b99","2026-06","sw_nm")[0]:+.1f} ({get(r,"b99","2026-06","sw_nm")[1]:.1f})')
put('NMatMixMgr',len(mixed),'int','managers reporting maturity for 10-90% of cohort positions')
put('MatNoTopShare',0,'pct0','placeholder')
del MAC['MatNoTopShare']
no=s9[~s9.hm].groupby('family').principal.sum().sort_values(ascending=False); yes=s9[s9.hm].groupby('family').principal.sum().sort_values(ascending=False)
put('MatNoTopThreeShare',no.head(3).sum()/no.sum(),'pct0','share of not-reported principal held by the three largest managers in that half')
put('MatNoTopThreeNames',', '.join(x.title() for x in no.head(3).index),'str','those managers')
L('[A3] maturity reporting by manager (share of positions with maturity, principal $bn), 20 largest:\n'+fm.sort_values('pr',ascending=False).assign(pr=lambda x:(x.pr/1e9).round(1)).round(2).head(20).to_string())
L(f'[A3] principal of the not-reported half: top 3 managers {no.head(3).round(-6).to_dict()} share {no.head(3).sum()/no.sum():.2f}')
# ---------------- (b)
T={}; d2_misc.h4(p,T,L)
# ---------------- (c)
s=d2_step3.frame(p); fz=frozen(p,set(s.index)); w=fz.pivot_table(index='pkey',columns=fz.ddate.dt.strftime('%Y%m'),values='mark')
s['dq']=(w['202606']-w['202603']).reindex(s.index)
z=s[s.cls.isin(['traded','private'])].dropna(subset=['dq']).reset_index()
def PQ(zz):
    out=[]
    for c in ['private','traded']:
        y=zz[zz.cls==c]; out.append(d2_step3.ols('dq ~ sw + C(cik)',y).params['sw'])
    return out
P0,Q0=PQ(z); L(f'[A6] rehearsal Mar-Jun 2026 all funds: P {P0:+.3f}, Q {Q0:+.3f} (compare with draft 6 macros RehPMarJun, RehQMarJun)')
# fast within-fund estimator for the bootstrap: demean dq and sw within fund in each class, slope = sum(sw~ dq~)/sum(sw~^2)
def fastPQ(zz):
    r=[]
    for c in ['private','traded']:
        y=zz[zz.cls==c]; sd=y.sw-y.groupby('bk').sw.transform('mean'); e=y.dq-y.groupby('bk').dq.transform('mean'); r.append(float((sd*e).sum()/(sd*sd).sum()))
    return r
z['bk']=z.cik; chk=fastPQ(z); assert abs(chk[0]-P0)<1e-6 and abs(chk[1]-Q0)<1e-6
funds=z.cik.unique(); idx=z.groupby('cik').indices; rng=np.random.default_rng(20261126); B=[]
for b in range(2000):
    dr=rng.choice(len(funds),len(funds),replace=True); ii=np.concatenate([idx[funds[j]] for j in dr])
    zz=z.iloc[ii].copy(); zz['bk']=np.repeat(np.arange(len(dr)),[len(idx[funds[j]]) for j in dr]); B.append(fastPQ(zz))
B=np.array(B); B=B[np.isfinite(B).all(1)]; eP=B[:,0]-B[:,0].mean(); eQ=B[:,1]-B[:,1].mean()
seP,seQ=eP.std(ddof=1),eQ.std(ddof=1); rho=float(np.corrcoef(eP,eQ)[0,1])
put('PowBootSEP',seP,'f2','fund-bootstrap SE of P, Mar-Jun 2026 rehearsal, all funds, 2000 draws'); put('PowBootSEQ',seQ,'f2','fund-bootstrap SE of Q')
put('PowBootRho',rho,'f2','fund-bootstrap correlation of P and Q',sign=True); put('NPowBootFunds',len(funds),'int','funds resampled')
L(f'[A6] fund bootstrap: {len(B)} draws, SE P {seP:.2f}, SE Q {seQ:.2f}, corr {rho:+.2f}')
rows=[]
for (Pt,Qt),nm in [((0,0),'Zero'),((-0.5,0),'Half'),((-1,0),'One'),((0,2),'Over')]:
    Ph=Pt+eP; Qh=Qt+eQ
    pr={'R1':np.mean((Ph<=-0.5)&(Qh<=1)),'R2':np.mean((Ph>-0.5)&(Qh<=1)),'R3':np.mean((Ph>-0.5)&(Qh>1)),'none':np.mean((Ph<=-0.5)&(Qh>1))}
    # draft 6 version for comparison (independent normals at the clustered SEs)
    for k,vv in pr.items(): put(f'PowBoot{nm}{k.replace("none","None").replace("R1","Ra").replace("R2","Rb").replace("R3","Rc")}',vv,'pct0',f'fund bootstrap, P(verdict {k}) when true (P,Q)=({Pt},{Qt})')
    rows.append(f'({Pt:+.1f}, {Qt:+.1f}): R1 {100*pr["R1"]:.0f}%, R2 {100*pr["R2"]:.0f}%, R3 {100*pr["R3"]:.0f}%, none {100*pr["none"]:.0f}%')
L('[A6] panel B with the fund bootstrap (rule: R1 P<=-0.5 & Q<=1; R2 P>-0.5 & Q<=1; R3 P>-0.5 & Q>1):\n  '+'\n  '.join(rows))
# the same with the bootstrap SEs but independence, to separate the SE change from the correlation
rows=[]
for (Pt,Qt),nm in [((0,0),'Zero'),((-0.5,0),'Half'),((-1,0),'One'),((0,2),'Over')]:
    pP=norm.cdf((-0.5-Pt)/seP); pQ=1-norm.cdf((1-Qt)/seQ); rows.append(f'({Pt:+.1f}, {Qt:+.1f}): R1 {100*pP*(1-pQ):.0f}%, R2 {100*(1-pP)*(1-pQ):.0f}%, R3 {100*(1-pP)*pQ:.0f}%, none {100*pP*pQ:.0f}%')
L('[A6] same bootstrap SEs, independence imposed:\n  '+'\n  '.join(rows))
write_macros(f'{OUT}/numbers-r7-06-extras.tex',list(MAC),'Draft 7 rerun item 6: A3 maturity split with manager FE, H4 with manager FE, A6 panel B fund bootstrap (r7_06_extras.py, 26 Nov 2026)')
L.save()
