# Draft 6 (26 Nov 2026): the new June-data checks of the final rewrite -> <out>/numbers-d6.tex and <out>/numbers-d6-log.txt.
# Data to 30 June 2026 only (d2_common.load() refuses later rows). Nothing registered is recomputed or changed.
#   R1 (#173, PAP v5 s.4 "Robustness: the last 8 and the last 10 pre-period changes, and the sum from March 2026"): software's rank
#      on the scaled statistic with those three variants.
#   R2 (#71) overlapping-window check of the sqrt(3) scaling: SD of the 9 overlapping three-quarter sums of pre-period changes
#      against sqrt(3) x SD of the 11 single changes; the rank with the overlapping SD as the scale.
#   R3 (#91, examiner Q2) education without Pluralsight: plain change, software's plain rank, scaled rank and Ferman-Pinto p (mark),
#      with Pluralsight's positions dropped from the rank sample (one borrower, bid 254). Post hoc, labelled so.
#   R4 (#122) Step 3, June: two-sided 5% trim (5% largest falls AND 5% largest rises dropped in every class x sector cell).
#   R5 (#113, disclosed addition, saved for draft 2) Step 3 kappa by fund type (vtype at Sep 2025).
# Macros use d4_common.put4 with prefix Dsix, so nothing can redefine an earlier macro.
# Usage: python3 d6_numbers.py <draft6 folder>
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..','..')))  # repo root
import sys, time
sys.path.insert(0,(_ROOT+'/bdc/scripts/draft4')); sys.path.insert(0,(_ROOT+'/bdc/scripts/draft2'))
from d4_common import *
import d2_step3 as S3
OUT=sys.argv[1]; LOG=[]; t0=time.time()
def log(s): LOG.append(str(s)); print(s,flush=True)
BASE,JUN=CM.BASE,CM.JUN
EXCL=['UNKN','FUND','ITSV','DATA','HARD']
p=CM.load(); assert p.ddate.max()==JUN
ch=changes(p); ch=ch[ch.grp.notna()&~ch.grp.isin(EXCL)]
qs=sorted(ch.ddate.unique()); pre=[q for q in qs if q<=BASE]; post=[q for q in qs if q>BASE]; assert len(pre)==11 and len(post)==3

def stats(ch_):
    rows=[]
    for g in sorted(ch_.grp.unique()):
        x=ch_ if g=='SOFT' else ch_[ch_.grp!='SOFT']
        M=x[x.grp==g].groupby('ddate').pkey.nunique().mean()
        for y in ['dmark','db99']:
            c=pd.Series({q:c_t(x[x.ddate==q],y,(x[x.ddate==q].grp==g).astype(float)) for q in qs})
            sd=c[pre].std(); s3=pd.Series([c[pre[i:i+3]].sum() for i in range(len(pre)-2)]).std()
            rows.append(dict(grp=g,y=y,post=c[post].sum(),M=M,T=c[post].sum()/sd/np.sqrt(3),T8=c[post].sum()/c[pre[-8:]].std()/np.sqrt(3),
                             T10=c[post].sum()/c[pre[-10:]].std()/np.sqrt(3),Tmar=c[post[1:]].sum()/sd/np.sqrt(2),Tov=c[post].sum()/s3,ratio=s3/(np.sqrt(3)*sd)))
    return pd.DataFrame(rows)
def rk(S,y,col,g='SOFT'):
    t=S[S.y==y].sort_values(col,ascending=(y=='dmark')).reset_index(drop=True); return int(t.index[t.grp==g][0])+1
S=stats(ch); G=S.grp.nunique()
assert rk(S,'db99','T')==1 and rk(S,'dmark','T')==1, 'headline rank does not reproduce'
put4('DsixNGroups',G,'int','groups in the rank')
for y,Y in [('db99','Below'),('dmark','Mark')]:
    for col,C in [('T8','Eight'),('T10','Ten'),('Tmar','Mar'),('Tov','Ov')]:
        put4(f'DsixRank{C}{Y}',rk(S,y,col),'int',f'software rank, {col}, {y}, {G} groups')
    sw=S[(S.y==y)&(S.grp=='SOFT')].iloc[0]
    put4(f'DsixOvRatioSw{Y}',sw.ratio,'f2',f'software: SD of overlapping 3-quarter pre sums / (sqrt3 x SD of single changes), {y}')
    put4(f'DsixOvRatioMed{Y}',S[S.y==y].ratio.median(),'f2',f'median over groups of the same ratio, {y}')
    put4(f'DsixOvShareAbove{Y}',(S[S.y==y].ratio>1).mean(),'pct0',f'share of groups with ratio above 1 (positive autocorrelation), {y}')
    log(f'[R1/R2] {y}: ranks T {rk(S,y,"T")} T8 {rk(S,y,"T8")} T10 {rk(S,y,"T10")} Tmar {rk(S,y,"Tmar")} Tov {rk(S,y,"Tov")}; software ratio {sw.ratio:.2f}, median ratio {S[S.y==y].ratio.median():.2f}, above 1 {(S[S.y==y].ratio>1).sum()} of {G}')
    log('   top 4 on Tov: '+'; '.join(f'{r.grp} {r.Tov:+.2f}' for _,r in S[S.y==y].sort_values('Tov',ascending=(y=='dmark')).head(4).iterrows()))

# R3 education without Pluralsight (bid 254)
Lb=pd.read_parquet(f'{CM.F}/labels.parquet').drop_duplicates('bid').set_index('bid')
assert Lb.loc[254,'operating_name']=='Pluralsight' and Lb.loc[254,'group']=='EDUC'
chx=ch[ch.bid!=254]
def plain(x,g):
    x=x if g=='SOFT' else x[x.grp!='SOFT']; return sum(c_t(x[x.ddate==q],'dmark',(x[x.ddate==q].grp==g).astype(float)) for q in post)
pe0=plain(ch,'EDUC'); pe=plain(chx,'EDUC'); ps=plain(chx,'SOFT')
others=pd.Series({g:plain(chx,g) for g in chx.grp.unique()}); prk=int((others<others['SOFT']).sum())+1
put4('DsixEducPlainAll',pe0,'f2','education plain mark change, rank sample (= Table A3 value)',sign=True)
put4('DsixEducPlainNoPl',pe,'f2','education plain mark change without Pluralsight (post hoc)',sign=True)
put4('DsixSwPlainNoPl',ps,'f2','software plain mark change in the same sample',sign=True)
put4('DsixSwPlainRankNoPl',prk,'int','software rank on the plain mark change without Pluralsight')
Sx=stats(chx); put4('DsixSwScaledRankNoPlMark',rk(Sx,'dmark','T'),'int','software scaled rank, mark, without Pluralsight')
put4('DsixEducNPosPl',ch[(ch.bid==254)&ch.ddate.isin(post)].pkey.nunique(),'int','Pluralsight positions in the post quarters of the rank sample')
x=Sx[Sx.y=='dmark'].set_index('grp'); a=x['post']; M=x['M']; ctrl=[g for g in x.index if g!='SOFT']
coef=np.linalg.lstsq(np.column_stack([np.ones(len(ctrl)),1/M[ctrl].values]),(a[ctrl]**2).values,rcond=None)[0]
v=lambda m: max(coef[0]+coef[1]/m,1e-12); resc=pd.Series({g:a[g]*np.sqrt(v(M['SOFT'])/v(M[g])) for g in ctrl})
ext=int((resc<=a['SOFT']).sum()); fp=(1+ext)/(len(ctrl)+1)
put4('DsixFpMarkNoPl',fp,'f3','Ferman-Pinto one-sided p, mark, without Pluralsight')
log(f'[R3] education plain {pe0:+.2f} -> {pe:+.2f} without Pluralsight; software {ps:+.2f}, plain rank {prk}; scaled rank {rk(Sx,"dmark","T")}; FP p {fp:.3f} ({ext} more extreme); top plain: '+'; '.join(f'{g} {v_:+.2f}' for g,v_ in others.sort_values().head(4).items()))

# R4 Step 3 two-sided trim, June; R5 by fund type
s=S3.frame(p); u=s[s.cls.isin(['traded','private'])].copy(); u['Tsw']=u.T_*u.sw; xx=u.dropna(subset=['dm202606']).copy()
r=S3.ols('dm202606 ~ T_ + Tsw + C(fs)',xx); assert abs(r.params['Tsw']-(-7.14))<0.01, r.params['Tsw']
q=xx.groupby(['cls','sw']).dm202606
keep=(xx.dm202606>q.transform(lambda v_:v_.quantile(0.05)))&(xx.dm202606<q.transform(lambda v_:v_.quantile(0.95)))
r2=S3.ols('dm202606 ~ T_ + Tsw + C(fs)',xx[keep]); putse4('DsixGapTwoTrim',r2.params['Tsw'],r2.bse['Tsw'],'Step 3 June kappa, 5% largest falls and 5% largest rises dropped in every class x sector cell')
put4('DsixGapTwoTrimLo',r2.params['Tsw']-1.96*r2.bse['Tsw'],'f2','95% CI',sign=True); put4('DsixGapTwoTrimHi',r2.params['Tsw']+1.96*r2.bse['Tsw'],'f2','95% CI',sign=True)
rp=S3.ols('dm202606 ~ sw + C(cik)',xx[keep&(xx.cls=='private')]); putse4('DsixPrivTwoTrim',rp.params['sw'],rp.bse['sw'],'private only, two-sided trim')
log(f'[R4] Step 3 June kappa {r.params["Tsw"]:+.2f} ({r.bse["Tsw"]:.2f}); two-sided 5% trim {r2.params["Tsw"]:+.2f} ({r2.bse["Tsw"]:.2f}), n {int(keep.sum())} of {len(xx)}; private only {rp.params["sw"]:+.2f} ({rp.bse["sw"]:.2f})')
vt=p[(p.ddate==BASE)].drop_duplicates('pkey').set_index('pkey').vtype
xx['vt']=xx.pkey.map(vt) if 'pkey' in xx else xx.index.map(vt)
log('[R5] fund types: '+str(xx.vt.value_counts(dropna=False).to_dict()))
for t in sorted(xx.vt.dropna().unique()):
    z=xx[xx.vt==t]; n=z.groupby(['sw','cls']).size()
    if min(n.get((1,'traded'),0),n.get((1,'private'),0))<20: log(f'[R5] {t}: too few ({n.to_dict()})'); continue
    rr=S3.ols('dm202606 ~ T_ + Tsw + C(fs)',z); rp=S3.ols('dm202606 ~ sw + C(cik)',z[z.cls=='private'])
    log(f'[R5] {t}: kappa {rr.params["Tsw"]:+.2f} ({rr.bse["Tsw"]:.2f}); private only {rp.params["sw"]:+.2f} ({rp.bse["sw"]:.2f}); n {len(z)}, software traded {n.get((1,"traded"),0)}, private {n.get((1,"private"),0)}')
h=write_macros(f'{OUT}/numbers-d6.tex','% numbers-d6.tex: draft 6 checks on data to 30 June 2026. Written by bdc/scripts/draft6/d6_numbers.py.\n')
write_log(f'{OUT}/numbers-d6-log.txt',f'd6_numbers.py run {time.strftime("%Y-%m-%d %H:%M UTC",time.gmtime())}, {time.time()-t0:.0f} s, sha256 {h[:16]}',LOG)
print('done',h[:16])
