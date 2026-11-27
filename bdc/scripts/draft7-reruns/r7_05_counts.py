# Items 4 and 5 (draft 7): sample counts.
# 5a Table 2 "Funds" row: funds holding at least one cohort position, by software / other / all (columns overlap).
# 5b Figure 3 borrower counts: the figure counts borrowers with a valid quarter-on-quarter change after Sep 2025 (d6_figs.py line 72),
#    Table 2 counts borrowers in the cohort on 30 Sep 2025. Count the difference and why.
# 4  Table 7 panel C: positions, borrowers, funds behind each estimate, and the funds that identify kappa under fund x sector FE.
from r7_common import *
from d2_rank import changes
import d2_step3
L=Log(f'{OUT}/log_05_counts.txt'); p=load()
keys=cohort_keys(p); s9=p[(p.ddate==BASE)&p.pkey.isin(keys)].copy(); s9['sw']=s9.grp=='SOFT'
for nm,m in [('Sw',s9.sw),('Ot',~s9.sw),('All',s9.sw|~s9.sw)]:
    x=s9[m]; put(f'RsTabTwoFunds{nm}',x.cik.nunique(),'int',f'Table 2 funds row, {nm}'); put(f'RsTabTwoBorr{nm}',x.bid.nunique(),'int',f'Table 2 borrowers row, {nm}')
    put(f'RsTabTwoPos{nm}',len(x),'int',f'Table 2 positions, {nm}')
both=s9.groupby('cik').sw.agg(['any','all']); put('RsTabTwoFundsOnlyOt',int((~both['any']).sum()),'int','funds with no software cohort position')
L(f'[5a] Table 2: positions sw/ot/all {MAC["RsTabTwoPosSw"][0]}/{MAC["RsTabTwoPosOt"][0]}/{MAC["RsTabTwoPosAll"][0]}; funds {MAC["RsTabTwoFundsSw"][0]}/{MAC["RsTabTwoFundsOt"][0]}/{MAC["RsTabTwoFundsAll"][0]}; '
  f'funds without any software position {MAC["RsTabTwoFundsOnlyOt"][0]}; borrowers {MAC["RsTabTwoBorrSw"][0]}/{MAC["RsTabTwoBorrOt"][0]}')
# 5b
ch=changes(p); ch=ch[ch.grp.notna()&~ch.grp.isin(['UNKN','FUND','ITSV','DATA','HARD'])]
post=[q for q in sorted(ch.ddate.unique()) if q>BASE]
fig=set(ch[ch.ddate.isin(post)&(ch.grp=='SOFT')].bid); tab=set(s9[s9.sw].bid)
miss=tab-fig; put('RsFigThreeSwBorr',len(fig),'int','Figure 3 software borrowers: with a valid quarter-on-quarter change Dec 2025-Jun 2026')
put('RsFigThreeSwGap',len(miss),'int','software cohort borrowers not in Figure 3')
# why: no position of the borrower held at Dec 2025 (all exited in the first quarter), or held but the change fails the 80-100 day gap / validity rule
dec_held=set(p[(p.ddate==DEC)&p.pkey.isin(keys)&p.mark.between(0,110)&(p.principal>=250e3)].bid)
anyheld=set(p[(p.ddate>BASE)&p.pkey.isin(keys)].bid)
put('RsFigThreeGoneAll',len(miss-anyheld),'int','of those, borrowers with no cohort position at any date after Sep 2025 (all positions exited by Dec 2025)')
put('RsFigThreeOther',len(miss&anyheld),'int','of those, borrowers held later but without a valid consecutive-quarter change (gap or validity rule)')
L(f'[5b] software borrowers: Table 2 {len(tab)}, Figure 3 {len(fig)}, gap {len(miss)}: exited before Dec 2025 {len(miss-anyheld)}, other {len(miss&anyheld)}')
# other groups for the figure note
nb_fig=ch[ch.ddate.isin(post)].groupby('grp').bid.nunique(); nb_tab=s9.groupby('grp').bid.nunique()
L('[5b] all groups, Table-2-style count vs Figure 3 count:\n'+pd.DataFrame({'cohort':nb_tab,'fig3':nb_fig}).fillna(0).astype(int).to_string())
# 4 Table 7 panel C
s=d2_step3.frame(p); u=s[s.cls.isin(['traded','private'])].copy(); u['Tsw']=u.T_*u.sw; x=u.dropna(subset=['dm202606']).copy()
def cnt(z,tag,lab):
    put(f'Rs{tag}Pos',len(z),'int',lab+': positions'); put(f'Rs{tag}Borr',z.bid.nunique(),'int',lab+': borrowers'); put(f'Rs{tag}Funds',z.cik.nunique(),'int',lab+': funds')
    put(f'Rs{tag}SwPos',int(z.sw.sum()),'int',lab+': software positions'); put(f'Rs{tag}SwBorr',z[z.sw==1].bid.nunique(),'int',lab+': software borrowers')
for c,Cn in [('traded','Tr'),('private','Pr')]:
    z=x[x.cls==c]; cnt(z,f'PanC{Cn}',f'Table 7 panel C, {c} only, June 2026')
    g=z.groupby('cik').sw.agg(['min','max']); idf=set(g[(g['min']==0)&(g['max']==1)].index)
    put(f'RsPanC{Cn}IdFunds',len(idf),'int',f'{c} only: funds holding both software and other {c} positions (identify the fund-FE estimate)')
cnt(x,'PanCGap','Table 7 panel C, traded minus private (Eq. 4), June 2026')
fs=x.groupby('fs').T_.agg(['min','max']); idfs=fs[(fs['min']==0)&(fs['max']==1)].index
kap=x[x.fs.isin(idfs)&(x.sw==1)]; put('RsKappaFunds',kap.cik.nunique(),'int','funds holding both traded and private software positions in June 2026 (identify kappa)')
put('RsKappaSwPos',len(kap),'int','software positions in those fund x software cells'); put('RsKappaSwBorr',kap.bid.nunique(),'int','software borrowers in those cells')
put('RsKappaSwTrPos',int((kap.cls=='traded').sum()),'int','traded software positions in identifying cells'); put('RsKappaSwPrPos',int((kap.cls=='private').sum()),'int','private software positions in identifying cells')
oth=x[x.fs.isin(idfs)&(x.sw==0)]; put('RsKappaOtFunds',oth.cik.nunique(),'int','funds identifying the traded effect for other loans')
# concentration: share of identifying software positions in the 5 largest identifying funds
top=kap.cik.value_counts(); put('RsKappaTopFiveShare',top.head(5).sum()/top.sum(),'pct0','share of identifying software positions in the five funds with most of them')
fam=p.drop_duplicates('cik').set_index('cik').family; top.index=top.index.astype(int)
L('[4] Table 7 panel C counts:\n'+'\n'.join(f'  {n}: {MAC[n][0]}' for n in MAC if n.startswith('RsPanC') or n.startswith('RsKappa')))
L('[4] identifying funds (software positions in both-class cells): '+', '.join(f'{fam.get(c,c)} {k}' for c,k in top.head(12).items()))
# check: the estimate on the identifying cells only equals the full estimate (it must)
import statsmodels.formula.api as smf
r=d2_step3.ols('dm202606 ~ T_ + Tsw + C(fs)',x); r2=d2_step3.ols('dm202606 ~ T_ + Tsw + C(fs)',x[x.fs.isin(idfs)])
L(f'[4] kappa full sample {r.params["Tsw"]:+.3f} ({r.bse["Tsw"]:.3f}); identifying cells only {r2.params["Tsw"]:+.3f} ({r2.bse["Tsw"]:.3f})')
write_macros(f'{OUT}/numbers-r7-05-counts.tex',list(MAC),'Draft 7 rerun items 4 and 5: Table 2 / Figure 3 counts and Table 7 panel C counts (r7_05_counts.py, 26 Nov 2026)')
L.save()
