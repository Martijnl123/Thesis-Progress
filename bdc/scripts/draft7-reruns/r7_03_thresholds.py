# Item 3 (draft 7): the Below99 threshold. Below98, Below97 and Below95 next to Below99 for
#  (i) Step 1 headline (Eq. 1, model labels, Sep 2025 cohort) and funds' labels on the cohort, Dec 2025 / Mar / Jun 2026 + joint pre-trend Wald;
#  (ii) Table 7 panel B row "share below, change to June, relative to other loans of the class" (Eq. 1 within the traded and the private class);
#  (iii) placebo cohorts Sep 2023 and Sep 2024 and the real Sep 2025 cohort (same code as d2_rank.fake_cohorts, funds' labels).
from r7_common import *
import d2_step3
L=Log(f'{OUT}/log_03_thresholds.txt'); p=load()
TH=[99,98,97,95]
W={99:'NineNine',98:'NineEight',97:'NineSeven',95:'NineFive'}   # macro names take letters only
for t in TH: p[f'b{t}']=100*(p.mark<t)
YS=tuple(f'b{t}' for t in TH)
keys=cohort_keys(p); d0=frozen(p,keys); d0['sw']=(d0.grp=='SOFT').astype(float)
c=d0[FUNDLAB(d0)].copy(); cf=c[c.pkey.isin(set(c.loc[c.ddate==BASE,'pkey']))].copy(); cf['sw']=(cf.fsec=='software').astype(float)
s9=d0[d0.ddate==BASE]
for t in TH:
    put(f'ThrShare{W[t]}Sw',(s9[s9.sw==1].mark<t).mean(),'pct1',f'cohort software positions marked below {t} on 30 Sep 2025')
    put(f'ThrShare{W[t]}Ot',(s9[s9.sw==0].mark<t).mean(),'pct1',f'cohort other positions marked below {t} on 30 Sep 2025')
for nm,lab,d in [('Model','model labels, Sep 2025 cohort',d0),('Fund',"funds' labels, Sep 2025 cohort",cf)]:
    r=es_v(d,ys=YS)
    for t in TH:
        P=r[f'b{t}']['path']
        for q,Q in [('2025-12','Dec'),('2026-03','Mar'),('2026-06','Jun')]:
            putse(f'Thr{nm}{W[t]}{Q}',P.b[q],P.se[q],f'Step 1 Eq. 1, {lab}, share below {t} (pp), {q}',d=1)
        putse(f'Thr{nm}{W[t]}First',P.b['2022-12'],P.se['2022-12'],f'Step 1, {lab}, below {t}, Dec 2022',d=1)
        put(f'Thr{nm}{W[t]}WaldP',r[f'b{t}']['p'],'f3',f'joint pre-trend Wald p, {lab}, below {t}')
        L(f'[step1] {nm} below{t}: '+'; '.join(f'{q} {P.b[q]:+.1f} ({P.se[q]:.1f})' for q in P.index)+f' | pre Wald p {r[f"b{t}"]["p"]:.3f}')
# (ii) within class
s=d2_step3.frame(p); d0['cls']=d0.pkey.map(s.cls)
for nm,m in [('Tr',d0.cls=='traded'),('Pr',d0.cls=='private')]:
    r=es(d0[m],ys=YS)
    for t in TH:
        b,se_=get(r,f'b{t}','2026-06'); putse(f'ThrWithin{nm}{W[t]}Jun',b,se_,f'Table 7 panel B: Eq. 1 within {nm} class, below {t}, June 2026',d=1)
    L(f'[panelB] {nm}: '+'; '.join(f'below{t} {get(r,f"b{t}","2026-06")[0]:+.1f} ({get(r,f"b{t}","2026-06")[1]:.1f})' for t in TH))
# raw class means (panel B style): change in share below t, Sep 2025 to June 2026, positions held at both
w=d0.pivot_table(index='pkey',columns=d0.ddate.dt.strftime('%Y%m'),values='mark')
for t in TH:
    for sw,SW in [(1,'Sw'),(0,'Ot')]:
        for cl,Cn in [('traded','Tr'),('private','Pr')]:
            k=s[(s.sw==sw)&(s.cls==cl)].index.intersection(w.index); z=w.loc[k,['202509','202606']].dropna()
            put(f'ThrRaw{SW}{Cn}{W[t]}',100*((z['202606']<t).mean()-(z['202509']<t).mean()),'f1',f'raw change in share below {t}, {SW} {cl}, held Sep 2025 and Jun 2026 (pp)',sign=True)
# (iii) placebo cohorts, same code as d2_rank.fake_cohorts with the threshold varied
lab=FUNDLAB(p); qs=sorted(p.ddate.unique())
for bq,nm in [('2023-09-30','Three'),('2024-09-30','Four'),('2025-09-30','Real')]:
    bq=pd.Timestamp(bq); i=qs.index(bq); after=qs[i+1:i+4]; shock=after[1]
    base=p[(p.ddate==bq)&p.ok&lab]; ks=set(base.pkey); sp=base.set_index('pkey').spread; pk=base.set_index('pkey').pik
    fz=p[p.pkey.isin(ks)&(p.ddate<=after[-1])]; post=fz.ddate>bq
    d=fz[~post&fz.ok | post&fz.mark.between(0,110)&(fz.principal>=250e3)].copy(); d['sw']=(d.fsec=='software').astype(float)
    d['post']=(d.ddate>=shock).astype(float); d['sp_b']=d.pkey.map(sp); d['pk_b']=d.pkey.map(pk).fillna(0)
    d['sp_post']=d.sp_b.fillna(0)*100*d.post; d['spm_post']=d.sp_b.isna()*d.post; d['pik_post']=d.pk_b*d.post; d['sw_post']=d.sw*d.post
    out=[]
    for t in TH:
        dd,b,sc,Ainv=fit(d,f'b{t}',['sw_post','sp_post','spm_post','pik_post'],['pkey','fq']); s_=se(dd,sc,Ainv,'clu')[0]
        putse(f'ThrFake{nm}{W[t]}',float(b['sw_post']),float(s_),f'placebo cohort {bq:%b %Y}, funds labels, post = 2 quarters from {shock:%b %Y}, below {t} (pp)',d=1)
        bb=float(b['sw_post']); out.append(f'below{t} {bb:+.1f} ({s_:.1f})')
        if nm!='Real': pass
    L(f'[placebo] {nm}: '+'; '.join(out))
# placebo as a share of the real effect, per threshold
for t in TH:
    put(f'ThrFakeRatio{W[t]}',MAC[f'ThrFakeThree{W[t]}'][1]/MAC[f'ThrFakeReal{W[t]}'][1],'pct0',f'2023 placebo as a share of the real-cohort estimate, below {t}',sign=True)
write_macros(f'{OUT}/numbers-r7-03-thresholds.tex',list(MAC),'Draft 7 rerun item 3: Below98/97/95 next to Below99 (r7_03_thresholds.py, 26 Nov 2026)')
L.save()
