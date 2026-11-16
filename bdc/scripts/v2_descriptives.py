# Descriptive tables and time series (30 Oct 2026, show the data first, min/max/missing values,
# plus a time series). Writes LaTeX tables and figures to notes/descriptives-2026-10-30/.
# Samples: (a) all debt rows in the panel (raw), (b) rows passing the quarterly filter 'ok' (cost/principal 0.90-1.05,
# mark 20-110, principal >= $250k; see v2_panel.py), (c) the frozen Sep 2025 cohort used for H1 (LLM labels, UNKN/FUND dropped).
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..')))  # repo root
import sys, warnings, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
V=(_ROOT+'/bdc/v2'); F=(_ROOT+'/bdc/industry_llm/full'); O=(_ROOT+'/outputs/descriptives-2026-10-30')
p=pd.read_parquet(f'{V}/panel.parquet')
L=pd.read_parquet(f'{F}/labels.parquet')[['bid','group']]
g=pd.read_parquet(f'{F}/position_borrower.parquet').merge(L,on='bid').merge(p[['pos_id','pkey']],on='pos_id').drop_duplicates('pkey').set_index('pkey').group
p['grp']=p.pkey.map(g)
p['principal_m']=p.principal/1e6; p['cost_m']=p.cost/1e6; p['fv_m']=p.mark*p.principal/100/1e6
p['fvc']=p.mark*p.principal/p.cost; p.loc[~np.isfinite(p.fvc),'fvc']=np.nan
p['spread_bp']=p.spread*1e4; p['spread_pre_bp']=p.spread_pre*1e4
out=[]
def tex(df,name,cap,fmt,colfmt=None):
    df=df.rename_axis(None)
    s=df.to_latex(index=True,escape=True,float_format=fmt,column_format=colfmt or 'l'+'r'*df.shape[1])
    box=colfmt is None
    s=s.replace('\\begin{tabular}','\\begin{table}[htbp]\\centering\\small\n\\caption{'+cap+'}\\label{tab:'+name+'}\n'+('\\begin{adjustbox}{max width=\\textwidth}' if box else '')+'\\begin{tabular}').replace('\\end{tabular}','\\end{tabular}'+('\\end{adjustbox}' if box else '')+'\n\\end{table}')
    open(f'{O}/{name}.tex','w').write(s); df.to_csv(f'{O}/{name}.csv'); print(f'\n== {name}: {cap}\n{df.to_string(float_format=fmt)}',flush=True)

# Table 1: variable summary, raw and cleaned
vars_={'Mark (fair value per 100 principal)':'mark','Principal (\\$m)':'principal_m','Cost (\\$m)':'cost_m','Fair value / cost (\\%)':'fvc',
       'Spread (bp)':'spread_bp','PIK (0/1)':'pik','Fund industry label present (0/1)':None,'LLM industry label present (0/1)':None}
def summ(x):
    rows={}
    for k,c in vars_.items():
        v=(x.sector.notna().astype(float) if 'Fund industry' in k else x.grp.notna().astype(float) if 'LLM' in k else x[c])
        n=len(v); vv=v.dropna()
        rows[k.replace('\\$','$').replace('\\%','%')]=dict(N=n,Missing_pct=100*(1-len(vv)/n),Mean=vv.mean(),SD=vv.std(),Min=vv.min(),P1=vv.quantile(.01),
                    Median=vv.median(),P99=vv.quantile(.99),Max=vv.max())
    return pd.DataFrame(rows).T
for lab,x,nm in [('all debt rows, Dec 2022 - Jun 2026 (raw)',p,'tab1a_vars_raw'),('rows passing the quarterly filter',p[p.ok],'tab1b_vars_clean')]:
    t=summ(x); t.columns=['N','Missing (%)','Mean','SD','Min','P1','Median','P99','Max']
    tex(t,nm,f'Summary statistics, {lab}',lambda v:(f'{v:,.0f}' if float(v).is_integer() and abs(v)>=1000 else f'{v:,.2f}'))

# Table 2: cleaning funnel
f=pd.DataFrame({'Rows':[len(p),(p.cost/p.principal).between(.9,1.05).sum(),((p.cost/p.principal).between(.9,1.05)&p.mark.between(20,110)).sum(),p.ok.sum()]},
    index=['All debt rows','Cost / principal 0.90-1.05','... and mark 20-110','... and principal at least $250k (= filter)'])
f['Share of all (%)']=100*f.Rows/len(p)
tex(f,'tab2_cleaning','How the quarterly filter shrinks the sample',lambda v:f'{v:,.1f}')

# Table 3: sample by quarter
q=p[p.ok]
t=q.groupby('ddate').agg(Funds=('cik','nunique'),Managers=('family','nunique'),Positions=('pos_id','size'),Loans=('loan','nunique'),
     Principal_bn=('principal',lambda v:v[v<=2e9].sum()/1e9),Mean_mark=('mark','mean'),Below99_pct=('mark',lambda v:100*(v<99).mean()))
# software share with the funds' own labels: LLM labels exist only for borrowers held in Sep 2025, so earlier quarters are survivors
fl=q.sector.notna()&~q.sector.isin(['it_services','tech_other'])
sw=q[fl].groupby('ddate').sector.apply(lambda v:100*(v=='software').mean())
t['Software (%)']=sw
t['Missing LLM label (%)']=q.groupby('ddate').grp.apply(lambda v:100*v.isna().mean())
t.index=t.index.strftime('%b %Y'); t.columns=['Funds','Managers','Positions','Loans','Principal ($bn)','Mean mark','Below 99 (%)','Software, fund labels (%)','No LLM label (%)']
tex(t,'tab3_by_quarter','Sample by quarter-end (rows passing the quarterly filter; principal excludes 27 rows above \\$2bn with a units error)',lambda v:f'{v:,.1f}')

# Frozen cohort (as v2_h1_q1start.py)
lab_=p.grp.notna()&~p.grp.isin(['UNKN','FUND'])
keys=set(p[(p.ddate=='2025-09-30')&p.ok&lab_].pkey)
fz=p[p.pkey.isin(keys)&lab_]; post=fz.ddate>'2025-09-30'
d=fz[~post&fz.ok | post&fz.mark.between(0,110)&(fz.principal>=250e3)].copy(); d['sw']=d.grp.eq('SOFT')
# Table 4: frozen cohort at Sep 2025, software vs other
b=d[d.ddate=='2025-09-30'].copy()
b['nholders']=b.groupby('loan').family.transform('nunique')
b['listed']=(b.vtype=='listed')*100; b['perp']=(b.vtype=='perpetual')*100; b['priv']=(b.vtype=='private')*100
b['b99']=(b.mark<99)*100; b['pik100']=b.pik*100
cols={'Positions':None,'Principal ($m)':'principal_m','Spread (bp)':'spread_bp','PIK (%)':'pik100','Mark':'mark','Below 99 (%)':'b99',
      'Fair value / cost (%)':'fvc','Managers holding the loan':'nholders','Listed BDC (%)':'listed','Perpetual BDC (%)':'perp','Private BDC (%)':'priv'}
rows={}
for k,c in cols.items():
    if c is None: rows[k]=dict(Software=b.sw.sum(),Other=(~b.sw).sum(),Difference=np.nan,t=np.nan); continue
    a,o=b.loc[b.sw,c].dropna(),b.loc[~b.sw,c].dropna(); df_=a.mean()-o.mean(); s=np.sqrt(a.var()/len(a)+o.var()/len(o))
    rows[k]=dict(Software=a.mean(),Other=o.mean(),Difference=df_,t=df_/s)
t4=pd.DataFrame(rows).T; tex(t4,'tab4_cohort_balance','Frozen cohort at 30 September 2025: software vs other loans (means; t from a simple two-sample test, not clustered)',lambda v:(f'{v:,.0f}' if float(v).is_integer() and abs(v)>=1000 else f'{v:,.2f}'))

# Table 5: what happens to the frozen cohort after Sep 2025
r=[]
for dt,x in fz.groupby('ddate'):
    if dt<pd.Timestamp('2025-09-30'): continue
    kept=x.pkey.nunique(); inn=d[d.ddate==dt]
    r.append(dict(Quarter=dt.strftime('%b %Y'),Still_reported=100*kept/len(keys),Kept_in_sample=100*inn.pkey.nunique()/len(keys),
             Software_kept=100*inn[inn.sw].pkey.nunique()/len(set(b.loc[b.sw,'pkey'])),Other_kept=100*inn[~inn.sw].pkey.nunique()/len(set(b.loc[~b.sw,'pkey'])),
             Mark_below_20=100*(inn.mark<20).mean()))
t5=pd.DataFrame(r).set_index('Quarter'); t5.columns=['Still reported (%)','In H1 sample (%)','Software in sample (%)','Other in sample (%)','Mark below 20 (%)']
tex(t5,'tab5_cohort_attrition',f'Frozen cohort ({len(keys):,} positions at Sep 2025): share still observed',lambda v:f'{v:,.1f}')

# Figures: time series (a) all clean rows, (b) frozen cohort; software vs other, mean mark and share below 99
def series(x):
    x=x.assign(b99=100*(x.mark<99)); return x.groupby(['ddate','sw']).agg(mark=('mark','mean'),b99=('b99','mean'),n=('mark','size')).reset_index()
qq=q[fl].copy(); qq['sw']=qq.sector.eq('software')
sa=series(qq); sf=series(d)
sa.to_csv(f'{O}/ts_all_clean.csv',index=False); sf.to_csv(f'{O}/ts_frozen_cohort.csv',index=False)
print('\n== time series, frozen cohort\n',sf.pivot(index='ddate',columns='sw',values=['mark','b99']).round(2).to_string(),flush=True)
print('\n== time series, all clean rows\n',sa.pivot(index='ddate',columns='sw',values=['mark','b99']).round(2).to_string(),flush=True)
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(2,2,figsize=(7.5,5),sharex=True)
for j,(s_,ttl) in enumerate([(sa,'All loans, funds\' own labels'),(sf,'Frozen Sep 2025 cohort, LLM labels')]):
    for i,(y,yl) in enumerate([('mark','Mean mark'),('b99','Share marked below 99 (%)')]):
        a=ax[i,j]
        for flag,col,lb in [(True,'#c0392b','Software'),(False,'#2c3e50','Other')]:
            z=s_[s_.sw==flag]; a.plot(z.ddate,z[y],marker='o',ms=3,color=col,label=lb)
        a.axvline(pd.Timestamp('2026-03-31'),color='grey',ls='--',lw=.8); a.set_ylabel(yl)
        if i==0: a.set_title(ttl)
ax[0,0].legend(frameon=False); fig.autofmt_xdate(); fig.tight_layout()
fig.savefig(f'{O}/fig_timeseries.pdf'); fig.savefig(f'{O}/fig_timeseries.png',dpi=160)
# Histogram of marks at Sep 2025 vs Jun 2026, frozen cohort
fig,ax=plt.subplots(1,2,figsize=(7.5,2.8),sharey=True)
for a,dt in zip(ax,['2025-09-30','2026-06-30']):
    z=d[d.ddate==dt]
    for flag,col,lb in [(False,'#2c3e50','Other'),(True,'#c0392b','Software')]:
        a.hist(z.loc[z.sw==flag,'mark'].clip(80,102),bins=np.arange(80,102.5,.5),density=True,alpha=.55,color=col,label=lb)
    a.set_title(pd.Timestamp(dt).strftime('%b %Y')); a.set_xlabel('Mark (clipped at 80)')
ax[0].legend(frameon=False); fig.tight_layout(); fig.savefig(f'{O}/fig_mark_hist.pdf'); fig.savefig(f'{O}/fig_mark_hist.png',dpi=160)
# Table 6: known data problems (none affects the mark, which is a ratio)
iss=pd.DataFrame({'Rows':[(q.principal>2e9).sum(),(q.spread<0).sum(),(q.spread*1e4>3000).sum(),(~p.mark.between(0,200)).sum(),p.cost.isna().sum(),p.sector.isna().sum()],
  'What it is':['Principal and cost both scaled up (TCW Mar-Jun 2023, Runway Mar 2023); mark unaffected','Negative spread (floors or parsing); only in the pre-shock spread control',
  'Spread above 3,000bp (parsing); only in the pre-shock spread control','Raw mark outside 0-200 (units errors); removed by the filter','Cost missing (raw rows)','No industry in the fund filing']},
  index=['Principal above $2bn (filtered rows)','Spread below 0 (filtered rows)','Spread above 3,000bp (filtered rows)','Mark outside 0-200 (raw)','Cost missing (raw)','Fund industry missing (raw)'])
tex(iss,'tab6_data_issues','Known data problems and how they are handled',lambda v:f'{v:,.0f}',colfmt='p{4.5cm}rp{8.5cm}')
print('done',flush=True)
