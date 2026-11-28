# Draft 6 (26 Nov 2026): redraws Figures 1 to 5 of draft 5 with the fixes asked for in reviewer notes, plus one new
# appendix figure. Same data and estimates as d2_figs.make (data to 30 June 2026; the September 2026 point in Figure 3 is
# read from numbers-sep.tex, written by draft6/d6_sep_numbers.py). Changes:
#   #87/#108 quarter-end ticks labelled 'Dec 22', 'Jun 23' ... (points sit on their quarter-end, not on month ticks)
#   #65/#145 Figure 2(a) from the iShares NAV total-return series (IGV, IVV) instead of Digrin; #155 a dotted line at
#            3 Feb 2026, the selloff Reuters reported (S&P 500 software and services index nearly -4% that day)
#   #89 Figure 3 adds the registered September 2026 coefficient (persistence check, build A) in another colour
#   #94 Figure 4 prints each group's number of borrowers next to its name
#   #86 new Figure A2: histogram of the June 2026 quarterly mark change, software against other cohort loans
# Usage: python3 d6_figs.py <draft6 folder>
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..','..')))  # repo root
import sys, os, re
sys.path.insert(0,(_ROOT+'/bdc/scripts/draft2'))
from d2_common import *
import d2_step2, d2_step3 as S3
from d2_rank import NAMES, changes, c_t
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.dates as mdates
SW,OT,SEPC='#eb6834','#2a78d6','#1d8a5a'; INK,INK2,MUTED,GRID,AXIS='#0b0b0b','#52514e','#898781','#e1e0d9','#c3c2b7'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.edgecolor':AXIS,'axes.labelcolor':INK2,'xtick.color':MUTED,'ytick.color':MUTED,
    'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':GRID,'grid.linewidth':0.6,'axes.axisbelow':True,'legend.frameon':False,'pdf.fonttype':42})
OUT=sys.argv[1]; FIG=f'{OUT}/figures'
def tz(ax): ax.axvspan(DEC,MAR,color='#f1f0ec',zorder=0,lw=0)
def qticks(ax,start,end,every=2):
    q=pd.date_range(start,end,freq='QE'); q=q[::every] if every>1 else q
    ax.set_xticks(q); ax.set_xticklabels([d.strftime('%b %y') for d in q],fontsize=7); ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[3,6,9,12],bymonthday=-1))
def save(fig,name):
    fig.savefig(f'{FIG}/{name}.pdf',bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None}); fig.savefig(f'{FIG}/{name}.png',dpi=200,bbox_inches='tight'); plt.close(fig)
def mac(f,n): return float(re.search(r'\\newcommand\{\\'+n+r'\}\{\$?([-+0-9.]+)',open(f).read()).group(1))
p=load(); assert p.ddate.max()==JUN
d0=frozen(p,cohort_keys(p)); d0['sw']=(d0.grp=='SOFT').astype(float)
# ---------------- Figure 1
m=d0.groupby(['ddate','sw']).agg(mark=('mark','mean'),b99=('b99','mean')).reset_index()
fig,axs=plt.subplots(1,2,figsize=(7.2,2.8))
for ax,(y,ttl) in zip(axs,[('b99','(a) Share marked below 99 (%)'),('mark','(b) Mean mark (points per 100 of principal)')]):
    for s,c,nm in [(1,SW,'Software'),(0,OT,'Other loans')]:
        x=m[m.sw==s]; ax.plot(x.ddate,x[y],color=c,lw=2,marker='o',ms=3.5,label=nm)
    ax.set_title(ttl,fontsize=9,color=INK,loc='left'); tz(ax); qticks(ax,m.ddate.min(),JUN)
axs[0].legend(loc='upper right',fontsize=7.5); fig.tight_layout(); save(fig,'fig1_cohort')
# ---------------- Figure 2
E=pd.read_csv(f'{V}/equity_quarter_ends_ishares_2026-11-26.csv',parse_dates=['date']).set_index('date')
q=d2_step2.build(p,'nport_matches_coupon.parquet'); w=q[q.px.notna()].groupby(['ddate','sw']).px.mean().reset_index()
fig,axs=plt.subplots(1,2,figsize=(7.2,2.8)); ax=axs[0]
qe=E[E.index.is_quarter_end|(E.index==pd.Timestamp('2026-09-30'))]
qe=E.loc[[d for d in E.index if d.month in (3,6,9,12) and d.day>=30 or (d.month==3 and d.day==31)]]
for col,c,nm in [('software',SW,'Software equities (IGV)'),('broad',OT,'S&P 500 (IVV)')]:
    b=E.loc[BASE,col]; ax.plot(qe.index,100*qe[col]/b,color=c,lw=2,marker='o',ms=3.5,label=nm)
    for d_ in ['2026-02-03']: ax.plot([pd.Timestamp(d_)],[100*E.loc[d_,col]/b],marker='x',ms=4,color=c,lw=0)
ax.axvline(pd.Timestamp('2026-02-03'),color=MUTED,lw=0.8,ls=':')
ax.text(pd.Timestamp('2026-02-06'),ax.get_ylim()[1]*0.995,'3 Feb 2026',fontsize=6.5,color=INK2,va='top')
ax.axhline(100,color=AXIS,lw=1); ax.set_title('(a) Equity total return, 30 Sep 2025 = 100',fontsize=9,color=INK,loc='left'); tz(ax); qticks(ax,qe.index.min(),qe.index.max()); ax.legend(loc='lower left',fontsize=7.5)
ax=axs[1]
for s,c,nm in [(1,SW,'Software loans'),(0,OT,'Other loans')]:
    x=w[w.sw==s]; ax.plot(x.ddate,x.px,color=c,lw=2,marker='o',ms=3.5,label=nm)
ax.set_title('(b) Mean N-PORT price of traded loans (points)',fontsize=9,color=INK,loc='left'); tz(ax); qticks(ax,w.ddate.min(),JUN,1); ax.legend(loc='lower left',fontsize=7.5)
fig.tight_layout(); save(fig,'fig2_shock')
# ---------------- Figure 3 (+ September 2026 point)
r=es(d0); SEPF=f'{OUT}/numbers-sep.tex'; SEPD=pd.Timestamp('2026-09-30')
fig,axs=plt.subplots(1,2,figsize=(7.2,2.8))
for ax,(y,ttl,M) in zip(axs,[('b99','(a) Below99: software minus other (pp)','PersBelow'),('mark','(b) Mark: software minus other (points)','PersMark')]):
    e=r[(y,'sw')].copy(); e.index=pd.to_datetime(e.index)+pd.offsets.MonthEnd(0); e.loc[BASE]=[0.0,0.0]; e=e.sort_index()
    ax.axhline(0,color=AXIS,lw=1); ax.fill_between(e.index,e.b-1.96*e.se,e.b+1.96*e.se,color=SW,alpha=0.18,lw=0)
    ax.plot(e.index,e.b,color=SW,lw=2,marker='o',ms=3.5); ax.plot([BASE],[0],marker='o',ms=5,mfc='white',mec=SW)
    b,s=mac(SEPF,M),mac(SEPF,M+'SE'); ax.errorbar([SEPD],[b],yerr=[[1.96*s],[1.96*s]],fmt='D',ms=4,color=SEPC,capsize=2,lw=1.2,label='Sep 2026 (registered quarter)')
    ax.set_title(ttl,fontsize=9,color=INK,loc='left'); tz(ax); qticks(ax,e.index.min(),SEPD)
axs[1].legend(loc='lower left',fontsize=7); fig.tight_layout(); save(fig,'fig3_event_study')
# ---------------- Figure 4 (borrower counts)
ch=changes(p); ch=ch[ch.grp.notna()&~ch.grp.isin(['UNKN','FUND','ITSV','DATA','HARD'])]
qs=sorted(ch.ddate.unique()); pre=[x for x in qs if x<=BASE]; post=[x for x in qs if x>BASE]; rows=[]
for g in sorted(ch.grp.unique()):
    x=ch if g=='SOFT' else ch[ch.grp!='SOFT']
    for y in ['dmark','db99']:
        c=pd.Series({t:c_t(x[x.ddate==t],y,(x[x.ddate==t].grp==g).astype(float)) for t in qs}); rows.append(dict(grp=g,y=y,T=c[post].sum()/c[pre].std()/np.sqrt(3)))
S=pd.DataFrame(rows); nb=ch[ch.ddate.isin(post)].groupby('grp').bid.nunique()
fig,axs=plt.subplots(1,2,figsize=(7.6,4.6))
for ax,(y,ttl,asc) in zip(axs,[('db99','(a) Share below 99',True),('dmark','(b) Mark',False)]):
    t=S[S.y==y].sort_values('T',ascending=asc).reset_index(drop=True)
    ax.axvline(0,color=AXIS,lw=1); ax.scatter(t['T'],range(len(t)),s=22,color=[SW if g=='SOFT' else OT for g in t.grp],zorder=3)
    ax.set_yticks(range(len(t))); ax.set_yticklabels([f'{NAMES[g]} ({nb[g]})' for g in t.grp],fontsize=7); ax.set_title(ttl,fontsize=9,color=INK,loc='left')
    ax.set_xlabel('scaled change, Sep 2025 to Jun 2026',fontsize=7.5); ax.grid(axis='y',visible=False)
fig.tight_layout(); save(fig,'fig4_rank')
# ---------------- Figure 5
fig,axs=plt.subplots(1,2,figsize=(7.2,2.9))
ax=axs[0]; qq=q.copy(); b0=qq[qq.ddate==BASE].set_index('pkey'); qq['dm']=qq.mark-qq.pkey.map(b0.mark); qq['dp']=qq.px-qq.pkey.map(b0.px); qq=qq[qq.px.notna()]
g=qq.groupby(['ddate','sw']).agg(dm=('dm','mean'),dp=('dp','mean')).reset_index()
for s,c,nm in [(1,SW,'Software'),(0,OT,'Other')]:
    x=g[g.sw==s]; ax.plot(x.ddate,x.dm,color=c,lw=2,marker='o',ms=3.5,label=f'{nm}: BDC mark'); ax.plot(x.ddate,x.dp,color=c,lw=1.4,ls='--',marker='s',ms=3,label=f'{nm}: market price')
ax.axhline(0,color=AXIS,lw=1); ax.set_title('(a) Same loan: mark and price (points)',fontsize=9,color=INK,loc='left'); tz(ax); qticks(ax,g.ddate.min(),JUN,1); ax.legend(fontsize=6.5,loc='lower left')
ax=axs[1]; s3=S3.frame(p); rows=[]
for c_ in ['traded','private']:
    for s in [1,0]:
        x=s3[(s3.cls==c_)&(s3.sw==s)]
        for qn,col in [(pd.Timestamp('2025-06-30'),'dm202506'),(BASE,None),(DEC,'dm202512'),(MAR,'dm202603'),(JUN,'dm202606')]:
            v=0.0 if col is None else (-x[col].mean() if col=='dm202506' else x[col].mean()); rows.append((c_,s,qn,v))
R=pd.DataFrame(rows,columns=['cls','sw','q','v'])
for (c_,s),x in R.groupby(['cls','sw']):
    ax.plot(x.q,x.v,color=SW if s else OT,lw=2 if c_=='traded' else 1.4,ls='-' if c_=='traded' else ':',marker='o',ms=3,label=f'{"Software" if s else "Other"}, {c_}')
ax.axhline(0,color=AXIS,lw=1); ax.set_title('(b) Traded and private loans (points)',fontsize=9,color=INK,loc='left'); tz(ax); qticks(ax,pd.Timestamp('2025-06-30'),JUN,1); ax.legend(fontsize=6.5,loc='lower left')
fig.tight_layout(); save(fig,'fig5_market')
# ---------------- Figure A2 (#86): change in mark from 30 Sep 2025 to 30 Jun 2026, cohort loans held at both dates
x=d0[d0.ddate.isin([BASE,JUN])].pivot_table(index='pkey',columns='ddate',values='mark').dropna(); x['d']=x[JUN]-x[BASE]
x['sw']=x.index.map(d0.drop_duplicates('pkey').set_index('pkey').sw)
fig,ax=plt.subplots(figsize=(5.4,2.8)); bins=np.arange(-20.5,6.6,1)
for s,c,nm in [(0,OT,'Other loans'),(1,SW,'Software')]:
    z=x[x.sw==s].d.clip(-20,6); ax.hist(z,bins=bins,weights=np.ones(len(z))*100/len(z),color=c,alpha=0.55 if s==0 else 0.75,label=f'{nm} ({len(z):,} positions)')
ax.set_xlabel('change in mark, 30 Sep 2025 to 30 Jun 2026 (points; clipped at -20 and +6)',fontsize=7.5); ax.set_ylabel('% of positions',fontsize=7.5); ax.legend(fontsize=7)
fig.tight_layout(); save(fig,'figA2_hist')
print('figures written; Sep25-Jun26 change median', {s:round(float(x[x.sw==s].d.median()),2) for s in [0,1]}, 'share below -5', {s:round(float((x[x.sw==s].d<-5).mean()),3) for s in [0,1]}, 'share 0', {s:round(float((x[x.sw==s].d.abs()<0.5).mean()),3) for s in [0,1]})
# ---------------- macros for the text (numbers-d6fig.tex): equity moves from the iShares series and Figure A2 shares
sys.path.insert(0,(_ROOT+'/bdc/scripts/draft4')); import d4_common as D4
SEPD=pd.Timestamp('2026-09-30')
for col,nm in [('software','Sw'),('broad','Br')]:
    s=E[col]
    for a,b_,K in [(BASE,DEC,'SepDec'),(DEC,MAR,'DecMar'),(BASE,MAR,'SepMar'),(MAR,JUN,'MarJun'),(JUN,SEPD,'JunSep'),(BASE,JUN,'SepJun')]:
        D4.put4(f'DsixEq{nm}{K}',s[b_]/s[a]-1,'pct0' if K!='JunSep' else 'pct1',f'{col} total return {a:%d %b %Y} to {b_:%d %b %Y}, iShares NAV with distributions',sign=True)
rel=lambda a,b_: np.log(E.software[b_]/E.software[a])-np.log(E.broad[b_]/E.broad[a])
D4.put4('DsixEqRecShare',-rel(JUN,SEPD)/rel(BASE,JUN),'pct0','share of the log relative fall of software equities Sep25-Jun26 won back Jun-Sep26')
for s,nm in [(1,'Sw'),(0,'Ot')]:
    z=x[x.sw==s].d; D4.put4(f'DsixHist{nm}Zero',(z.abs()<0.5).mean(),'pct0',f'Figure A2: share of {nm} cohort positions whose mark moved less than half a point Sep25-Jun26')
    D4.put4(f'DsixHist{nm}Five',(z<-5).mean(),'pct0','share that fell more than 5 points'); D4.put4(f'DsixHist{nm}Med',z.median(),'f1','median change',sign=True)
h=D4.write_macros(f'{OUT}/numbers-d6fig.tex','% numbers-d6fig.tex: equity moves (iShares NAV total return, bdc/v2/equity_quarter_ends_ishares_2026-11-26.csv) and Figure A2 shares. Written by bdc/scripts/draft6/d6_figs.py.\n')
print('numbers-d6fig.tex',h[:16]); print({k:v[0] for k,v in D4.MAC4.items()})
