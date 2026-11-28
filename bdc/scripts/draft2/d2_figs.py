# Figures for draft 2, all from the same objects as numbers.tex. Data to 30 June 2026.
import os
from d2_common import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
SW,OT='#eb6834','#2a78d6'; INK,INK2,MUTED,GRID,AXIS='#0b0b0b','#52514e','#898781','#e1e0d9','#c3c2b7'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.edgecolor':AXIS,'axes.labelcolor':INK2,'xtick.color':MUTED,'ytick.color':MUTED,
    'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':GRID,'grid.linewidth':0.6,'axes.axisbelow':True,'legend.frameon':False,'pdf.fonttype':42})
def tz(ax):
    ax.axvspan(DEC,MAR,color='#f1f0ec',zorder=0,lw=0)
def yr(ax,fmt='%Y'):
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter(fmt)); ax.tick_params(axis='x',labelsize=7.5)
def save(fig,out,name):
    fig.savefig(f'{out}/{name}.pdf',bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None}); fig.savefig(f'{out}/{name}.png',dpi=200,bbox_inches='tight'); plt.close(fig)

def make(p,T,out,log):
    os.makedirs(out,exist_ok=True)
    d0=frozen(p,cohort_keys(p)); d0['sw']=(d0.grp=='SOFT').astype(float)
    # Figure 1: raw cohort
    m=d0.groupby(['ddate','sw']).agg(mark=('mark','mean'),b99=('b99','mean')).reset_index()
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.8))
    for ax,(y,ttl) in zip(axs,[('b99','(a) Share marked below 99 (%)'),('mark','(b) Mean mark (points per 100 of principal)')]):
        for s,c,nm in [(1,SW,'Software'),(0,OT,'Other loans')]:
            x=m[m.sw==s]; ax.plot(x.ddate,x[y],color=c,lw=2,marker='o',ms=3.5,label=nm)
        ax.set_title(ttl,fontsize=9,color=INK,loc='left'); tz(ax); yr(ax)
    axs[0].legend(loc='upper left',fontsize=7.5); fig.tight_layout(); save(fig,out,'fig1_cohort')
    # Figure 2: equity indices and traded loan prices
    E=pd.read_csv(f'{V}/equity_quarter_ends_2026-11-17.csv',parse_dates=['date'])
    q=T['q2']; w=q[q.px.notna()].groupby(['ddate','sw']).px.mean().reset_index()
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.8))
    ax=axs[0]
    for col,c,nm in [('software',SW,'Software equities (IGV)'),('broad',OT,'S&P 500 (SPY)')]:
        b=E.loc[E.date==BASE,col].iloc[0]; ax.plot(E.date,100*E[col]/b,color=c,lw=2,marker='o',ms=3.5,label=nm)
    ax.axhline(100,color=AXIS,lw=1); ax.set_title('(a) Equity prices, 30 Sep 2025 = 100',fontsize=9,color=INK,loc='left'); tz(ax); yr(ax,'%b %y'); ax.legend(loc='lower left',fontsize=7.5)
    ax=axs[1]
    for s,c,nm in [(1,SW,'Software loans'),(0,OT,'Other loans')]:
        x=w[w.sw==s]; ax.plot(x.ddate,x.px,color=c,lw=2,marker='o',ms=3.5,label=nm)
    ax.set_title('(b) Mean N-PORT price of traded loans (points)',fontsize=9,color=INK,loc='left'); tz(ax); yr(ax,'%b %y'); ax.legend(loc='lower left',fontsize=7.5)
    fig.tight_layout(); save(fig,out,'fig2_shock')
    # Figure 3: event study
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.8))
    for ax,(y,ttl) in zip(axs,[('b99','(a) Below 99: software minus other (pp)'),('mark','(b) Mark: software minus other (points)')]):
        e=T['h1_path'][y].copy(); e.index=pd.to_datetime(e.index)+pd.offsets.MonthEnd(0); e.loc[BASE]=[0.0,0.0]; e=e.sort_index()
        ax.axhline(0,color=AXIS,lw=1); ax.fill_between(e.index,e.b-1.96*e.se,e.b+1.96*e.se,color=SW,alpha=0.18,lw=0)
        ax.plot(e.index,e.b,color=SW,lw=2,marker='o',ms=3.5); ax.plot([BASE],[0],marker='o',ms=5,mfc='white',mec=SW)
        ax.set_title(ttl,fontsize=9,color=INK,loc='left'); tz(ax); yr(ax)
    fig.tight_layout(); save(fig,out,'fig3_event_study')
    # Figure 4: scaled rank
    from d2_rank import NAMES
    S=T['rank']; fig,axs=plt.subplots(1,2,figsize=(7.4,4.6))
    for ax,(y,ttl,asc) in zip(axs,[('db99','(a) Share below 99',True),('dmark','(b) Mark',False)]):
        t=S[S.y==y].sort_values('T',ascending=asc).reset_index(drop=True)
        ax.axvline(0,color=AXIS,lw=1); ax.scatter(t['T'],range(len(t)),s=22,color=[SW if g=='SOFT' else OT for g in t.grp],zorder=3)
        ax.set_yticks(range(len(t))); ax.set_yticklabels(t.grp.map(NAMES),fontsize=7); ax.set_title(ttl,fontsize=9,color=INK,loc='left')
        ax.set_xlabel('scaled change, Sep 2025 to Jun 2026',fontsize=7.5); ax.grid(axis='y',visible=False)
    fig.tight_layout(); save(fig,out,'fig4_rank')
    # Figure 5: (a) Step 2 marks and prices, held and priced at Sep 2025 and at the date; (b) Step 3 classes
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.9))
    ax=axs[0]; qq=q.copy(); b0=qq[qq.ddate==BASE].set_index('pkey'); qq['dm']=qq.mark-qq.pkey.map(b0.mark); qq['dp']=qq.px-qq.pkey.map(b0.px)
    qq=qq[qq.px.notna()]
    g=qq.groupby(['ddate','sw']).agg(dm=('dm','mean'),dp=('dp','mean')).reset_index()
    for s,c,nm in [(1,SW,'Software'),(0,OT,'Other')]:
        x=g[g.sw==s]; ax.plot(x.ddate,x.dm,color=c,lw=2,marker='o',ms=3.5,label=f'{nm}: BDC mark'); ax.plot(x.ddate,x.dp,color=c,lw=1.4,ls='--',marker='s',ms=3,label=f'{nm}: market price')
    ax.axhline(0,color=AXIS,lw=1); ax.set_title('(a) Same loan: mark and price (points)',fontsize=9,color=INK,loc='left'); tz(ax); yr(ax,'%b %y'); ax.legend(fontsize=6.5,loc='lower left')
    ax=axs[1]; s3=T['s3']; rows=[]
    for c_ in ['traded','private']:
        for s in [1,0]:
            x=s3[(s3.cls==c_)&(s3.sw==s)]
            for qn,col in [(pd.Timestamp('2025-06-30'),'dm202506'),(BASE,None),(DEC,'dm202512'),(MAR,'dm202603'),(JUN,'dm202606')]:
                v=0.0 if col is None else (-x[col].mean() if col=='dm202506' else x[col].mean()); rows.append((c_,s,qn,v))
    R=pd.DataFrame(rows,columns=['cls','sw','q','v'])
    for (c_,s),x in R.groupby(['cls','sw']):
        ax.plot(x.q,x.v,color=SW if s else OT,lw=2 if c_=='traded' else 1.4,ls='-' if c_=='traded' else ':',marker='o',ms=3,label=f'{"Software" if s else "Other"}, {c_}')
    ax.axhline(0,color=AXIS,lw=1); ax.set_title('(b) Traded and private loans (points)',fontsize=9,color=INK,loc='left'); tz(ax); yr(ax,'%b %y'); ax.legend(fontsize=6.5,loc='lower left')
    fig.tight_layout(); save(fig,out,'fig5_market')
    log('[figs] written to '+out)
