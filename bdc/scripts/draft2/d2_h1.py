# Step 1 (H1) numbers for draft 2. Frozen Sep 2025 cohort, held at each date. June 2026 is the quarter of the text.
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..','..')))  # repo root
from d2_common import *
from scipy.stats import norm

def run(p,T,log):
    keys=cohort_keys(p); d0=frozen(p,keys); d0['sw']=(d0.grp=='SOFT').astype(float)
    S='H1 sample: frozen cohort (model labels), held at each date, position + fund x quarter + vintage x quarter FE, borrower-clustered'
    R={}
    def go(nm,d,**kw):
        r=es(d,**kw); R[nm]=r; log(f'[H1] {nm}: '+sample_note(r,nm)+' | mark Mar %+.2f (%.2f) Jun %+.2f (%.2f) | b99 Mar %+.1f (%.1f) Jun %+.1f (%.1f)'%(*get(r,'mark','2026-03'),*get(r,'mark','2026-06'),*get(r,'b99','2026-03'),*get(r,'b99','2026-06'))); return r
    r=go('headline',d0)
    for y,Y in [('mark','Mark'),('b99','Below')]:
        for q,Q in [('2025-12','Dec'),('2026-03','Mar'),('2026-06','Jun')]:
            b,s=get(r,y,q); putse(f'Hone{Y}{Q}',b,s,S,d=2 if y=='mark' else 1)
    put('HoneMarkFirstPre',get(r,'mark','2022-12')[0],'f1',S,sign=True)
    n=r['n']; put('NHonePos',n['pos'],'int',S); put('NHoneSw',n['sw'],'int',S); put('NHoneObs',n['obs'],'int',S); put('NHoneFunds',n['funds'],'int',S)
    put('NHoneBorr',n['borrowers'],'int',S); put('NHoneMgr',n['managers'],'int',S)
    T['h1_path']={y:r[(y,'sw')] for y in ['mark','b99']}
    # ---- label versions on the SAME cohort (#45): funds' labels voted on filings to Sep 2025 (#93, #95)
    c=d0.copy(); fl=c.fsec.notna()&~c.fsec.isin(['it_services','tech_other'])
    cf=c[c.pkey.isin(set(c.loc[fl&(c.ddate==BASE),'pkey']))].copy(); cf['sw']=(cf.fsec=='software').astype(float)
    rf=go('fund labels on cohort',cf)
    ag=c[c.pkey.isin(set(c.loc[fl&(c.ddate==BASE)&((c.grp=='SOFT')==(c.fsec=='software')),'pkey']))].copy(); ra=go('both agree',ag)
    # ---- disagreement markdowns (#46): model yes / fund no and the reverse, against both-no, in one regression
    dz=c[c.pkey.isin(set(c.loc[fl&(c.ddate==BASE),'pkey']))].copy()
    dz['myfn']=((dz.grp=='SOFT')&(dz.fsec!='software')).astype(float); dz['mnfy']=((dz.grp!='SOFT')&(dz.fsec=='software')).astype(float)
    dz['both']=((dz.grp=='SOFT')&(dz.fsec=='software')).astype(float); dz['sw']=dz.both
    rd=es(dz,swcols=('both','myfn','mnfy')); R['disagree']=rd
    u=dz.drop_duplicates('pkey'); log(f'[H1] disagreement groups: both {int(u.both.sum())}, model-yes-fund-no {int(u.myfn.sum())}, model-no-fund-yes {int(u.mnfy.sum())}')
    # ---- rerun labels (8 Nov, 500 borrowers): apply run-2 labels to those borrowers (#27, #92)
    rr=_rerun_labels()
    c2=d0.copy(); flip=c2.bid.map(rr)
    c2['sw']=np.where(flip.notna(),(flip=='SOFT').astype(float),c2.sw)
    sw9a=d0[d0.ddate==BASE].set_index('pkey').sw; sw9b=c2[c2.ddate==BASE].set_index('pkey').sw
    nsw=int((sw9a!=sw9b).sum()); put('NRerunSwitch',nsw,'int','cohort positions whose software flag differs under the 8 Nov rerun labels (500 borrowers)')
    put('NRerunBorr',int(len(rr)),'int','borrowers in the 8 Nov rerun');
    c2=c2[~c2.bid.map(rr).isin(['UNKN','FUND'])]
    rre=go('rerun labels',c2)
    # ---- robustness rows
    d1=d0[~d0.grp.isin(['ITSV','DATA','HARD'])]; go('no adjacent',d1)
    dw=d0.copy(); dw['w']=dw.prin9; go('principal weighted',dw,w='w')
    rmq=go('manager x quarter',d0,fe=('pkey','mq','vq'))
    go('two-way SE',d0,clu=['clu','family'])
    big=d0[d0.prin9>=1e6]; go('1m floor',big)
    rev=d0.ident.fillna('').str.lower().str.contains(r'revolv|\brcf\b|delayed|\bddtl\b',regex=True)
    norev=d0[~d0.pkey.isin(set(d0.loc[rev&(d0.ddate==BASE),'pkey']))]; go('no revolvers',norev)
    mx=d0[d0.ddate>BASE].groupby('pkey').principal.max(); p9=d0.drop_duplicates('pkey').set_index('pkey').prin9
    grown=set(mx[mx>1.005*p9.reindex(mx.index)].index); go('principal not grown',d0[~d0.pkey.isin(grown)])
    bal=set(d0[d0.ddate==JUN].pkey); go('held in June',d0[d0.pkey.isin(bal)])
    hasm=set(d0.loc[d0.ttm.notna(),'pkey']); go('maturity reported',d0[d0.pkey.isin(hasm)]); go('maturity not reported',d0[~d0.pkey.isin(hasm)])
    go('maturity reported, ttm control',d0[d0.pkey.isin(hasm)],extra='ttm')
    # ---- leave one manager out (T8): 25 largest managers by Sep 2025 principal
    s9=d0[d0.ddate==BASE]; mp=s9.groupby('family').principal.sum().sort_values(ascending=False)
    put('BxShare',mp.get('BLACKSTONE',0)/mp.sum(),'pct0','Blackstone share of cohort principal at Sep 2025')
    put('TopMgrName','Blackstone' if mp.index[0]=='BLACKSTONE' else mp.index[0].title(),'str','largest manager by cohort principal')
    loo=[]
    for fam in mp.index[:25]:
        rl=es(d0[d0.family!=fam],ys=('mark','b99')); loo.append((fam,*get(rl,'mark','2026-06'),*get(rl,'b99','2026-06')))
    L=pd.DataFrame(loo,columns=['fam','m','ms','b','bs']); T['loo']=L
    log('[H1] leave-one-manager-out (25 largest), June: mark %.2f to %.2f, b99 %.1f to %.1f'%(L.m.min(),L.m.max(),L.b.min(),L.b.max()))
    put('LooMarkLo',L.m.min(),'f2','LOO 25 largest managers, June mark',sign=True); put('LooMarkHi',L.m.max(),'f2','LOO June mark',sign=True)
    put('LooBelowLo',L.b.min(),'f1','LOO June b99',sign=True); put('LooBelowHi',L.b.max(),'f1','LOO June b99',sign=True)
    put('LooBxMark',float(L.loc[L.fam=='BLACKSTONE','m'].iloc[0]),'f2','H1 June mark without Blackstone',sign=True)
    # ---- manager x quarter row at three decimals (#53) + FE groups
    put('HoneMqMarkJunThree',get(rmq,'mark','2026-06')[0],'f3','manager x quarter FE, June mark',sign=True)
    put('HoneMarkJunThree',get(r,'mark','2026-06')[0],'f3','headline June mark, 3 decimals',sign=True)
    put('NFqGroups',d0.fq.nunique(),'int','fund x quarter groups in the H1 sample'); put('NMqGroups',d0.mq.nunique(),'int','manager x quarter groups')
    # ---- maturity: which funds report it (#55)
    s9=d0[d0.ddate==BASE].copy(); s9['hm']=s9.ttm.notna()
    put('MatShare',s9.hm.mean(),'pct0','share of cohort positions with a maturity'); put('MatSharePrin',(s9.principal*s9.hm).sum()/s9.principal.sum(),'pct0','principal share with maturity')
    fm=s9.groupby('family').agg(h=('hm','mean'),pr=('principal','sum')); full=fm[fm.h>=0.9]; none=fm[fm.h<=0.1]
    put('NMgrMatFull',len(full),'int','managers reporting maturity for 90%+ of positions'); put('NMgrMatNone',len(none),'int','managers reporting maturity for 10% or less')
    put('MatFullPrin',full.pr.sum()/fm.pr.sum(),'pct0','principal share of the full-reporting managers')
    T['mat_funds']=fm.sort_values('pr',ascending=False)
    log('[H1] maturity reporting by manager (share of positions, principal $bn):\n'+fm.sort_values('pr',ascending=False).assign(pr=lambda x:(x.pr/1e9).round(1)).round(2).head(30).to_string())
    # June macros for robustness rows quoted in the text
    RN={'fund labels on cohort':'Fund','both agree':'Agree','principal weighted':'Pw','maturity reported':'MatYes','maturity not reported':'MatNo',
        'held in June':'Held','no adjacent':'NoAdj','rerun labels':'Rerun'}
    for k,nm in RN.items():
        for y,Y in [('mark','Mark'),('b99','Below')]:
            putse(f'Hone{nm}{Y}Jun',*get(R[k],y,'2026-06'),f'H1 June, {k}',d=2 if y=='mark' else 1)
    for c_,nm in [('myfn','Myfn'),('mnfy','Mnfy'),('both','Both')]:
        for y,Y in [('mark','Mark'),('b99','Below')]:
            putse(f'Dis{nm}{Y}Jun',*get(R['disagree'],y,'2026-06',c_),f'H1 June, disagreement group {c_} against both-not-software',d=2 if y=='mark' else 1)
    T['h1']=R
    return d0,R

def _rerun_labels():
    """8 Nov rerun (500 random borrowers, same prompt, same model alias via subagents): bid -> run-2 group."""
    import glob
    D=(_ROOT+'/bdc/industry_llm/rerun-2026-11-08')
    o=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{D}/out_*.csv'))])
    L=pd.read_parquet(f'{F}/labels.parquet')[['item','bid']]
    return o.merge(L,on='item').drop_duplicates('bid').set_index('bid').group

def pyfixest_check(p,T,log):
    """#101: replicate the headline June coefficients with pyfixest (same regressors, FE and borrower clusters)."""
    import pyfixest as pf
    keys=cohort_keys(p); d=frozen(p,keys).copy(); d['sw']=(d.grp=='SOFT').astype(float)
    ev=[q for q in sorted(d.ddate.unique()) if q!=BASE]; X=[]
    for q in ev:
        k=q.strftime('%Y%m'); dq=(d.ddate==q).astype(float)
        d[f'sw_{k}']=d.sw*dq; d[f'sp{k}']=d.spread_pre.fillna(0)*100*dq; d[f'spm{k}']=d.spread_pre.isna()*dq; d[f'pk{k}']=d.pik_pre.fillna(0)*dq
        X+=[f'sw_{k}',f'sp{k}',f'spm{k}',f'pk{k}']
    for y,Y,dd_ in [('mark','Mark',2),('b99','Below',1)]:
        m=pf.feols(f"{y} ~ {' + '.join(X)} | pkey + fq + vq",data=d,vcov={'CRV1':'clu'},fixef_rm='singleton')
        b=float(m.coef()['sw_202606']); s=float(m.se()['sw_202606'])
        putse(f'Pyfx{Y}Jun',b,s,f'pyfixest {pf.__version__} replication of the headline, June 2026',d=dd_)
        log(f'[pyfixest] {y} June {b:+.3f} ({s:.3f})')
    put('PyfxVersion',pf.__version__,'str','pyfixest version')
