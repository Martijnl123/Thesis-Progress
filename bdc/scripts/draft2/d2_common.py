# Shared loaders, samples, estimators and the macro registry for draft 2 (v2_numbers.py). Written 17 Nov 2026.
# Data through 30 June 2026 only: every loader asserts it. No September 2026 file is opened.
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..','..')))  # repo root
import sys, re, warnings, numpy as np, pandas as pd
sys.path.insert(0,(_ROOT+'/bdc/scripts'))
from ai_rev2_lib import fit, se, vcov
warnings.filterwarnings('ignore')
V=(_ROOT+'/bdc/v2'); F=(_ROOT+'/bdc/industry_llm/full')
BASE=pd.Timestamp('2025-09-30'); JUN=pd.Timestamp('2026-06-30'); MAR=pd.Timestamp('2026-03-31'); DEC=pd.Timestamp('2025-12-31')
CAP=pd.Timestamp('2026-06-30')

# ---------------- macro registry ----------------
MAC={}   # name -> (latex value, raw value, sample note)
PCT_SIGN=False   # draft 3 (v3_numbers.py) sets True so percent kinds print with \%; False keeps draft 2 reproducible
def _name_ok(n): assert re.fullmatch(r'[A-Za-z]+',n), f'bad macro name {n}'; assert n not in MAC, f'duplicate macro {n}'
def put(name,raw,kind='f2',sample='',sign=False):
    """kind: f0/f1/f2/f3 decimals, int (thousands sep), pct0/pct1 (raw is a share, printed x100), str."""
    _name_ok(name)
    if kind=='str': tex=str(raw)
    elif kind=='int': tex=f'{int(round(raw)):,}'.replace(',','{,}')
    elif kind.startswith('pct'): tex=_num(100*raw,int(kind[3:]),sign)+('\\%' if PCT_SIGN else '')
    else: tex=_num(raw,int(kind[1:]),sign)
    MAC[name]=(tex,raw,sample); return tex
def _num(x,d,sign):
    if x is None or (isinstance(x,float) and np.isnan(x)): raise ValueError('NaN macro')
    s=f'{x:+,.{d}f}' if sign else f'{x:,.{d}f}'
    if s in ('-0','-0.0','-0.00','-0.000'): s=s[1:]
    s=s.replace(',','{,}')
    return f'${s}$' if (s.startswith('-') or s.startswith('+')) else s
def putse(name,b,s,sample,d=2):
    """coefficient with sign and its SE: macros <name> and <name>SE"""
    put(name,b,f'f{d}',sample,sign=True); put(name+'SE',s,f'f{d}',sample)

# ---------------- data ----------------
def load():
    p=pd.read_parquet(f'{V}/panel.parquet'); assert p.ddate.max()<=CAP
    pos=pd.read_parquet(f'{V}/positions.parquet',columns=['pos_id','ident','maturity','filed','form']); p=p.merge(pos,on='pos_id',how='left')
    p['fq']=p.cik.astype(str)+'_'+p.ddate.dt.strftime('%Y%m'); p['mq']=p.family.astype(str)+'_'+p.ddate.dt.strftime('%Y%m')
    p['vint']=p.pkey.map(p.groupby('pkey').ddate.min()).dt.strftime('%Y%m'); p['vq']=p.vint+'|'+p.ddate.dt.strftime('%Y%m')
    L=pd.read_parquet(f'{F}/labels.parquet')[['bid','group']]
    pb=pd.read_parquet(f'{F}/position_borrower.parquet').merge(L,on='bid').merge(p[['pos_id','pkey']],on='pos_id').drop_duplicates('pkey').set_index('pkey')
    p['grp']=p.pkey.map(pb.group); p['bid']=p.pkey.map(pb.bid)
    p['clu']=np.where(p.bid.notna(),'b'+p.bid.astype(str),'l'+p.loan.astype(str))
    p['b99']=100*(p.mark<99)
    fl=pd.read_parquet(f'{V}/fund_label_pre_2026-11-16.parquet').set_index('pkey'); p['fsec']=p.pkey.map(fl.sector_pre)
    s9=p[(p.ddate==BASE)&p.ok].set_index('pkey')
    p['mat9']=p.pkey.map(pd.to_datetime(s9.maturity,errors='coerce')); p['ttm']=(p.mat9-BASE).dt.days/365.25
    p.loc[~p.ttm.between(0,15),'ttm']=np.nan
    p['prin9']=p.pkey.map(s9.principal); p['mark9']=p.pkey.map(s9.mark)
    return p

def model_lab(p): return p.grp.notna()&~p.grp.isin(['UNKN','FUND'])
def cohort_keys(p): return set(p.loc[(p.ddate==BASE)&p.ok&model_lab(p),'pkey'])
def frozen(p,keys):
    fz=p[p.pkey.isin(keys)]; post=fz.ddate>BASE
    return fz[(~post&fz.ok)|(post&fz.mark.between(0,110)&(fz.principal>=250e3))].copy()

# ---------------- Step 1 event study (PAP v5 headline spec) ----------------
def es(d,swcols=('sw',),fe=('pkey','fq','vq'),clu='clu',w=None,extra=None,ys=('mark','b99')):
    d=d.copy(); qs=sorted(d.ddate.unique()); ev=[q for q in qs if q!=BASE]; X=[]; SW={c:[] for c in swcols}
    for q in ev:
        k=q.strftime('%Y%m'); dq=(d.ddate==q).astype(float)
        for c in swcols: d[f'{c}_{k}']=d[c]*dq; SW[c].append(f'{c}_{k}')
        d[f'sp{k}']=d.spread_pre.fillna(0)*100*dq; d[f'spm{k}']=d.spread_pre.isna()*dq; d[f'pk{k}']=d.pik_pre.fillna(0)*dq
        X+=[f'sp{k}',f'spm{k}',f'pk{k}']
        if extra is not None: d[f'ex{k}']=d[extra]*dq; X.append(f'ex{k}')
    allsw=[x for c in swcols for x in SW[c]]; r={}
    for y in ys:
        dd,b,sc,Ainv=fit(d,y,allsw+X,list(fe),w=w); s=se(dd,sc,Ainv,clu)[:len(allsw)]
        for c in swcols:
            idx=[allsw.index(x) for x in SW[c]]
            r[(y,c)]=pd.DataFrame({'b':b[SW[c]].values,'se':s[idx]},index=[q.strftime('%Y-%m') for q in ev])
        u=dd.drop_duplicates('pkey')
        r['n']=dict(pos=dd.pkey.nunique(),sw=int(u[swcols[0]].sum()),funds=dd.cik.nunique(),obs=len(dd),clusters=dd[clu if isinstance(clu,str) else clu[0]].nunique(),
                    borrowers=u.bid.nunique(),managers=dd.family.nunique())
    return r
def get(r,y,q,c='sw'): x=r[(y,c)].loc[q]; return float(x.b),float(x.se)
def sample_note(r,label): n=r['n']; return f'{label}: {n["pos"]} positions ({n["sw"]} software), {n["funds"]} funds, {n["obs"]} position-quarters'
