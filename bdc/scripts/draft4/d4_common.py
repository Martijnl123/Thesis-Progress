# Shared helpers for draft 4 (23 Nov 2026). Written by the draft 4 session; nothing in draft2/ or draft3/ is changed.
# - put4(): a macro registry for the NEW draft 4 macro files (numbers-pap.tex, numbers-sep.tex, numbers-labels.tex),
#   same formatting rules as d2_common.put() with the percent sign on (as v3_numbers.py sets it), but its own namespace,
#   so a draft 4 file can never redefine a numbers.tex macro by accident (checked again by d4_check_macros.py).
# - es_opts(): a copy of d2_common.es() with the PIK control switchable (PAP v5: "PIK controls with and without").
# - rank_table(): the scaled sector rank of d2_rank.rank() with the group set and the post quarters as arguments.
# - load_sep(): loader for the September 2026 builds (bdc/v2/sep2026/buildA or buildB). Only d4_sep_numbers.py uses it.
#   The June pipeline (v3_numbers.py, d2_common.load()) is untouched and still refuses any row after 30 June 2026.
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..','..')))  # repo root
import sys, re, hashlib, numpy as np, pandas as pd
sys.path.insert(0,(_ROOT+'/bdc/scripts/draft2')); sys.path.insert(0,(_ROOT+'/bdc/scripts'))
import d2_common as CM
from ai_rev2_lib import fit, se

MAC4={}
def _ok(n):
    assert re.fullmatch(r'[A-Za-z]+',n), f'bad macro name {n}'; assert n not in MAC4, f'duplicate macro {n}'
def put4(name,raw,kind='f2',sample='',sign=False):
    _ok(name)
    if kind=='str': tex=str(raw)
    elif kind=='int': tex=f'{int(round(raw)):,}'.replace(',','{,}')
    elif kind.startswith('pct'): tex=CM._num(100*raw,int(kind[3:]),sign)+'\\%'
    else: tex=CM._num(raw,int(kind[1:]),sign)
    MAC4[name]=(tex,raw,sample); return tex
def putse4(name,b,s,sample,d=2):
    put4(name,b,f'f{d}',sample,sign=True); put4(name+'SE',s,f'f{d}',sample)
def write_macros(path,header,extra=''):
    body=header+''.join(f'\\newcommand{{\\{k}}}{{{v[0]}}}\n' for k,v in sorted(MAC4.items()))+extra
    open(path,'w').write(body); return hashlib.sha256(body.encode()).hexdigest()
def write_log(path,title,lines):
    out=[title,'','MACROS (name | value | sample)']+[f'{k} | {v[0]} | {v[2]}' for k,v in sorted(MAC4.items())]+['','LOG']+list(lines)
    open(path,'w').write('\n'.join(out)+'\n')
def sha(path): return hashlib.sha256(open(path,'rb').read()).hexdigest()

# ---------------------------------------------------------------- Step 1 with switchable controls (copy of d2_common.es)
def es_opts(d,fe=('pkey','fq','vq'),clu='clu',pik=True,spread=True,ys=('mark','b99')):
    d=d.copy(); qs=sorted(d.ddate.unique()); ev=[q for q in qs if q!=CM.BASE]; X=[]; SW=[]
    for q in ev:
        k=q.strftime('%Y%m'); dq=(d.ddate==q).astype(float)
        d[f'sw_{k}']=d['sw']*dq; SW.append(f'sw_{k}')
        if spread: d[f'sp{k}']=d.spread_pre.fillna(0)*100*dq; d[f'spm{k}']=d.spread_pre.isna()*dq; X+=[f'sp{k}',f'spm{k}']
        if pik: d[f'pk{k}']=d.pik_pre.fillna(0)*dq; X.append(f'pk{k}')
    r={}
    for y in ys:
        dd,b,sc,Ainv=fit(d,y,SW+X,list(fe)); s=se(dd,sc,Ainv,clu)[:len(SW)]
        r[y]=pd.DataFrame({'b':b[SW].values,'se':s},index=[q.strftime('%Y-%m') for q in ev])
    r['n']=dict(pos=dd.pkey.nunique(),funds=dd.cik.nunique(),sw=int(dd.drop_duplicates('pkey').sw.sum()))
    return r

# ---------------------------------------------------------------- scaled sector rank with a chosen group set
def changes(p):
    ever_ok=set(p.loc[p.ok,'pkey'])
    valid=np.where(p.ddate<=CM.BASE,p.ok,p.mark.between(0,110)&(p.principal>=250e3)&p.pkey.isin(ever_ok))
    v=p[valid].sort_values(['pkey','ddate']).copy(); g=v.groupby('pkey'); v['dmark']=g.mark.diff(); v['db99']=g.b99.diff(); v['gapd']=g.ddate.diff().dt.days
    return v[v.gapd.between(80,100)].copy()
def c_t(x,y,ind):
    s=ind-ind.groupby(x.cik).transform('mean'); e=x[y]-x.groupby('cik')[y].transform('mean'); den=(s*s).sum(); return (s*e).sum()/den if den>0 else np.nan
def rank_table(ch,drop=('UNKN','FUND','ITSV','DATA','HARD'),npost=3):
    """Same statistic as d2_rank.rank(): per group, chain-linked change relative to other loans in the same fund (software left out of
    the comparison loans when another group is treated), summed over the post quarters, divided by sqrt(npost) x SD of the 11 pre changes."""
    ch=ch[ch.grp.notna()&~ch.grp.isin(list(drop))]
    qs=sorted(ch.ddate.unique()); pre=[q for q in qs if q<=CM.BASE]; post=[q for q in qs if q>CM.BASE]; assert len(pre)==11 and len(post)==npost, (len(pre),len(post))
    rows=[]
    for gname in sorted(ch.grp.unique()):
        x=ch if gname=='SOFT' else ch[ch.grp!='SOFT']
        npos=x[x.grp==gname].groupby('ddate').pkey.nunique().mean()
        for y in ['dmark','db99']:
            c=pd.Series({q:c_t(x[x.ddate==q],y,(x[x.ddate==q].grp==gname).astype(float)) for q in qs}); sd=c[pre].std()
            rows.append(dict(grp=gname,y=y,sd=sd,post=c[post].sum(),T=c[post].sum()/sd/np.sqrt(npost),M=npos))
    return pd.DataFrame(rows)
def rank_of(S,y,g='SOFT',col='T'):
    asc=(y=='dmark'); t=S[S.y==y].sort_values(col,ascending=asc).reset_index(drop=True); return int(t.index[t.grp==g][0])+1, len(t), t

# ---------------------------------------------------------------- September 2026 builds
def load_sep(VD,PB,OLDPANEL):
    """Panel of a September 2026 build with model labels, funds' labels (pre-Sep-2025 vote, carried along the new key as in
    sep2026/c1x.py) and the d2_common columns needed by es(). No date cap: this is the confirmatory quarter."""
    p=pd.read_parquet(f'{VD}/panel.parquet')
    p['fq']=p.cik.astype(str)+'_'+p.ddate.dt.strftime('%Y%m'); p['mq']=p.family.astype(str)+'_'+p.ddate.dt.strftime('%Y%m')
    p['vint']=p.pkey.map(p.groupby('pkey').ddate.min()).dt.strftime('%Y%m'); p['vq']=p.vint+'|'+p.ddate.dt.strftime('%Y%m')
    L=pd.read_parquet(f'{CM.F}/labels.parquet')[['bid','group']]
    pb=pd.read_parquet(PB).merge(L,on='bid').merge(p[['pos_id','pkey']],on='pos_id').drop_duplicates('pkey').set_index('pkey')
    p['grp']=p.pkey.map(pb.group); p['bid']=p.pkey.map(pb.bid)
    p['clu']=np.where(p.bid.notna(),'b'+p.bid.astype(str),'l'+p.loan.astype(str)); p['b99']=100*(p.mark<99)
    fl=pd.read_parquet(f'{CM.V}/fund_label_pre_2026-11-16.parquet').set_index('pkey')
    old=pd.read_parquet(OLDPANEL,columns=['pos_id','pkey']).set_index('pos_id').pkey
    o=p.pos_id.map(old); t=pd.DataFrame({'pkey':p.pkey,'d':p.ddate,'s':o.map(fl.sector_pre),'o':o}).dropna(subset=['o']).sort_values('d').drop_duplicates('pkey',keep='last').set_index('pkey')
    p['fsec']=p.pkey.map(t.s)
    s9=p[(p.ddate==CM.BASE)&p.ok].set_index('pkey'); p['prin9']=p.pkey.map(s9.principal); p['mark9']=p.pkey.map(s9.mark)
    return p
