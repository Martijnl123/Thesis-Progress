# 17 Nov 2026 (draft 2): same as v2_nport_match_coupon_2026-11-08.py but on ALL clean positions (no fund-sector filter), for Step 3 classes.
# The registered C2 sample keeps the 8 Nov match (PAP v5 'C2 sample'); this file is used for Step 3 and a Step 2 robustness row only.
# Registered pre-step (PAP v5 section 5 'C2 sample'), run 8 Nov 2026 on quarters to 30 Jun 2026: when a BDC position's coupon
# field is empty, read the all-in coupon from its description, then rerun the N-PORT match on all quarters.
# Parser: 'interest rate [of] X%' first; else, if the description has a spread (S/L/SOFR/Prime + x%), the largest percentage
# between 4 and 25 that is not the spread. Percentages above 25 (e.g. '% of net assets') are ignored.
# Same match rules as v2_nport_match.py (TF-IDF name similarity >= 0.6; reject maturity > 62 days or coupon > 0.5pp apart;
# verified if maturity or coupon agrees). Writes nport_matches_coupon.parquet; the old nport_matches.parquet is kept.
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..')))  # repo root
import re, sys, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.insert(0,(_ROOT+'/bdc/scripts'))
from matchlib import clean_text
V=(_ROOT+'/bdc/v2')
QE=pd.to_datetime(['2025-06-30','2025-09-30','2025-12-31','2026-03-31','2026-06-30'])
n=pd.read_parquet(f'{V}/nport_qe_loans.parquet'); assert n.report_date.max()<=pd.Timestamp('2026-06-30')
RX1=re.compile(r'interest rate\s*(?:of\s*)?[:=]?\s*([0-9]{1,2}(?:\.[0-9]+)?)\s*%',re.I)
RXS=re.compile(r'(?:\b(?:s|l|sofr|libor|prime|p|e|sonia|euribor)\s*\+\s*)([0-9]{1,2}(?:\.[0-9]+)?)\s*%',re.I)
RXP=re.compile(r'([0-9]{1,2}(?:\.[0-9]+)?)\s*%')
def coupon(s):
    if not isinstance(s,str): return np.nan
    m=RX1.search(s)
    if m: return float(m.group(1))/100
    sp=[float(x) for x in RXS.findall(s)]
    if not sp: return np.nan
    c=[float(x) for x in RXP.findall(s) if 4<=float(x)<=25 and float(x) not in sp]
    return max(c)/100 if c else np.nan
p=pd.read_parquet(f'{V}/panel.parquet',columns=['pos_id','cik','family','ddate','mark','ok','sector','loan','pkey'])
p=p.merge(pd.read_parquet(f'{V}/positions.parquet',columns=['pos_id','ident','maturity','rate']),on='pos_id')
p=p[p.ddate.isin(QE)&p.ok].copy()  # 17 Nov: fund-sector filter removed (review #68), all clean positions searched
miss=p.rate.isna(); p['rate_desc']=np.where(miss,p.ident.map(coupon),np.nan)
p['rate2']=p.rate.fillna(p.rate_desc)
# check the parser where both exist
chk=p[~miss].copy(); chk['parsed']=chk.ident.map(coupon); ok=chk.parsed.notna()
print(f'coupon field empty: {miss.mean():.1%} of positions; parsed from description for {p.rate_desc.notna().sum()} of {miss.sum()} ({p.rate_desc.notna().sum()/miss.sum():.1%})')
print(f'parser test where the field exists: parsed {ok.mean():.1%}; within 0.5pp of the field {((chk.parsed-chk.rate).abs()<=0.005)[ok].mean():.1%} (n {ok.sum()})')
p['nm']=p.ident.map(clean_text); n['nm']=n.issuer_name.fillna('').map(clean_text)
rows=[]
for d in QE:
    a=p[p.ddate==d]; b=n[n.report_date==d]
    ua=a.nm[a.nm.str.len()>0].drop_duplicates().reset_index(drop=True); ub=b.nm[b.nm.str.len()>0].drop_duplicates().reset_index(drop=True)
    vec=TfidfVectorizer(token_pattern=r'(?u)\b[a-z&]{2,}\b',sublinear_tf=True,max_df=0.02).fit(pd.concat([ua,ub]))
    XB=vec.transform(ub).T.tocsr(); cand=[]
    for i in range(0,len(ua),2000):
        S=(vec.transform(ua.iloc[i:i+2000])@XB).tocoo(); k=S.data>=0.6
        cand.append(pd.DataFrame({'nm':ua.values[i+S.row[k]],'nm_b':ub.values[S.col[k]],'cos':S.data[k]}))
    cand=pd.concat(cand,ignore_index=True)
    m=a[['pos_id','nm','maturity','rate2']].rename(columns={'rate2':'rate'}).merge(cand,on='nm').merge(
        b[['nm','maturity','rate','price_median','lvl','n_series']].rename(columns={'nm':'nm_b','maturity':'mat_n','rate':'rate_n'}),on='nm_b')
    gap=(m.maturity-m.mat_n).abs().dt.days; dr=(m.rate-m.rate_n).abs()
    m=m[~(gap>62) & ~(dr>0.005)].copy()
    m['verified']=(gap.reindex(m.index)<=62)|(dr.reindex(m.index)<=0.005)
    m['ddate']=d; rows.append(m)
M=pd.concat(rows,ignore_index=True); M.to_parquet(f'{V}/nport_matches_all_2026-11-17.parquet',index=False)
O=pd.read_parquet(f'{V}/nport_matches_coupon.parquet')
k=lambda X: set(zip(X.pos_id,X.nm_b,X.lvl))
o2=O[O.verified&(O.lvl=='2')]; n2=M[M.verified&(M.lvl=='2')]
print(f'verified Level 2 match rows: 8 Nov (filtered universe) {len(o2)}, new {len(n2)}; dropped {len(k(o2)-k(n2))}, added {len(k(n2)-k(o2))}')
print(f'positions with a verified Level 2 match: 8 Nov {o2.pos_id.nunique()}, new {n2.pos_id.nunique()}')
iv=p[p.ident.str.contains('ivanti',case=False,na=False)].pos_id
print('Ivanti verified L2 matches old:',o2[o2.pos_id.isin(iv)][['nm_b','price_median']].drop_duplicates().to_dict('records')[:6])
print('Ivanti verified L2 matches new:',n2[n2.pos_id.isin(iv)][['nm_b','price_median']].drop_duplicates().to_dict('records')[:6])
