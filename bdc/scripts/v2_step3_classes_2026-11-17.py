# Step 3 classes for draft 2 (17 Nov 2026). Data through 30 June 2026 only; no September 2026 file is read.
# Fixes two problems review found in the 13 Nov classes (v2_traded_untraded_2026-11-13.py):
#  #68  the N-PORT search skipped positions whose fund labels the borrower IT services / other tech or gives no label
#       (926 software positions never searched). Here the search is nport_matches_all_2026-11-17.parquet (all clean positions).
#  #25  'traded' was set at LOAN level, so a revolver in the same cross-fund loan group as a priced term loan counted as
#       traded. Here 'traded' is set at TRANCHE level: a position is traded if it has its own verified Level 2 match, or if a
#       position of the same cross-fund loan with the same tranche type (lien x revolver/delayed-draw flag) has one.
# Classes at 30 Sep 2025 (cohort positions, model labels):
#   traded           tranche has a verified Level 2 N-PORT price
#   L3 only          tranche matched to an N-PORT holding valued at Level 3 only (a registered fund holds it, no quote)
#   traded borrower  tranche not priced, but the borrower has a Level 2 priced loan (verified match, or a name match to a
#                    Level 2 issuer at cosine >= 0.6 that was not ruled out by the 13 Nov hand check)
#   unclear          hand check of 13 Nov could not tell
#   private          none of the above: the borrower has no debt with a market price
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..')))  # repo root
import sys, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.insert(0,(_ROOT+'/bdc/scripts'))
from matchlib import clean_text
V=(_ROOT+'/bdc/v2'); F=(_ROOT+'/bdc/industry_llm/full')
base=pd.Timestamp('2025-09-30'); out=[]
def say(s): out.append(str(s)); print(s,flush=True)
p=pd.read_parquet(f'{V}/panel.parquet'); assert p.ddate.max()<=pd.Timestamp('2026-06-30')
P=pd.read_parquet(f'{V}/positions.parquet',columns=['pos_id','ident','first_lien','second_lien','maturity','nonaccrual'])
p=p.merge(P,on='pos_id',how='left')
L=pd.read_parquet(f'{F}/labels.parquet')[['bid','group']]
pb=pd.read_parquet(f'{F}/position_borrower.parquet').merge(L,on='bid').merge(p[['pos_id','pkey']],on='pos_id').drop_duplicates('pkey').set_index('pkey')
p['grp']=p.pkey.map(pb.group); p['bid']=p.pkey.map(pb.bid)
lab=p.grp.notna()&~p.grp.isin(['UNKN','FUND'])
s=p[(p.ddate==base)&p.ok&lab].copy(); s['sw']=s.grp.eq('SOFT').astype(int)
say(f'Cohort positions at Sep 2025: {len(s)} (software {s.sw.sum()})')
idl=s.ident.fillna('').str.lower()
s['rev']=idl.str.contains(r'revolv|\brcf\b|delayed|\bddtl\b',regex=True).astype(int)
s['lien']=np.where(s.second_lien==1,'2L',np.where(s.first_lien==1,'1L','other'))
s['tr']=s.lien+'|'+s.rev.astype(str)
M=pd.read_parquet(f'{V}/nport_matches_all_2026-11-17.parquet'); assert M.ddate.max()<=pd.Timestamp('2026-06-30')
M9=M[M.verified&(M.ddate==base)]
s['own2']=s.pos_id.isin(set(M9.loc[M9.lvl=='2','pos_id'])); s['own3']=s.pos_id.isin(set(M9.loc[M9.lvl=='3','pos_id']))
# tranche-level spill-over within a cross-fund loan, using ALL clean Sep 2025 positions (not only the cohort) as donors
a=p[(p.ddate==base)&p.ok].copy(); ia=a.ident.fillna('').str.lower()
a['rev']=ia.str.contains(r'revolv|\brcf\b|delayed|\bddtl\b',regex=True).astype(int)
a['lien']=np.where(a.second_lien==1,'2L',np.where(a.first_lien==1,'1L','other')); a['tr']=a.lien+'|'+a.rev.astype(str)
a['own2']=a.pos_id.isin(set(M9.loc[M9.lvl=='2','pos_id'])); a['own3']=a.pos_id.isin(set(M9.loc[M9.lvl=='3','pos_id']))
t2=set(zip(a.loc[a.own2,'loan'],a.loc[a.own2,'tr'])); t3=set(zip(a.loc[a.own3,'loan'],a.loc[a.own3,'tr']))
key=list(zip(s.loan,s.tr))
s['traded']=s.own2|pd.Series([k in t2 for k in key],index=s.index)
s['l3']=~s.traded&(s.own3|pd.Series([k in t3 for k in key],index=s.index))
# old (13 Nov) loan-level definition, for the comparison line
lt=a.groupby('loan').own2.any(); s['traded_loanlevel']=s.loan.map(lt).fillna(False)
# borrower has priced debt: (i) any position of the borrower (any fund, any tranche) with a verified L2 match
a['bid']=a.pkey.map(pb.bid)
tb_auto=set(a.loc[a.own2,'bid'].dropna())
# (ii) name match to a Level 2 issuer at Sep 2025, cosine >= 0.6, no maturity/coupon rule (borrower-level, not tranche-level)
n=pd.read_parquet(f'{V}/nport_qe_loans.parquet'); n=n[(n.report_date==base)&(n.lvl=='2')]
ua=s.ident.fillna('').map(clean_text); ub=n.issuer_name.fillna('').map(clean_text).drop_duplicates()
uu=pd.Series(ua.unique()); uu=uu[uu.str.len()>0].reset_index(drop=True); ub=ub[ub.str.len()>0].reset_index(drop=True)
vec=TfidfVectorizer(token_pattern=r'(?u)\b[a-z&]{2,}\b',sublinear_tf=True,max_df=0.02).fit(pd.concat([uu,ub]))
S=(vec.transform(uu)@vec.transform(ub).T).tocsr(); hit=set(uu[np.unique(S.multiply(S>=0.6).nonzero()[0])])
s['namehit']=ua.isin(hit)
tb_name=set(s.loc[s.namehit,'bid'].dropna())
HV=pd.read_csv(f'{V}/handcheck/traded_untraded_borrower_verdicts_2026-11-13.csv')
hv=HV.drop_duplicates('bid').set_index('bid').verdict
tb=tb_auto|{b for b in tb_name if hv.get(b)!='wrong'}|set(hv[hv=='traded_borrower'].index)
unc=set(hv[hv=='unclear'].index)
s['cls']=np.select([s.traded,s.l3,s.bid.isin(tb),s.bid.isin(unc)],['traded','L3 only','traded borrower','unclear'],'private')
say('\nA. Classes at Sep 2025, tranche level, all positions searched (positions)')
say(pd.crosstab(s.cls,s.sw.map({1:'software',0:'other'}),margins=True).to_string())
say('\nBorrowers by class, software: '+'; '.join(f'{k} {v}' for k,v in s[s.sw==1].groupby('cls').bid.nunique().items()))
say(f'Tranche vs loan level: positions traded at loan level but not at tranche level: {int((s.traded_loanlevel&~s.traded).sum())} '
    f'(software {int((s.traded_loanlevel&~s.traded&(s.sw==1)).sum())}); traded at tranche level but not loan level: {int((s.traded&~s.traded_loanlevel).sum())}')
old=pd.read_parquet(f'{V}/traded_untraded_sep25_2026-11-13.parquet').set_index('pkey').cls.rename({'untraded':'private'})
old=old.replace({'untraded':'private'})
say('\nOld (13 Nov) class x new class, software positions:')
x=s[s.sw==1].set_index('pkey'); say(pd.crosstab(old.reindex(x.index).fillna('NA'),x.cls,margins=True).to_string())
say(f'Name-hit borrowers (any sector) not hand-checked, now "traded borrower": {len(tb_name-set(hv.index)-tb_auto)}')
keep=['pos_id','pkey','bid','cik','family','loan','vtype','sw','grp','sector','mark','principal','spread_pre','pik_pre','rev','lien','tr','own2','own3','traded','l3','namehit','cls','maturity','first_lien','second_lien']
s[keep].to_parquet(f'{V}/step3_classes_2026-11-17.parquet',index=False)
open(f'{V}/results_step3_classes_2026-11-17.txt','w').write('\n'.join(out)+'\n')
