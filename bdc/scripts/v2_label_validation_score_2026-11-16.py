# Scores the blind label-validation sample (PAP v5 section 6). Written 16 Nov 2026; run it when the coding sheets are filled.
#
#   python3 v2_label_validation_score_2026-11-16.py <coder1.xlsx or .csv> [<coder2.xlsx or .csv>]
#
# Reads the coders' sheets (columns code_id, group_code, is_software_0_1), the key (strata, design weights) and the
# model's labels. Reports, with design weights (each sampled borrower stands for N_stratum / n_stratum borrowers):
#   1. agreement of the model's software flag with coder 1: precision, recall, false omission rate, by stratum too;
#   2. Cohen's kappa between the two coders on the rows both coded (software flag and 26-group code);
#   3. a design-based supervised learning (DSL; Egami et al. 2023) correction of the Step 1 estimate. The outcome is each
#      cohort position's change from Sep 2025 to Mar 2026 (and Jun 2026) relative to its fund's mean, the regressor is the
#      software flag. DSL replaces each moment m(model label) by m(model) - (R/pi) * (m(model) - m(coder)), where R marks
#      the sampled borrowers and pi is their sampling probability. SE: borrower bootstrap within strata, 500 draws.
# Rows a coder marks UNKN (or leaves empty) are dropped and the stratum weights rescaled.
# Data through 30 June 2026 only. No language model is used here.
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..')))  # repo root
import sys, numpy as np, pandas as pd
V=(_ROOT+'/bdc/v2'); F=(_ROOT+'/bdc/industry_llm/full')
O=(_ROOT+'/bdc/industry_llm/validation')
def sheet(f):
    d=pd.read_excel(f,sheet_name='Coding') if f.endswith('.xlsx') else pd.read_csv(f)
    d['group_code']=d.group_code.astype(str).str.strip().str.upper().replace({'NAN':np.nan,'':np.nan})
    d['sw']=pd.to_numeric(d.is_software_0_1,errors='coerce')
    d.loc[d.sw.isna()&d.group_code.notna()&(d.group_code!='UNKN'),'sw']=(d.group_code=='SOFT').astype(float)
    d.loc[d.group_code=='UNKN','sw']=np.nan
    return d.set_index('code_id')[['group_code','sw']]
if len(sys.argv)<2: sys.exit(__doc__ or 'usage: v2_label_validation_score_2026-11-16.py coder1 [coder2]')
c1=sheet(sys.argv[1]); c2=sheet(sys.argv[2]) if len(sys.argv)>2 else None
K=pd.read_csv(f'{O}/key_DO_NOT_OPEN_BEFORE_CODING_2026-11-08.csv').set_index('code_id')
L=pd.read_parquet(f'{F}/labels.parquet').set_index('bid')
S=K.join(c1).join(L[['group']],on='bid'); S['msw']=(S.group=='SOFT').astype(float)
out=[]
def say(s): out.append(s); print(s,flush=True)
say(f'Label validation, scored {pd.Timestamp.today():%d %b %Y}. Coder 1 file: {sys.argv[1]}')
say(f'Coded by coder 1: {int(S.sw.notna().sum())} of {len(S)} (UNKN or empty: {int(S.sw.isna().sum())})')
if S.sw.notna().sum()<20: sys.exit('Fewer than 20 coded rows: nothing scored.')
C=S[S.sw.notna()].copy(); C['w']=C.N_stratum/C.groupby('stratum').sw.transform('size')
def wmean(x,w): return (x*w).sum()/w.sum() if w.sum()>0 else np.nan
def stats(C):
    ms,hs,w=C.msw==1,C.sw==1,C.w
    return dict(precision=wmean(hs[ms].astype(float),w[ms]),recall=wmean(ms[hs].astype(float),w[hs]),false_omission=wmean(hs[~ms].astype(float),w[~ms]),
                human_share=wmean(hs.astype(float),w),model_share=wmean(ms.astype(float),w))
st=stats(C); rng=np.random.default_rng(20261116); B=[]
for b in range(1000):
    Cb=pd.concat([g.iloc[rng.integers(0,len(g),len(g))] for _,g in C.groupby('stratum')]); B.append(stats(Cb))
B=pd.DataFrame(B)
say('\n1. Model software flag against coder 1 (design-weighted; 90% bootstrap interval)')
for k,v in st.items(): say(f'   {k}: {v:.3f} [{B[k].quantile(.05):.3f}, {B[k].quantile(.95):.3f}]')
say('   by stratum (unweighted): '+'; '.join(f'{s}: coded {len(g)}, model software {int(g.msw.sum())}, coder software {int(g.sw.sum())}, agree {(g.msw==g.sw).mean():.2f}' for s,g in C.groupby('stratum')))
g26=C[C.group_code.notna()]; say(f'   same 26-group code as the model: {(g26.group_code==g26.group).mean():.3f} (unweighted, n {len(g26)}); design-weighted {wmean((g26.group_code==g26.group).astype(float),g26.w):.3f}')
fell=C.stratum.str.endswith('fell')
for nm,m in [('borrowers whose loans fell 3+ points',fell),('other borrowers',~fell)]:
    x=C[m]; say(f'   {nm}: model says software where coder does not {wmean(((x.msw==1)&(x.sw==0)).astype(float),x.w):.3f}; coder says software where model does not {wmean(((x.msw==0)&(x.sw==1)).astype(float),x.w):.3f}')
say('   (Leakage would show as the first rate being higher among fallers than among other borrowers.)')
def kappa(a,b):
    cats=sorted(set(a)|set(b)); po=(a==b).mean(); pe=sum((a==c).mean()*(b==c).mean() for c in cats); return (po-pe)/(1-pe) if pe<1 else np.nan
if c2 is not None:
    J=c1.join(c2,lsuffix='1',rsuffix='2').dropna(subset=['sw1','sw2'])
    say(f'\n2. Two coders, rows both coded: {len(J)} | software flag agreement {(J.sw1==J.sw2).mean():.3f}, Cohen kappa {kappa(J.sw1.values,J.sw2.values):.3f}')
    J2=J.dropna(subset=['group_code1','group_code2']); say(f'   26-group code agreement {(J2.group_code1==J2.group_code2).mean():.3f}, Cohen kappa {kappa(J2.group_code1.values,J2.group_code2.values):.3f} (n {len(J2)})')
else: say('\n2. No second coder file given.')
# 3. DSL-corrected Step 1 estimate (cross-section of changes, within fund)
p=pd.read_parquet(f'{V}/panel.parquet'); assert p.ddate.max()<=pd.Timestamp('2026-06-30')
pb=pd.read_parquet(f'{F}/position_borrower.parquet').merge(p[['pos_id','pkey']],on='pos_id').drop_duplicates('pkey').set_index('pkey')
p['bid']=p.pkey.map(pb.bid); p['grp']=p.bid.map(L.group); base=pd.Timestamp('2025-09-30')
keys=set(p.loc[(p.ddate==base)&p.ok&p.grp.notna()&~p.grp.isin(['UNKN','FUND']),'pkey'])
z=p[p.pkey.isin(keys)&p.mark.between(0,110)]; z=z[(z.ddate==base)|(z.principal>=250e3)]
say('\n3. Step 1 estimate corrected for label error (DSL), within-fund change since Sep 2025; borrower bootstrap')
pi=(1/C.w).rename('pi'); hsw=C.set_index('bid').sw; pib=C.set_index('bid').w.rdiv(1)
for q in ['2026-03-31','2026-06-30']:
    w=z[z.ddate.isin([base,pd.Timestamp(q)])].pivot_table(index='pkey',columns='ddate',values='mark').dropna()
    d=pd.DataFrame({'dm':w.iloc[:,1]-w.iloc[:,0],'db':100*(w.iloc[:,1]<99).astype(float)-100*(w.iloc[:,0]<99)}).join(p.drop_duplicates('pkey').set_index('pkey')[['cik','bid','grp']])
    d['xm']=(d.grp=='SOFT').astype(float); d['R']=d.bid.isin(hsw.index).astype(float); d['xh']=d.bid.map(hsw); d['pi']=d.bid.map(pib)
    def est(d,y):
        d=d.reset_index(drop=True); yd=d[y]-d.groupby('cik')[y].transform('mean'); xm=d.xm-d.groupby('cik').xm.transform('mean')
        naive=(xm*yd).sum()/(xm*xm).sum()
        s=d[d.R==1]; xh=(s.xh-d.groupby('cik').xm.transform('mean').reindex(s.index)); k=1/s.pi
        A=(xm*xm).sum()-(k*(xm.reindex(s.index)**2-xh**2)).sum(); c=(xm*yd).sum()-(k*(xm.reindex(s.index)*yd.reindex(s.index)-xh*yd.reindex(s.index))).sum()
        return naive,c/A
    for y,nm in [('dm','mark'),('db','Below99')]:
        nv,ds=est(d,y); bids=d.bid.dropna().unique(); idx=d.groupby('bid').indices; bb=[]
        for b in range(500):
            dr=rng.choice(bids,len(bids),replace=True); bb.append(est(d.iloc[np.concatenate([idx[k] for k in dr])],y))
        bb=np.array(bb); say(f'   {q[:7]} {nm}: model labels {nv:+.2f} (SE {bb[:,0].std():.2f}); DSL-corrected {ds:+.2f} (SE {bb[:,1].std():.2f})')
open(f'{O}/results_label_validation_{pd.Timestamp.today():%Y-%m-%d}.txt','w').write('\n'.join(out)+'\n')
