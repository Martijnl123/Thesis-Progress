# #93: registered common-shock test, part (b) (PAP v5 line 68): software's sensitivity to market-wide credit stress from the
# relative N-PORT price change of traded software loans and other traded loans in Q1 2020 (31 Dec 2019 -> 31 Mar 2020) and
# Q2 2022 (31 Mar 2022 -> 30 Jun 2022). N-PORT loans are labelled by the borrower's model label where the borrower is in the
# label set (matched N-PORT names from nport_matches_coupon.parquet), excluded otherwise. Comparison group = model-labelled
# other loans without ITSV/DATA/HARD, as in the registered Model index.
# Loan = cleaned issuer name + maturity date; price = 100 * value / balance, Level 2, balance > 0, price in (0, 150];
# median across funds per loan and quarter-end; a loan enters an episode if priced at both ends. Outcomes: mean price change
# (the mark analogue) and the change in the share priced below 99. Stress slope = software change / other change; SE from a
# 1,000-draw bootstrap over borrower names (seed 20261108). Prediction for Mar and Jun 2026 = slope x other loans' change,
# (i) in N-PORT prices (like for like) and (ii) in BDC marks (other-loan change from the (a) test, d3_extra.py). Registered
# rule: high beta rejected for a quarter if the actual software change is beyond the prediction by more than two SE.
# Usage: python3 r7_08_common_shock_nport.py <folder with loans_2020q1/q2, loans_2022q2/q3 .csv.gz>
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..','..')))  # repo root
from r7_common import *
sys.path.insert(0,(_ROOT+'/bdc/scripts')); from matchlib import clean_text
SD=sys.argv[1]; H=f'{OUT}/common-shock'; os.makedirs(H,exist_ok=True); L=Log(f'{H}/log_08_common_shock_nport.txt')
RAW=(_ROOT+'/raw-sec/nport/')
COLS=['accession_number','series_id','filing_date','report_date','issuer_name','balance','currency_value','fair_value_level','maturity_date']
def prices(files,dates):
    out=[]
    for f in files:
        for ch in pd.read_csv(f,usecols=COLS,dtype=str,chunksize=500000):
            ch=ch[ch.report_date.isin(dates)&(ch.fair_value_level=='2')]
            if len(ch): out.append(ch)
    d=pd.concat(out)
    # amendments: keep each series' latest filing per report date (files are stacked, NPORT-P/A repeat a quarter-end)
    d['series_id']=d.series_id.fillna(d.accession_number)
    d['fd']=pd.to_datetime(d.filing_date,format='%d-%b-%Y',errors='coerce')
    last=d.sort_values('fd').groupby(['series_id','report_date']).accession_number.last()
    d=d[d.set_index(['series_id','report_date']).accession_number.values==last.reindex(pd.MultiIndex.from_frame(d[['series_id','report_date']])).values]
    d['bal']=pd.to_numeric(d.balance,errors='coerce'); d['val']=pd.to_numeric(d.currency_value,errors='coerce')
    d=d[d.bal>0]; d['px']=100*d.val/d.bal; d=d[(d.px>0)&(d.px<=150)]
    d['nm']=d.issuer_name.fillna('').map(clean_text); d=d[d.nm.str.len()>0]
    return d.groupby(['nm','maturity_date','report_date']).px.median().reset_index()
# label set: N-PORT names matched to model-labelled BDC borrowers (majority group per name)
p=load(); M=pd.read_parquet(f'{CM.V}/nport_matches_coupon.parquet'); M=M[M.verified&(M.ddate<=CM.CAP)]
lab=p[['pos_id','grp']].dropna(); lab=lab[~lab.grp.isin(['UNKN','FUND'])]
nl=M[['pos_id','nm_b']].merge(lab,on='pos_id').groupby('nm_b').grp.agg(lambda s:s.value_counts().index[0])
nl=nl[~nl.isin(['ITSV','DATA','HARD'])]; SW=set(nl[nl=='SOFT'].index); OT=set(nl[nl!='SOFT'].index)
L(f'label set: {len(nl)} N-PORT borrower names ({len(SW)} software, {len(OT)} other) from verified matches to model-labelled BDC borrowers')
def episode(px,d0,d1):
    a=px[px.report_date==d0].set_index(['nm','maturity_date']).px; b=px[px.report_date==d1].set_index(['nm','maturity_date']).px
    j=pd.concat([a.rename('p0'),b.rename('p1')],axis=1).dropna().reset_index()
    j['sw']=j.nm.isin(SW); j=j[j.sw|j.nm.isin(OT)].copy(); j['d']=j.p1-j.p0; j['db99']=100*((j.p1<99).astype(float)-(j.p0<99).astype(float))
    return j
def stats(j):
    s=j[j.sw]; o=j[~j.sw]
    return {y:(s[y].mean(),o[y].mean()) for y in ['d','db99']}
def boot(j,B=1000):
    rng=np.random.default_rng(20261108); names=j.nm.unique(); idx=j.groupby('nm').indices; R={'d':[],'db99':[]}
    for _ in range(B):
        x=j.iloc[np.concatenate([idx[names[k]] for k in rng.choice(len(names),len(names))])]; st=stats(x)
        for y in R: R[y].append((st[y][0]/st[y][1]) if st[y][1]!=0 else np.nan)
    return {y:np.nanstd(v,ddof=1) for y,v in R.items()}
import glob
F=lambda q: (glob.glob(f'{SD}/loans_{q}_qe.csv.gz')+glob.glob(f'{SD}/loans_{q}.csv.gz'))[0]
EP={} if os.environ.get('CSTEST') else {'2020Q1':(F('2020q1'),F('2020q2'),'31-DEC-2019','31-MAR-2020'),
    '2022Q2':(F('2022q2'),F('2022q3'),'31-MAR-2022','30-JUN-2022')}
if os.environ.get('CSTEST'):  # dry run of the code on Jun->Sep 2025 (not a result)
    EP={'2025Q3':(RAW+'loans_2025q3.csv.gz',RAW+'loans_2025q4.csv.gz','30-JUN-2025','30-SEP-2025')}; H=H+'/dryrun'; os.makedirs(H,exist_ok=True); L=Log(f'{H}/log_dryrun.txt')
NOW={'Mar':('31-DEC-2025','31-MAR-2026'),'Jun':('31-MAR-2026','30-JUN-2026')}
px26=prices([RAW+f'loans_{q}.csv.gz' for q in ['2025q4','2026q1','2026q2','2026q3']],['31-DEC-2025','31-MAR-2026','30-JUN-2026'])
now={Q:stats(episode(px26,*NOW[Q])) for Q in NOW}; nnow={Q:episode(px26,*NOW[Q]).sw.value_counts().to_dict() for Q in NOW}
# BDC other-loan changes from the (a) test (frozen cohort, within-position changes)
bdc={}
for ln in open((_ROOT+'/outputs/draft7-2026-11-26/numbers.tex')):
    for k in ['CsOtMarkMar','CsOtMarkJun','CsOtBelowMar','CsOtBelowJun','CsActMarkMar','CsActMarkJun','CsActBelowMar','CsActBelowJun']:
        if ln.startswith('\\newcommand{\\'+k+'}'): bdc[k]=float(ln.split('}{')[1].split('}')[0].replace('$','').replace('{,}',''))
L(f'BDC (a)-test changes read from numbers.tex: {bdc}')
TAG={'2020Q1':'Covid','2022Q2':'Hike','2025Q3':'Dry'}
for e,(f0,f1,d0,d1) in EP.items():
    px=prices([f0,f1],[d0,d1]); j=episode(px,d0,d1); st=stats(j); se=boot(j)
    L(f'\n=== {e}: {d0} -> {d1}: loans priced at both ends {len(j)} ({int(j.sw.sum())} software, {int((~j.sw).sum())} other); borrowers {j.nm.nunique()}')
    for y,Y,dg in [('d','Mark',2),('db99','Below',1)]:
        sw,ot=st[y]; sl=sw/ot
        L(f'  {y}: software {sw:+.{dg}f}, other {ot:+.{dg}f}, relative {sw-ot:+.{dg}f}; stress slope {sl:.2f} (bootstrap SE {se[y]:.2f})')
        put(f'Cs{TAG[e]}{Y}Slope',sl,'f2',f'N-PORT stress slope {e}, {y}'); put(f'Cs{TAG[e]}{Y}SlopeSE',se[y],'f2',f'bootstrap SE, {e}, {y}')
        put(f'Cs{TAG[e]}{Y}Sw',sw,f'f{dg}',f'software change {e}, {y}',sign=True); put(f'Cs{TAG[e]}{Y}Ot',ot,f'f{dg}',f'other change {e}, {y}',sign=True)
        for Q in NOW:
            for src,(swa,ota) in [('Np',now[Q][y]),('Bdc',(bdc[f'CsAct{Y}{Q}'],bdc[f'CsOt{Y}{Q}']))]:
                pred=sl*ota; sep=se[y]*abs(ota); rej=(swa<pred-2*sep) if y=='d' else (swa>pred+2*sep)
                L(f'    {Q} 2026 {src}: other {ota:+.{dg}f}, predicted software {pred:+.{dg}f} (SE {sep:.{dg}f}), actual {swa:+.{dg}f} -> high beta {"REJECTED" if rej else "not rejected"}')
                put(f'Cs{TAG[e]}{Y}{Q}{src}Pred',pred,f'f{dg}',f'predicted software change {Q} 2026 at {e} slope, {src}',sign=True)
                put(f'Cs{TAG[e]}{Y}{Q}{src}PredSE',sep,f'f{dg}','SE of prediction')
                put(f'Cs{TAG[e]}{Y}{Q}{src}Rej','rejected' if rej else 'not rejected','str','registered rule')
for Q in NOW: L(f'N-PORT 2026 {Q}: software/other loans {nnow[Q]}; mean change sw/other {now[Q]["d"][0]:+.2f}/{now[Q]["d"][1]:+.2f}; below99 {now[Q]["db99"][0]:+.1f}/{now[Q]["db99"][1]:+.1f}')
write_macros(f'{H}/numbers-r7-08-common-shock.tex',list(MAC),'Draft 7 item #93: common-shock test part (b) on N-PORT 2020Q1 and 2022Q2 (r7_08_common_shock_nport.py)')
L.save()
