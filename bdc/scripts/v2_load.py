# Read every soi.tsv into one table (v2: as load.py, plus the qtrs and cstm fields the SEC added in Sep 2026).
# Hold-out lock (pre-analysis plan v4, 'Lock', added 7 Nov 2026): the loader stops, before writing anything, if a row is
# dated 30 September 2026 or comes from a filing whose period of report is after 30 June 2026, unless the plan is posted
# (OSF-POSTED.txt) and --allow-sep2026 is given. Other dates after June 2026 in filings for earlier periods are typos or
# subsequent events (1 row dated 31 Mar 2028 in 2025q3, 11 rows dated 31 Dec 2026 in 2026_05, 21 rows in 2026_08, all
# in filings made before the September quarter ended) and are kept as before, so the output for the existing files
# does not change.
# Usage: python v2_load.py <src folder with */soi.tsv> <out.parquet> [--allow-sep2026]
import sys, glob, os, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'edgar'))
from holdout_lock import max_period, is_quarter_end, OPEN_MAX
ALLOW='--allow-sep2026' in sys.argv
args=[x for x in sys.argv[1:] if x!='--allow-sep2026']
src=args[0]; out=args[1]
MAXP=max_period(ALLOW)
KEEP={'adsh':'adsh','cik':'cik','name':'bdc','ddate':'ddate','form':'form','filed':'filed','period':'period',
 'Industry Sector Axis':'industry','Investment, Identifier Axis':'ident','Investment, Issuer Affiliation Axis':'affil',
 'Investment Type Axis':'invtype_axis','Investment Interest Rate':'rate','Investment, Basis Spread, Variable Rate':'spread',
 'Investment Maturity Date':'maturity','Investment Owned, Balance, Principal Amount':'principal',
 'Adjusted cost basis':'cost','Initial fair value of Investment':'fv','Investment Owned, Cost':'cost2','Investment Owned, Fair Value':'fv2',
 'Investment, Interest Rate, Paid in Kind':'pik_rate','Investment, Interest Rate, Paid in Cash':'cash_rate',
 'Investment, Variable Interest Rate, Type [Extensible Enumeration]':'refrate','Investment, Interest Rate, Floor':'floor',
 'Investment, Non-income Producing [true false]':'nonincome','Investment, Type [Extensible Enumeration]':'invtype_enum',
 'Investment, Issuer Name Axis':'issuer_axis','Investment, Issuer Name [Extensible Enumeration]':'issuer_enum',
 'Lien Category Axis':'lien_axis','Investment, Acquisition Date':'acqdate','Investment, Industry Sector [Extensible Enumeration]':'industry_enum',
 'Investment shares':'shares','Investment Owned, Net Assets, Percentage':'pct_na'}
KEEP.update({'qtrs':'qtrs','cstm':'cstm'})
frames=[]
for f in sorted(glob.glob(src+'/*/soi.tsv')):
    if os.path.getsize(f)==0: continue
    d=pd.read_csv(f,sep='\t',dtype=str,quoting=3,encoding_errors='replace',usecols=lambda c: c in KEEP)
    d=d.rename(columns=KEEP); d['srcfile']=f.split('/')[-2]
    frames.append(d)
d=pd.concat(frames,ignore_index=True)
late=(d.period.fillna('')>MAXP) | (d.ddate.fillna('').between('2026-07-01',OPEN_MAX) & d.ddate.fillna('').map(is_quarter_end) & (OPEN_MAX>MAXP))
if late.any():
    sys.exit(f'LOCK: {int(late.sum())} rows for a period after {MAXP} in {sorted(set(d.srcfile[late]))}. '
             'Nothing was written. Remove those files from the src folder, or post the plan and use --allow-sep2026.')
for a,b in [('cost','cost2'),('fv','fv2')]:
    if b in d: d[a]=d[a].fillna(d[b]); d=d.drop(columns=b)
for c in ['rate','spread','principal','cost','fv','pik_rate','cash_rate','floor','shares','pct_na','qtrs']:
    if c in d: d[c]=pd.to_numeric(d[c],errors='coerce')
d['cik']=pd.to_numeric(d.cik,errors='coerce').astype('Int64')
print('raw rows', len(d), 'with ident', d.ident.notna().sum(), flush=True)
d.to_parquet(out,index=False)
print(d.srcfile.value_counts().sort_index().to_dict())
