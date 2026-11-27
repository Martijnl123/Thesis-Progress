# Magenta Security: N-PORT Level 2 prices by tranche at 30 Jun and 30 Sep 2025 (draft 7, 26 Nov 2026).
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..','..')))  # repo root
import pandas as pd
D=(_ROOT+'/raw-sec/nport/')
out=[]
for f in ['loans_2025q3','loans_2025q4']:
    for ch in pd.read_csv(D+f+'.csv.gz',chunksize=400000,dtype=str):
        m=ch[ch.issuer_name.str.contains('magenta',case=False,na=False)]
        if len(m): out.append(m)
d=pd.concat(out)
d['bal']=pd.to_numeric(d.balance,errors='coerce'); d['val']=pd.to_numeric(d.currency_value,errors='coerce')
d=d[d.report_date.isin(['30-JUN-2025','30-SEP-2025'])&(d.bal>0)&(d.fair_value_level=='2')]
d['px']=100*d.val/d.bal
t=d.issuer_title.str.upper()
lx=d.other_identifiers.fillna('').str.extract(r'(LX24211[3-6]|BYD784\w+)')[0]
d['tranche']=t.str.extract(r'(FIRST OUT|SECOND OUT|THIRD OUT|SUPER PRIORITY)')[0]
d['maturity']=d.maturity_date
print(d.groupby(['report_date','tranche']).agg(n=('px','size'),median_px=('px','median'),min=('px','min'),max=('px','max'),maturities=('maturity',lambda s: ','.join(sorted(s.dropna().unique())))).round(2).to_string())
print('\nAll Magenta tranches share maturity:', sorted(d.maturity.dropna().unique()))
