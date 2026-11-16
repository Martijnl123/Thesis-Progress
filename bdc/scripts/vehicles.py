# Vehicle type per BDC (cik): listed (with listing date), perpetual non-traded (evergreen, wealth channel), or private
import pandas as pd, re
SP='/tmp/claude-0/-home-claude/dc0028bd-8cc6-5691-8e20-9ef3d2183413/scratchpad'
g=pd.read_csv(SP+'/bdc2/sub_by_cik.csv')
d=pd.read_parquet(SP+'/bdc/positions.parquet',columns=['cik','bdc','family','ptype'])
v=d.drop_duplicates('cik').merge(g[['cik','maxfloat','afs']],on='cik',how='left')
# listed during the whole window unless a listing date is given; Guggenheim credit income funds report a float but are not listed
LISTED_FROM={1782524:'2024-01-24',  # Morgan Stanley Direct Lending Fund
             1737924:'2024-01-25',  # Nuveen Churchill Direct Lending
             1794776:'2024-01-17',  # Palmer Square Capital BDC
             1747172:'2024-05-22',  # Kayne Anderson BDC
             1807427:'2024-01-25',  # Blue Owl Capital Corp III
             1747777:'2025-06-12',  # Blue Owl Technology Finance Corp
             1535778:'2025-01-29'}  # MSC Income Fund
NOT_LISTED=v.bdc.str.contains('GUGGENHEIM CREDIT INCOME')
v['listed_from']=pd.NaT
v.loc[(v.maxfloat>0)&~NOT_LISTED,'listed_from']=pd.Timestamp('1900-01-01')
for c,dt in LISTED_FROM.items(): v.loc[v.cik==c,'listed_from']=pd.Timestamp(dt)
PERP=re.compile(r'PRIVATE CREDIT FUND|PRIVATE INCOME FUND|STRATEGIC INCOME|CREDIT INCOME|INCOME TRUST|CORPORATE LENDING FUND|CAPITAL SOLUTIONS FUND|DEBT SOLUTIONS|TECHNOLOGY INCOME|PRIVATE CREDIT CORP|INCOME PLUS|CAPITAL INCOME|STRATEGIC CREDIT FUND$|EVERGREEN|PRIVATE LENDING FUND$|ALTERNATIVE INCOME|^KENNEDY LEWIS CAPITAL CO|^BAIN CAPITAL PRIVATE CREDIT$|^MSC INCOME FUND|^GUGGENHEIM CREDIT INCOME|^ARES CORE INFRASTRUCTURE|PRIVATE CREDIT$')
NOT_PERP=re.compile(r'GUARDIAN IV INCOME FUND|^SCP PRIVATE CREDIT INCOME|^MONROE CAPITAL ENHANCED CORPORATE LENDING|^FIDELITY PRIVATE CREDIT CO LLC|^LGAM PRIVATE CREDIT|INSTITUTIONAL PRIVATE CREDIT')
v=v.merge(g[['cik','name']],on='cik',how='left')
clean=lambda x: x.str.upper().str.replace(r',? (INC|LLC|L\.L\.C|CORP)\.?$','',regex=True).str.strip()
nm2=clean(v.name.fillna(v.bdc))
nm=v.bdc.str.upper().str.replace(r',? (INC|LLC|L\.L\.C|CORP)\.?$','',regex=True).str.strip()
isp=lambda s: bool(PERP.search(s)) and not NOT_PERP.search(s)
v['perpetual']=nm.map(isp)|nm2.map(isp)
v.to_parquet('vehicles.parquet',index=False)
print(v.assign(listed=v.listed_from.notna()).groupby(['listed','perpetual']).size())
print(v[v.perpetual].bdc.tolist())
