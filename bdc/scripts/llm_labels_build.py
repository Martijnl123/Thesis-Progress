# Build the borrower list for the full LLM industry run (2026-10-26). Universe: clean v2 positions at Sep 2025 (the frozen
# panel's base). Borrower = positions linked as the same loan across funds (v2 loan id) or, within one fund, by near-identical
# descriptions (TF-IDF on that fund's descriptions, so header and loan-type words carry little weight). The model sees only
# names: descriptions from managers that print no industry are preferred, and industry words are cut from the rest.
import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..')))  # repo root
import sys, re, numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
V=(_ROOT+'/bdc/v2'); OUT=sys.argv[1]
S0=pd.Timestamp('2025-09-30'); COVER=0.95
IND=r'software|it services|information technology|internet|health ?care|pharma\w*|biotech\w*|life sciences|insurance|financial|capital markets|banks?\b|professional services|commercial (?:services|& professional)|business services|consumer|retail|media|entertainment|chemicals|construction|engineering|capital goods|building products|machinery|aerospace|transport\w*|logistics|automobiles?|auto components|automotive|food|beverage|household|hotels|leisure|education|real estate|energy|utilit\w*|telecom\w*|equipment|distributors|containers|packaging|electrical|electronic\w*|trading compan\w*|metals|mining|oil|environmental|textiles|apparel|semiconductors?|hardware|wholesale|paper|plastics|personal (?:care|products)|durables|marketing|advertising|publishing|broadcasting|industrial\w*|technology|high tech|supplies|services|diversified|products|providers?|manufacturing|healthcare|restaurants|gaming|chemical|materials|containers|defense|wireless|diversified|sovereign|forest|beverages|tobacco|broadcasting'
JUNK=r'\b(?:investments?|non[- ]?controlled|non[- ]?affiliat\w*|affiliat\w*|controlled|portfolio|debt|united states|first|second|1st|2nd|lien|senior|secured|unsecured|subordinated|term|loans?|delayed|draw|revolv\w*|unitranche|one stop|incremental|initial|tranche|interest|rate|reference|spread|sofr|euribor|sonia|libor|floor|maturity|dissolution|date|acquisition|type|investment type|issuer name|industry|current coupon|basis point|above index|cash|pik|due|index|s\+|bank loans|corporate|funded|unfunded|commitment|class|note|notes|asset type)\b'

# industry label phrases (GICS and Moody's names) cut from every description; single words are cut only for label-printing managers
PH=['application software','systems software','software (?:&|and) services','internet software (?:&|and) services','high tech(?:nology)? industries',
    'it consulting (?:&|and) other services','it consulting','information technology services','internet services (?:&|and) infrastructure',
    'data processing (?:&|and) outsourced services','health ?care (?:providers|equipment|supplies|services|facilities|distributors|technology)(?: (?:&|and) (?:services|supplies))?',
    'healthcare (?:&|and) pharmaceuticals','life sciences tools (?:&|and) services','insurance brokers','property (?:&|and) casualty insurance',
    'diversified support services','environmental (?:&|and) facilities services','commercial printing','office services (?:&|and) supplies',
    'security (?:&|and) alarm services','human resource (?:&|and) employment services','research (?:&|and) consulting services','professional services',
    'commercial (?:services|&|and) (?:&|and )?(?:supplies|professional services)','commercial (?:&|and) professional services','trading companies (?:&|and) distributors',
    'building products','construction (?:&|and) (?:engineering|building)','electrical equipment','industrial machinery','aerospace (?:&|and) defen[cs]e',
    'air freight (?:&|and) logistics','ground transportation','transportation: (?:cargo|consumer)','specialized consumer services','education services',
    'diversified consumer services','hotels?,? (?:restaurants|gaming) (?:&|and) leisure','leisure products','household durables','consumer goods: (?:non-)?durable',
    'textiles,? apparel (?:&|and) luxury goods','specialty retail','internet (?:&|and) direct marketing retail','food products','personal (?:care )?products',
    'household products','interactive media (?:&|and) services','media: [a-z ,&]+','diversified telecommunication services','wireless telecommunication services',
    'containers,? (?:&|and )?packaging(?: (?:&|and) glass)?','metals (?:&|and) mining','paper (?:&|and) forest products','forest products (?:&|and) paper',
    'oil,? gas (?:&|and) consumable fuels','energy equipment (?:&|and) services','energy: [a-z &]+','(?:electric|gas|water) utilities','utilities: [a-z &]+',
    'real estate management (?:&|and) development','capital markets','diversified financial services','consumer finance','financial services',
    'banking,? finance,? insurance (?:&|and) real estate','fire: [a-z ]+','services: (?:business|consumer)','beverage,? food (?:&|and) tobacco',
    'chemicals,? plastics (?:&|and) rubber','capital equipment','environmental industries','technology hardware[a-z ,&]*','electronic equipment[a-z ,&]*',
    'semiconductors? (?:&|and) semiconductor equipment','communications equipment','auto(?:motive)? components','consumer staples distribution (?:&|and) retail',
    'consumer discretionary','consumer staples','information technology','communication services','business services','consumer services','capital goods']
PHR=re.compile(r'\b(?:'+'|'.join(PH)+r')\b',re.I)

pn=pd.read_parquet(f'{V}/panel.parquet',columns=['pos_id','cik','family','ddate','ok','loan','sector','principal'])
pn=pn[(pn.ddate==S0)&pn.ok]
d=pn.merge(pd.read_parquet(f'{V}/positions.parquet',columns=['pos_id','ident']),on='pos_id')
d['text']=d.ident.str.replace(r'\s+',' ',regex=True).str.strip()
share=d.assign(h=d.text.str.contains(IND,case=False)).groupby('family').h.mean()
d['lab_fam']=d.family.isin(share[share>=0.6].index)   # managers that print an industry in most descriptions
print('positions',len(d),'| label-printing families',int((share>=0.6).sum()))

# links: same loan across funds (v2 loan id), and same borrower within a fund (descriptions with cosine >= 0.75)
d=d.reset_index(drop=True); n=len(d); rows=[]; cols=[]
lid=pd.factorize(d.loan)[0]; first=pd.Series(np.arange(n)).groupby(lid).transform('first').values
rows+=list(range(n)); cols+=list(first)
d['norm']=d.text.str.lower().str.replace(r'\([^)]*\)',' ',regex=True).str.replace(r'[^a-z& ]',' ',regex=True)
for cik,ix in d.groupby('cik').groups.items():
    ix=np.asarray(ix)
    if len(ix)<2: continue
    vec=TfidfVectorizer(token_pattern=r'(?u)\b[a-z&]{2,}\b',sublinear_tf=True,max_df=0.05 if len(ix)>200 else 0.3)
    try: X=vec.fit_transform(d.norm.values[ix])
    except ValueError: continue
    S=(X@X.T).tocoo(); k=(S.data>=0.75)&(S.row<S.col)
    rows+=list(ix[S.row[k]]); cols+=list(ix[S.col[k]])
G=coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(n,n))
_,comp=connected_components(G,directed=False); d['bid']=comp
sz=d.groupby('bid').loan.nunique()
print('borrowers',d.bid.nunique(),'| loans per borrower: median',sz.median(),'99th pct',sz.quantile(.99),'max',sz.max())

# names the model sees: trade names in brackets from any description, and up to two descriptions with numbers and junk cut
BAD=re.compile(r'\d|%|sofr|libor|euribor|sonia|floor|pik|cash|loan|lien|revolv|delayed|draw|term|tranche|warrant|equity|units|shares|note|bps|spread|non-|affiliat|controlled|incremental|fund|jv|senior|secured|investment|industry|^[a-z]{1,3}$',re.I)
def alts(s):
    out=[]
    for m in re.findall(r'\(([^()]{3,80})\)',s):
        m=re.sub(r'^(dba|d/b/a|aka|a/k/a|fka|f/k/a|formerly known as|formerly)\s*','',m.strip(),flags=re.I).strip(' .,')
        if m and not BAD.search(m) and not re.search(IND,m,re.I): out.append(m)
    return out
def strip(s,lab):
    s=re.sub(r'\([^)]*\)',' ',s)
    # cut dates, rates, spreads, amounts and tranche numbers, but keep digits that belong to names (Summit 7, C-4, 1-800)
    s=re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}|\b\d{1,2}/\d{4}\b|[+-]?\d+(?:\.\d+)?\s*%|\b\d+\.\d+\b|\b\d{1,3}(?:,\d{3})+\b',' ',s)
    s=re.sub(r'\b(?:S|SOFR|E|L|P|SONIA|EURIBOR)?\s*\+\s*[\d.]+\b|\b\d+(?:st|nd|rd|th)\b|\s\d{1,2}\s*$',' ',s,flags=re.I)
    s=PHR.sub(' ',s)
    s=re.sub(JUNK,' ',s,flags=re.I)
    s=re.sub(r'(?<![&\w])[Ss](?![&\w])|\b(?:instrument|ref|total coupon|coupon|original|last[- ]out|first[- ]out|super[- ]priority)\b',' ',s,flags=re.I)
    if lab: s=re.sub(r'\b(?:'+IND+r')\b',' ',s,flags=re.I)
    s=re.sub(r'\+|\b\d{1,2}/\d{0,2}/?\d{0,4}\b',' ',s)
    s=re.sub(r'\s*[-|:/]\s*(?=[-|:/]|$)',' ',s)
    return re.sub(r'\s+',' ',s).strip(' -|,:;/.')[:140]
d['clean']=[strip(t,l) for t,l in zip(d.text,d.lab_fam)]
B=[]
for bid,g in d.groupby('bid'):
    g=g.sort_values(['lab_fam','principal'],ascending=[True,False])
    names=list(dict.fromkeys(x for t in g.text for x in alts(t)))[:4]
    descs=list(dict.fromkeys(c for c in g.clean if len(c)>=3))[:2]
    B.append((bid,g.principal.sum(),g.family.nunique(),g.loan.nunique(),bool(g.lab_fam.all()),' || '.join(descs),'; '.join(names),
              g.sector.dropna().mode().iat[0] if g.sector.notna().any() else None,g.sector.notna().mean()))
B=pd.DataFrame(B,columns=['bid','principal','n_fam','n_loans','only_lab_fam','descs','trade_names','mgr_sector','mgr_share']).sort_values('principal',ascending=False)
B['cum']=B.principal.cumsum()/B.principal.sum(); B['in_run']=B.cum.shift(fill_value=0)<COVER
B['item']=np.where(B.in_run,np.arange(1,len(B)+1),0)
print('borrowers covering',COVER,'of principal:',int(B.in_run.sum()),'of',len(B),'| of these, input only from label-printing managers:',round(B[B.in_run].only_lab_fam.mean(),3))
B.to_parquet(f'{OUT}/borrowers.parquet',index=False)
d[['pos_id','bid']].to_parquet(f'{OUT}/position_borrower.parquet',index=False)
r=B[B.in_run]
r['text']=r.descs+np.where(r.trade_names.str.len()>0,' | other names: '+r.trade_names,'')
r[['item','text']].to_csv(f'{OUT}/run_input.csv',index=False)
print(r[['item','text']].sample(25,random_state=1).to_string(index=False))
