import re, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
STOP=r'''\b(investments?|non[- ]?controlled|non[- ]?affiliated|affiliated|controlled|portfolio|company|companies|issuer|first|second|1st|2nd|lien|senior|secured|unsecured|subordinated|debt|term|loans?|delayed|draw|ddtl|revolver|revolving|credit|facility|commitment|type|expiration|date|maturity|interest|rate|reference|spread|sofr|libor|euribor|sonia|sora|bbsy|cdor|corra|prime|base|floor|cash|pik|paid|kind|industry|one|stop|unitranche|incremental|tranche|initial|acquisition|equity|common|stock|preferred|units?|class|warrants?|notes?|bonds?|united|states|canada|total|and|the|of|dba|fka|aka|due|effective|yield|non|accrual|percent|net|assets|instruments?|investment|other|structured|finance|obligation|sr|tl|rcf|dd)\b'''
def clean_text(s):
    s=s.lower()
    s=re.sub(r'\d+(\.\d+)?\s*%','',s)
    s=re.sub(r'\d{1,4}[/-]\d{1,2}([/-]\d{2,4})?','',s)
    s=re.sub(r'[^a-z&\s]',' ',s)
    s=re.sub(STOP,' ',s)
    s=re.sub(r'\s+',' ',s).strip()
    return s
# v2 stable key: normalise a description across template changes (affiliation labels, punctuation, rates and dates
# removed; tranche numbers kept)
AFF=r'\b(?:non[- ]?controlled|non[- ]?affiliated|affiliated|controlled|issuers?|investments?|portfolio compan(?:y|ies))\b'
def norm_desc(s):
    s=s.lower()
    s=re.sub(r'\d+(?:\.\d+)?\s*%','',s)
    s=re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{1,2}[/-]\d{4}\b','',s)
    s=re.sub(AFF,' ',s)
    s=re.sub(r'[^a-z0-9&]+',' ',s)
    return re.sub(r'\s+',' ',s).strip()
class UF:
    def __init__(s,n): s.p=list(range(n))
    def f(s,x):
        while s.p[x]!=x: s.p[x]=s.p[s.p[x]]; x=s.p[x]
        return x
    def u(s,a,b):
        a,b=s.f(a),s.f(b)
        if a!=b: s.p[b]=a
def match_period(g, thr=0.5, maxdays=62, return_pairs=False):
    g=g.reset_index(drop=True)
    docs=g.ident.map(clean_text)
    vec=TfidfVectorizer(token_pattern=r'(?u)\b[a-z&]{2,}\b',sublinear_tf=True,min_df=1)
    X=vec.fit_transform(docs)
    idf=dict(zip(vec.get_feature_names_out(),vec.idf_))
    key=np.where(g.spread.notna(), 's'+(g.spread*10000).round().astype('Int64').astype(str), 'r'+(g.rate*10000).round().astype('Int64').astype(str))
    g['blk']=key
    uf=UF(len(g)); pairs=[]
    mat=g.maturity.values; cik=g.cik.values
    for b,idx in g.groupby('blk').indices.items():
        if len(idx)<2 or len(set(cik[idx]))<2: continue
        Xb=X[idx]; S=(Xb@Xb.T).tocoo()
        for i,j,v in zip(S.row,S.col,S.data):
            if i>=j or v<thr: continue
            a,c=idx[i],idx[j]
            if cik[a]==cik[c]: continue
            ma,mc=mat[a],mat[c]
            if not (pd.isna(ma) or pd.isna(mc)) and abs((ma-mc)/np.timedelta64(1,'D'))>maxdays: continue
            uf.u(a,c); 
            if return_pairs: pairs.append((a,c,v))
    g['cluster']=[uf.f(i) for i in range(len(g))]
    return (g,pairs,docs) if return_pairs else g
