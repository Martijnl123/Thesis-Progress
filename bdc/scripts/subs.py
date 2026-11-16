import pandas as pd, glob, os
SP='/tmp/claude-0/-home-claude/dc0028bd-8cc6-5691-8e20-9ef3d2183413/scratchpad'
fs=sorted(glob.glob(SP+'/bdc_data/*/sub.tsv'))
s=pd.concat([pd.read_csv(f,sep='\t',dtype=str,quoting=3,encoding_errors='replace') for f in fs],ignore_index=True).drop_duplicates('adsh')
s['pubfloat']=pd.to_numeric(s.pubfloatusd,errors='coerce')
s['cik']=s.cik.astype(int)
g=s.groupby('cik').agg(name=('name','last'),n=('adsh','size'),afs=('afs',lambda x: ','.join(sorted(set(x.dropna())))),
    maxfloat=('pubfloat','max'),forms=('form',lambda x: ','.join(sorted(set(x)))),first=('period','min'),last=('period','max'))
g.to_csv('sub_by_cik.csv')
print(len(g)); print(g.afs.value_counts().head(12))
pd.set_option('display.width',200); pd.set_option('display.max_rows',300)
print(g[g.maxfloat>0][['name','afs','maxfloat']].sort_values('maxfloat',ascending=False).to_string())
