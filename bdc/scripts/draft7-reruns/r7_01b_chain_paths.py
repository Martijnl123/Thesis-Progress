# Item 1 add-on: chain-linked index paths (no bootstrap) for the two registered indices of Appendix B, to compare with the
# cohort-only and balanced paths of r7_01_survivorship.py chain runs. Breakdowns of the registered indices are in draft 6 numbers.tex.
from r7_common import *
from d2_rank import changes, c_t
L=Log(f'{OUT}/log_01b_chain_paths.txt'); p=load(); ch=changes(p)
def theta(x,y):
    q2=sorted(x.ddate.unique()); c=pd.Series({q:c_t(x[x.ddate==q],y,x[x.ddate==q].sw) for q in q2}); th={}
    for k in sorted(set(q2)|{q2[0]-pd.offsets.QuarterEnd()}):
        if k<BASE: th[k]=-c[(c.index>k)&(c.index<=BASE)].sum()
    th[BASE]=0.0
    for q in q2:
        if q>BASE: th[q]=c[(c.index>BASE)&(c.index<=q)].sum()
    return pd.Series(th).sort_index()
cm=ch[model_lab(ch)&~ch.grp.isin(['ITSV','DATA','HARD'])].assign(sw=lambda x:(x.grp=='SOFT').astype(float)).dropna(subset=['bid'])
cf=ch[FUNDLAB(ch)].assign(sw=lambda x:(x.fsec=='software').astype(float))
for nm,x in [('Model',cm),('Fund',cf)]:
    for y,Y in [('dmark','Mark'),('db99','Below')]:
        th=theta(x,y); D=th[th.index<=BASE].diff().abs().max(); dg=2 if Y=='Mark' else 1
        put(f'Th{nm}{Y}FirstSv',th.iloc[0],f'f{dg}',f'registered {nm} chain-linked index at Dec 2022',sign=True)
        L(f'[chain] {nm} registered {y}: {len(x)} changes, {x.pkey.nunique()} positions | path '+'; '.join(f'{q:%Y-%m} {v:+.2f}' for q,v in th.items())+f' | D {D:.2f}')
write_macros(f'{OUT}/numbers-r7-01b-chain-paths.tex',list(MAC),'Draft 7 rerun item 1 add-on: registered chain-linked index paths (r7_01b_chain_paths.py)')
L.save()
