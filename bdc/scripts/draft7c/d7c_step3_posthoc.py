import os as _os
_ROOT=_os.environ.get('THESIS_DATA',_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..','..','..')))  # repo root
import sys
sys.path.insert(0,(_ROOT+'/bdc/scripts/draft2'))
from d2_common import *
import d2_step3 as S3
p=load(); s=S3.frame(p)
u=s[s.cls.isin(['traded','private'])].copy(); u['Tsw']=u.T_*u.sw
x=u.dropna(subset=['dm202606']).copy()
r=S3.ols('dm202606 ~ T_ + Tsw + C(fs)',x); print('base',r.params['Tsw'],r.bse['Tsw'],len(x))
x['mk']=x.mark
ctr=['spr','pikp','ln_bsize','nfam','mk']
xx=x.dropna(subset=ctr).copy()
r=S3.ols('dm202606 ~ T_ + Tsw + C(fs)',xx); print('base same n',r.params['Tsw'],r.bse['Tsw'],len(xx))
f='dm202606 ~ T_ + Tsw + C(fs) + '+' + '.join(ctr)+' + '+' + '.join(f'{c}:sw' for c in ctr)
r=S3.ols(f,xx); print('controls x soft',r.params['Tsw'],r.bse['Tsw'],len(xx))
# L3-only and traded borrower classes: software vs other, fund FE
for c in ['L3 only','traded borrower','private','traded']:
    y=s[(s.cls==c)].dropna(subset=['dm202606'])
    rr=S3.ols('dm202606 ~ sw + C(cik)',y); print(c, round(rr.params['sw'],2), round(rr.bse['sw'],2), len(y))
y=s[s.cls.isin(['private','L3 only','traded borrower'])].dropna(subset=['dm202606'])
rr=S3.ols('dm202606 ~ sw + C(cik)',y); print('all unpriced', round(rr.params['sw'],2), round(rr.bse['sw'],2), len(y), len(y)/len(s.dropna(subset=['dm202606'])))
# two-way-ish: cluster by manager for kappa
x['fam']=x.family.astype(str)
r=S3.ols('dm202606 ~ T_ + Tsw + C(fs)',x,clu='fam'); print('kappa mgr-cluster',r.params['Tsw'],r.bse['Tsw'])
