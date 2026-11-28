import pandas as pd, numpy as np
d=pd.read_pickle('raw.pkl')
M={'CDMO':'PHAR','Graduate products':'CPRD','PPE wholsale':'DIST','Vetinary':'HCPS','dental':'HCPS','engineering':'PROF',
 'entertainment / amusement':'CONS','food and beverage':'FOOD','haircare':'CPRD','hardware':'HARD','healthcare provider':'HCPS',
 'investor':'FUND','medical devices':'HCEQ','online travel agency':'CONS','petcare':'HCPS','priate credit fund':'FUND','printing':'COMM',
 'renawable electricity producers':'ENER','retail':'CPRD','skincare / makeup':'CPRD','telecom':'TELE','constructioon and building':'CONB'}
# ambiguous free text: keep software flag, drop from 26-group comparison
AMB=['bus tours','group purchaisng','infrastructure','manufacting','music']
g=d.group_code.astype(str).str.strip()
g=g.map(lambda s: M.get(s, s.upper() if s.upper()==s.upper() else s))
g=g.where(~d.group_code.astype(str).str.strip().isin(AMB), np.nan)
d['gc']=g
out=d[['code_id','gc','is_software_0_1','part']].rename(columns={'gc':'group_code'})
out.to_csv('coder1_all.csv',index=False)
o2=out.copy(); o2.loc[o2.part=='zephyr_weak',['group_code','is_software_0_1']]=[np.nan,np.nan]; o2.to_csv('coder1_noweak.csv',index=False)
print(out.group_code.value_counts().to_dict()); print(out.part.value_counts())
