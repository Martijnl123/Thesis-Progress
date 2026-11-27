# Draft 6 (26 Nov 2026), #166: the power table (tab_power_d5.tex, written by draft5/d5_numbers.py) uses hyphens for minus signs.
# Writes tab_power_d6.tex with every hyphen-minus before a digit in a cell turned into a math minus. Values unchanged.
# Usage: python3 d6_tabfix.py <draft6 folder>
import sys, re
T=sys.argv[1]+'/tables'; s=open(f'{T}/tab_power_d5.tex').read()
out=re.sub(r'(?<![$\w])-(?=[0-9])',r'$-$',s); open(f'{T}/tab_power_d6.tex','w').write(out)
print(s.count('-'),'hyphens before;',out.count('$-$'),'math minus after')
# Layout only: the Step 3 table (tab_step3_d6.tex, written by d6_tables.py) was 77pt too wide; wrap its label column. Idempotent.
s=open(f'{T}/tab_step3_d6.tex').read(); s2=s.replace('\\begin{tabular}{lcccc}','\\begin{tabular}{L{6.2cm}cccc}',1)
open(f'{T}/tab_step3_d6.tex','w').write(s2); print('tab_step3_d6 label column wrapped:', s!=s2)
