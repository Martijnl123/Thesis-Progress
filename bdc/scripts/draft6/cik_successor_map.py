# Successor-CIK map for the September 2026 pull (documented deviation, 26 Nov 2026).
# Carlyle Credit Solutions (old CIK 1702510) merged into a successor trust (CIK 2144472) on 4 Nov 2026; the successor
# filed the 30 Sep 2026 10-Q on 13 Nov 2026 (accession 0002144472-26-000013), before the 20 Nov cut-off. The panel keys
# funds by CIK, so the successor's rows are given the old CIK (and the old CIK's printed name, which sets the manager
# family) so that its Sep 2026 positions link to the same fund's June 2026 positions. Nothing else is changed.
# Usage: python cik_successor_map.py <raw.parquet>   (rewrites it in place; the as-loaded file is kept as raw_asfiled.parquet)
import sys, os, shutil, pandas as pd
MAP = {2144472: 1702510}
f = sys.argv[1]
shutil.copyfile(f, os.path.join(os.path.dirname(f), 'raw_asfiled.parquet'))
d = pd.read_parquet(f)
for new, old in MAP.items():
    m = d.cik == new
    prev = d[(d.cik == old)].sort_values('filed')
    name = prev.bdc.dropna().iloc[-1] if len(prev) else None
    print(f'CIK map {new} -> {old}: {int(m.sum())} rows; names as filed {sorted(d.loc[m, "bdc"].dropna().unique())} -> {name!r}')
    d.loc[m, 'cik'] = old
    if name is not None: d.loc[m, 'bdc'] = name
d.to_parquet(f, index=False)
