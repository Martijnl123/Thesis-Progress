# Carry the June-panel-keyed files to a rebuild (26 Nov 2026) in two hops, both bookkeeping:
#   new build -> 21 Nov build A: content match on cik, date, description, principal, fair value (rows sharing all five in pos_id order);
#   build A -> June panel: the 21 Nov maps in sep2026/buildA_aux (posmap, oldpanel_map, position_borrower).
# buildA_local (this PC) has the same positions and pos_ids as the 21 Nov build A (checked: same rows, same pos_id order).
# Also writes, for the hybrid build, the 21 Nov build-A links translated to the new pos_ids for every fund except CCS.
# Usage: python make_aux_via_A.py <buildA_local> <new build dir> <21 Nov buildA_aux dir> <21 Nov buildA dir>
import sys, os, pandas as pd
A, N, X, CA = sys.argv[1:5]
CCS = 1702510
K = ['cik', 'ddate', 'ident', 'principal', 'fv']
def keyed(d):
    pos = pd.read_parquet(f'{d}/positions.parquet', columns=['pos_id'] + K).sort_values('pos_id')
    pos['k'] = pos.groupby(K, dropna=False).cumcount(); return pos
a = keyed(A); n = keyed(N)
m = n.merge(a, on=K + ['k'], suffixes=('_n', '_a'))
nm = n[~n.pos_id.isin(m.pos_id_n)]; am = a[~a.pos_id.isin(m.pos_id_a)]
print(f'build A rows {len(a):,}, new rows {len(n):,}, matched {len(m):,}')
print(f'  new rows without a match: {len(nm):,}, by fund and date:\n{nm.groupby(["cik", "ddate"]).size().to_string()}')
print(f'  build-A rows without a match: {len(am):,}, by fund and date:\n{am.groupby(["cik", "ddate"]).size().to_string()}')
a2n = m.set_index('pos_id_a').pos_id_n
os.makedirs(f'{N}/aux', exist_ok=True)
pm = pd.read_parquet(f'{X}/posmap.parquet')            # pos_id_n (build A) -> pos_id_o (June panel)
pm = pm[pm.pos_id_n.isin(a2n.index)].assign(pos_id_n=lambda x: x.pos_id_n.map(a2n))
pm.to_parquet(f'{N}/aux/posmap.parquet', index=False)
om = pd.read_parquet(f'{X}/oldpanel_map.parquet')      # pos_id (build A) -> pkey (June panel)
om = om[om.pos_id.isin(a2n.index)].assign(pos_id=lambda x: x.pos_id.map(a2n))
om.to_parquet(f'{N}/aux/oldpanel_map.parquet', index=False)
pb = pd.read_parquet(f'{X}/position_borrower.parquet')
pb = pb[pb.pos_id.isin(a2n.index)].assign(pos_id=lambda x: x.pos_id.map(a2n))
pb.to_parquet(f'{N}/aux/position_borrower.parquet', index=False)
print(f'posmap {len(pm):,}, oldpanel_map {len(om):,}, position_borrower {len(pb):,} rows')
# 21 Nov links for all funds except CCS, in new pos_ids
L = pd.read_parquet(f'{CA}/links.parquet'); cik = a.set_index('pos_id').cik
L = L[cik.reindex(L.pos_a).values != CCS]
ok = L.pos_a.isin(a2n.index) & L.pos_b.isin(a2n.index)
print(f'21 Nov links outside CCS: {len(L):,}, both ends matched: {int(ok.sum()):,}')
L = L[ok].assign(pos_a=lambda x: x.pos_a.map(a2n), pos_b=lambda x: x.pos_b.map(a2n))
L.to_parquet(f'{N}/aux/links_21nov_nonccs.parquet', index=False)
