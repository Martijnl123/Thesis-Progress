# Hybrid position links (26 Nov 2026). v2_stablekey.py links positions within each fund separately, and its greedy
# matcher breaks ties between equal scores with np.argsort, whose order differs between machines/numpy builds: on this PC
# 562 of 366,903 build-A links differ from the 21 Nov cloud run (all tie swaps), which moves C1 by about 0.04.
# To isolate the effect of adding CCS, the hybrid build keeps the 21 Nov links for every fund except CCS and takes the
# links this PC computes for CCS (the only fund whose data changed). Chains -> sid exactly as in v2_stablekey.py.
# Usage: python hybrid_sid.py <new build dir> <hybrid out dir>
import sys, os, shutil, pandas as pd
N, H = sys.argv[1:3]; CCS = 1702510
os.makedirs(H, exist_ok=True)
for f in ('positions.parquet', 'pairs_0.5.parquet'): shutil.copyfile(f'{N}/{f}', f'{H}/{f}')
p = pd.read_parquet(f'{N}/positions.parquet', columns=['pos_id', 'cik'])
loc = pd.read_parquet(f'{N}/links.parquet'); cik = p.set_index('pos_id').cik
loc = loc[cik.reindex(loc.pos_a).values == CCS]
L = pd.concat([pd.read_parquet(f'{N}/aux/links_21nov_nonccs.parquet'), loc], ignore_index=True)
assert L.pos_a.is_unique and L.pos_b.is_unique
L.to_parquet(f'{H}/links.parquet', index=False)
succ = dict(zip(L.pos_a, L.pos_b)); has_pred = set(L.pos_b); sid = {}
for s in p.pos_id[~p.pos_id.isin(has_pred)]:
    x = s
    while True:
        sid[x] = s
        if x not in succ: break
        x = succ[x]
pd.DataFrame({'pos_id': list(sid.keys()), 'sid': list(sid.values())}).to_parquet(f'{H}/sid.parquet', index=False)
print(f'links: {len(L):,} ({len(loc):,} CCS links from this PC); positions {len(p):,}, with sid {len(sid):,}')
