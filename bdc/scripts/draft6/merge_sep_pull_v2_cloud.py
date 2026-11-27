# Cloud copy of merge_sep_pull_v2.py (26 Nov 2026, Carlyle deviation): the 21 Nov soi.tsv (only read, hash asserted)
# plus the Carlyle Credit Solutions rows (successor CIK 2144472, accession 0002144472-26-000013). Same logic, paths as args.
# Usage: python3 merge_sep_pull_v2_cloud.py <orig soi.tsv> <ccs soi.tsv> <out soi.tsv>
import hashlib, sys
so, sa, out = sys.argv[1:4]
sha = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
assert sha(so) == '1f474234f8b08ce7952b17444744815f672d059652fbf716f93768035c861da6', 'original soi.tsv changed'
def read(f):
    lines = open(f, encoding='utf-8', newline='').read().split('\n')
    hdr = lines[0].split('\t'); rows = [l.split('\t') for l in lines[1:] if l]
    assert all(len(r) == len(hdr) for r in rows); return hdr, rows
h1, r1 = read(so); h2, r2 = read(sa)
cols = h1 + [c for c in h2 if c not in h1]
with open(out, 'w', encoding='utf-8', newline='') as f:
    f.write('\t'.join(cols) + '\n')
    for hdr, rows in ((h1, r1), (h2, r2)):
        ix = {c: i for i, c in enumerate(hdr)}
        for r in rows: f.write('\t'.join(r[ix[c]] if c in ix else '' for c in cols) + '\n')
print(f'{len(r1):,} + {len(r2):,} = {len(r1)+len(r2):,} rows, {len(cols)} columns; SHA-256 {sha(out)}; original unchanged {sha(so)[:16]}')
