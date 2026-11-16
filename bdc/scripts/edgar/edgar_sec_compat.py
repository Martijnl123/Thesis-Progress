"""Turn a pulled soi.tsv (full EDGAR text and values) into the form the SEC BDC data sets would have given, so the pulled
quarter is built the same way as the history in positions.parquet.

The June 2026 test (7 Nov 2026) showed three conventions of the SEC data sets that the pull did not copy:
  1. values are rounded to 4 decimals (a rate of 0.000839 is 0.0008 in the data sets);
  2. the investment description (the typed InvestmentIdentifierAxis member) is cut at 255 characters, accents are
     removed (S.À R.L. becomes S.A R.L.) and other non-ASCII characters dropped (an en dash disappears, leaving the
     spaces around it);
  3. holdings whose cut descriptions are the same, and that agree on every other dimension, become ONE row: for each
     column the first value in the filing is kept, so principal and fair value can come from different loans.
Point 3 is a defect of the data sets, not something the thesis wants: such rows can carry a mark that belongs to no
loan. This script reproduces it only so that the pulled quarter is comparable with the history; the full pull is kept
next to it, and edgar_compare_june.py reports how many rows the merge affects.

Usage: python3 edgar_sec_compat.py <pulled src folder> <output src folder>
       (each folder holds edgar_<period>/soi.tsv)
"""
import csv, glob, os, sys, unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edgar_pull_bdc import VAL, META, EXTRA

NUMERIC = {'Investment Interest Rate', 'Investment, Basis Spread, Variable Rate', 'Investment Owned, Balance, Principal Amount',
           'Adjusted cost basis', 'Initial fair value of Investment', 'Investment, Interest Rate, Paid in Kind',
           'Investment, Interest Rate, Paid in Cash', 'Investment, Interest Rate, Floor', 'Investment shares',
           'Investment Owned, Net Assets, Percentage'}
IDENT = 'Investment, Identifier Axis'


def asc(s):
    # accented letters lose their accent (S.À R.L. -> S.A R.L.), other non-ASCII characters (en dash) are dropped
    return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')


def r4(v):
    try:
        return f'{float(v):.4f}'
    except ValueError:
        return v


def convert(src_file, out_file):
    csv.field_size_limit(10 ** 9)
    with open(src_file, encoding='utf-8', newline='') as f:
        r = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        h = next(r)
        rows = [dict(zip(h, x)) for x in r if len(x) == len(h)]
    vals = [c for c in h if c in VAL.values()]
    axes = [c for c in h if c not in META and c not in VAL.values() and c not in EXTRA and c != 'ident_full']
    merged, order = {}, []
    n_in = len(rows)
    for x in rows:
        full = x.get(IDENT, '')
        cut = asc(full[:255])
        key = (x['adsh'], x['ddate'], x['qtrs'], cut) + tuple(asc(x.get(c, '')) for c in axes if c != IDENT) + (x.get('custom_axes', ''),)
        m = merged.get(key)
        if m is None:
            m = dict(x)
            m[IDENT] = cut
            for c in axes:
                if c != IDENT:
                    m[c] = asc(m.get(c, ''))
            m['ident_full'] = full
            m['n_merged'] = 1
            for c in vals:
                if c in NUMERIC and m.get(c):
                    m[c] = r4(m[c])
            merged[key] = m
            order.append(key)
            continue
        m['n_merged'] += 1
        for c in vals:                      # first value in the filing wins, as in the data sets
            if not m.get(c) and x.get(c):
                m[c] = r4(x[c]) if c in NUMERIC else x[c]
    cols = h + [c for c in ('ident_full', 'n_merged') if c not in h]
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w', encoding='utf-8', newline='') as f:
        f.write('\t'.join(cols) + '\n')
        for k in order:
            f.write('\t'.join(str(merged[k].get(c, '')) for c in cols) + '\n')
    n_multi = sum(1 for k in order if merged[k]['n_merged'] > 1)
    print(f'{os.path.basename(os.path.dirname(src_file))}: {n_in:,} rows in, {len(order):,} out; {n_multi:,} output rows merge '
          f'{sum(merged[k]["n_merged"] for k in order if merged[k]["n_merged"] > 1):,} holdings that share a cut description')


def main():
    src, out = sys.argv[1], sys.argv[2]
    for f in sorted(glob.glob(os.path.join(src, '*', 'soi.tsv'))):
        convert(f, os.path.join(out, os.path.basename(os.path.dirname(f)), 'soi.tsv'))


if __name__ == '__main__':
    main()
