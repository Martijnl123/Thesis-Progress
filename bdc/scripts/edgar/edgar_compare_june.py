"""June 2026 test of the EDGAR pulls (pre-analysis plan v4: the pull must reproduce the June 2026 rows before it is used).

BDC leg: runs the pulled soi.tsv through the unchanged v2_load.py and v2_clean.py, and compares the 30 June 2026 debt
positions with those in v2/debt.parquet (the SEC data-set build; identical to the June rows of positions.parquet).
N-PORT leg: compares the pulled quarter-end aggregate with the 30 June 2026 lines of v2/nport_qe_loans.parquet.

Pass rules (fixed before the test is run, 7 Nov 2026):
  BDC    at least 99.5% of data-set June positions found in the pull with the same fund, description, principal and fair
         value, and at least 99.5% the other way; at least 99.5% of data-set June principal covered; on matched rows,
         spread, coupon, maturity and the software, first-lien, PIK and non-accrual flags agree on at least 99.5%.
  N-PORT at least 98% of data-set June Level 2 lines found with the same level, issuer, title, maturity and coupon, and
         on those lines the median price agrees within 0.01 points on at least 98%.
A failure is reported as such; the pull is then fixed and the test rerun, and every run is kept.

Runs, all kept in bdc/v2/ (7 Nov 2026, on the same pulled files from a local PC):
  1. results_edgar_june_test_2026-11-07.txt       pull as written: FAIL on both legs.
  2. results_edgar_june_test_2026-11-07_run2.txt  BDC through edgar_sec_compat.py (4-decimal rounding, descriptions cut
     at 255 characters, non-ASCII dropped, holdings with the same cut description merged); N-PORT n_series counts
     holding lines and equal coupons written differently are one line: FAIL on both legs (BDC 99.46%).
  3. results_edgar_june_test_2026-11-07_run3.txt  accents transliterated (S.A R.L.) instead of dropped; the N-PORT
     comparison counts each line once (run 1 and 2 counted lines with a repeated key twice, which flattered run 1).
  4. results_edgar_june_test_2026-11-07_run4.txt  N-PORT re-pulled on the PC with 45 more funds in nport_series.csv
     (funds that first held loans by June, found in the 30 June rows of loans_2026q3; only those
     rows); BDC files unchanged: PASS on both legs (BDC same as run 3; N-PORT 99.63% of Level 2 lines, prices 99.97%).
  The 'full' BDC variant (no data-set conventions) is reported in every run but is not the test.

Usage: python3 edgar_compare_june.py --pull <folder with bdc/ and nport/ from the PC run> --out <results.txt>
"""
import argparse, os, subprocess, sys
import numpy as np, pandas as pd

S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # bdc/scripts
V = os.path.join(os.path.dirname(S), 'v2')                                # bdc/v2
JUNE = pd.Timestamp('2026-06-30')
lines = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    lines.append(s)


def norm_ws(s):
    return s.fillna('').str.replace(r'\s+', ' ', regex=True).str.strip()


def bdc_leg(pull, variant):
    say(f'\n=== BDC leg, variant "{variant}" ===')
    src = os.path.join(pull, 'bdc', 'src')
    raw = os.path.join(pull, 'bdc', f'raw_edgar_{variant}.parquet')
    debt = os.path.join(pull, 'bdc', f'debt_edgar_{variant}.parquet')
    cmds = []
    if variant == 'sec':        # copy the data-set conventions first (edgar_sec_compat.py)
        cmds.append([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'edgar_sec_compat.py'), src, src + '_sec'])
        src = src + '_sec'
    cmds += [[sys.executable, os.path.join(S, 'v2_load.py'), src, raw], [sys.executable, os.path.join(S, 'v2_clean.py'), raw, debt]]
    for cmd in cmds:
        r = subprocess.run(cmd, capture_output=True, text=True)
        say('$', ' '.join(os.path.basename(c) if i < 2 else c for i, c in enumerate(cmd)))
        for l in (r.stdout + r.stderr).strip().splitlines()[-6:]:
            say('   ', l)
        if r.returncode != 0:
            say('STOPPED: the step above failed.')
            return False
    cols = ['cik', 'bdc', 'adsh', 'ddate', 'ident', 'principal', 'fv', 'cost', 'spread', 'rate', 'maturity', 'software',
            'first_lien', 'pik', 'nonaccrual', 'family', 'ptype']
    a = pd.read_parquet(os.path.join(V, 'debt.parquet'), columns=cols)
    b = pd.read_parquet(debt, columns=cols)
    a, b = a[a.ddate == JUNE].copy(), b[b.ddate == JUNE].copy()
    say(f'\nBDC, 30 June 2026 debt positions: data sets {len(a):,} in {a.cik.nunique()} funds; EDGAR pull {len(b):,} in {b.cik.nunique()} funds')
    fa, fb = set(a.cik), set(b.cik)
    say(f'  funds only in the data sets: {len(fa - fb)}; only in the pull: {len(fb - fa)}')
    if fa - fb:
        g = a[a.cik.isin(fa - fb)].groupby(['cik', 'bdc']).agg(rows=('ident', 'size'), principal_m=('principal', lambda x: x.sum() / 1e6))
        say(g.sort_values('rows', ascending=False).head(15).round(1).to_string())
    pick_a = a.groupby('cik').adsh.first()
    pick_b = b.groupby('cik').adsh.first()
    both = pick_a.index.intersection(pick_b.index)
    same = (pick_a[both] == pick_b[both])
    say(f'  same filing chosen for {same.sum()} of {len(both)} funds in both ({same.mean():.1%})')
    if (~same).any():
        say('  different filing:', ', '.join(f'{c} ({pick_a[c]} vs {pick_b[c]})' for c in both[~same.values][:10]))
    key = ['cik', 'ident', 'principal', 'fv']
    res = {}
    for label, A, B in [('exact description', a, b), ('description with spaces normalised', a.assign(ident=norm_ws(a.ident)), b.assign(ident=norm_ws(b.ident)))]:
        A = A.assign(_n=A.groupby(key).cumcount())
        B = B.assign(_n=B.groupby(key).cumcount())
        m = A.merge(B, on=key + ['_n'], how='outer', suffixes=('_a', '_b'), indicator=True)
        in_a = (m._merge != 'right_only').sum()
        found = (m._merge == 'both').sum()
        in_b = (m._merge != 'left_only').sum()
        cov = m.loc[m._merge == 'both', 'principal'].sum() / A.principal.sum()
        res[label] = (found / in_a, found / in_b, cov, m)
        say(f'  {label}: {found / in_a:.2%} of data-set rows found in the pull, {found / in_b:.2%} of pull rows found in the data sets, '
            f'{cov:.2%} of data-set principal')
    r1, r2, cov, m = res['exact description']
    mm = m[m._merge == 'both']
    agree = {}
    for c in ['spread', 'rate', 'cost']:
        x, y = mm[c + '_a'], mm[c + '_b']
        agree[c] = ((x - y).abs().le(1e-9) | (x.isna() & y.isna())).mean()
    x, y = mm.maturity_a, mm.maturity_b
    agree['maturity'] = ((x == y) | (x.isna() & y.isna())).mean()
    for c in ['software', 'first_lien', 'pik', 'nonaccrual', 'family', 'ptype', 'bdc']:
        agree[c] = (mm[c + '_a'] == mm[c + '_b']).mean()
    say('  agreement on matched rows: ' + ', '.join(f'{k} {v:.2%}' for k, v in agree.items()))
    miss = m[m._merge == 'left_only']
    if len(miss):
        say('  examples of data-set rows not found in the pull:')
        say(miss[['cik', 'ident', 'principal', 'fv']].head(8).to_string(index=False))
    extra = m[m._merge == 'right_only']
    if len(extra):
        say('  examples of pull rows not in the data sets:')
        say(extra[['cik', 'ident', 'principal', 'fv']].head(8).to_string(index=False))
    flags = ['spread', 'rate', 'maturity', 'software', 'first_lien', 'pik', 'nonaccrual']
    ok = r1 >= 0.995 and r2 >= 0.995 and cov >= 0.995 and all(agree[c] >= 0.995 for c in flags)
    say(f'BDC leg: {"PASS" if ok else "FAIL"} (rules: 99.5% both ways, 99.5% of principal, 99.5% field agreement)')
    return ok


def nport_leg(pull):
    f = os.path.join(pull, 'nport', 'nport_loans_2026-06-30.csv.gz')
    if not os.path.isfile(f):
        say('\nN-PORT leg: no pulled file found, skipped')
        return None
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import edgar_pull_nport as NP
    rows = pd.read_csv(f, dtype=str, keep_default_na=False)
    rows['_filing_date_iso'] = pd.to_datetime(rows.filing_date, format='%d-%b-%Y').dt.strftime('%Y-%m-%d')
    b = pd.DataFrame(NP.aggregate(rows.to_dict('records'), '2026-06-30')).astype(str)   # same rules as the pull script
    b['fair_value_level'] = b.fair_value_level.replace('N/A', 'NA')
    a = pd.read_parquet(os.path.join(V, 'nport_qe_loans.parquet'))
    a = a[a.report_date == JUNE].copy()
    for c in ['annualized_rate', 'n_series', 'total_balance', 'price_median']:
        b[c] = pd.to_numeric(b[c], errors='coerce')
    b['lvl'] = b.fair_value_level.fillna('NA').str.strip()
    b['maturity'] = pd.to_datetime(b.maturity_date, format='%d-%b-%Y', errors='coerce')
    for d in (a, b):
        d['rk'] = d.annualized_rate.round(4)
        d['mk'] = d.maturity.dt.strftime('%Y-%m-%d').fillna('')
        d['nk'] = d.issuer_name.fillna('').str.strip()
        d['tk'] = d.issuer_title.fillna('').str.strip()
    key = ['lvl', 'nk', 'tk', 'mk', 'rk']
    say(f'\nN-PORT, 30 June 2026 loan lines: data set {len(a):,} ({(a.lvl == "2").sum():,} Level 2, {a.n_series.sum():,} fund holdings); '
        f'EDGAR pull {len(b):,} ({(b.lvl == "2").sum():,} Level 2, {b.n_series.sum():,} fund holdings)')
    # one row per line on each side (run 3: a left merge on a side with repeated keys counted some lines twice)
    ag = a.groupby(key, dropna=False).agg(n_series_a=('n_series', 'sum'), price_median_a=('price_median', 'first')).reset_index()
    bg = b.groupby(key, dropna=False).agg(n_series_b=('n_series', 'sum'), price_median_b=('price_median', 'first')).reset_index()
    m = ag.merge(bg, on=key, how='outer', indicator=True)
    l2 = m[m.lvl == '2']
    ds = l2[l2._merge != 'right_only']
    hit = l2[l2._merge == 'both']
    found = len(hit) / len(ds)
    px = (hit.price_median_a - hit.price_median_b).abs().le(0.01).mean()
    ns = (hit.n_series_a == hit.n_series_b).mean()
    allf = (m._merge == 'both').sum() / (m._merge != 'right_only').sum()
    say(f'  Level 2 lines found: {found:.2%}; same median price (within 0.01): {px:.2%}; same number of holding lines: {ns:.2%}; all levels found: {allf:.2%}')
    say(f'  Level 2 holdings on lines only in the data set: {int(l2[l2._merge == "left_only"].n_series_a.sum())}; only in the pull: {int(l2[l2._merge == "right_only"].n_series_b.sum())}')
    if (ds._merge != 'both').any():
        say('  examples of Level 2 lines not found:')
        say(ds.loc[ds._merge != 'both', ['nk', 'tk', 'mk', 'rk', 'n_series_a', 'price_median_a']].head(8).to_string(index=False))
    man = os.path.join(pull, 'nport', 'nport_manifest_2026-06-30.csv')
    if os.path.isfile(man):
        mf = pd.read_csv(man, dtype=str)
        say('  candidate filings by status: ' + ', '.join(f'{k} {v}' for k, v in mf.status.str.split(':').str[0].value_counts().items()))
    ok = found >= 0.98 and px >= 0.98
    say(f'N-PORT leg: {"PASS" if ok else "FAIL"} (rules: 98% of Level 2 lines, 98% same price)')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pull', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    say(f'EDGAR pull June 2026 test, pull folder {a.pull}')
    have = os.path.isdir(os.path.join(a.pull, 'bdc', 'src'))
    b = bdc_leg(a.pull, 'sec') if have else None
    if have:
        bdc_leg(a.pull, 'full')     # reported only: the full pull differs from the data sets where they merged holdings
    n = nport_leg(a.pull)
    say(f'\nResult: BDC {"PASS" if b else ("not run" if b is None else "FAIL")}; N-PORT {"PASS" if n else ("not run" if n is None else "FAIL")}')
    with open(a.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
