"""Pull Form N-PORT loan holdings filing by filing from EDGAR (primary_doc.xml) and write them in the layout of the
per-quarter loan extracts (loans_<quarter>.csv.gz) and of the aggregated quarter-end file (nport_qe_loans.csv.gz).

Why: the SEC N-PORT data set with October to December 2026 filings comes out in January 2027, after the thesis deadline.
The pre-analysis plan (v4, 'N-PORT') parses the 30 September 2026 filings one by one, with this script tested first on
the 30 June 2026 filings (edgar_compare_june.py compares the result with the June 2026 rows of nport_qe_loans).

Hold-out lock: --period must be 2026-06-30 or earlier unless the plan is posted (holdout_lock.py). Filings are chosen by
the period of report in the EDGAR filing list, so no September 2026 filing is downloaded while the lock is closed; a
filing whose own report date turns out to be later than allowed is dropped unread and listed.

Fund universe (--series): registrants and series that held loans (asset category LON) in the N-PORT data sets
2025q3 to 2026q2, plus the funds with loans in the 30 June 2026 rows of loans_2026q3 (added 7 Nov 2026 after the June
test: 45 funds that first held loans by June were missing; only 30 June rows were read). A trust files one NPORT-P per series, so
for each candidate filing the small EDGAR header is read first and the holdings file is downloaded only for series in
the universe.

Rules copied from the data-set build (notes-on-inputs.md, 'N-PORT quarter-end loan holdings'): asset category LON;
report date equal to the quarter-end; each series' latest filing; unit PA and USD only for the aggregate; one row per
report date, fair value level, issuer name, issuer title, maturity date and coupon, with n_series, total_balance,
total_value, price_median/min/max (value / balance x 100), issuer_lei and n_default.

Output (under --out):
  nport_loans_<period>.csv.gz     every LON holding, columns of loans_<quarter>.csv.gz
  nport_qe_loans_<period>.csv.gz  the aggregate, columns of nport_qe_loans.csv.gz
  nport_manifest_<period>.csv     one line per candidate filing (kept, skipped and why)
  cache/                          gzipped raw downloads, reused on reruns

Usage: python edgar_pull_nport.py --period 2026-06-30 --filed-by 2026-09-30 --out OUT --ua "Name email"
       add --limit-registrants 10 for a smoke test.
Standard library only (Python 3.8+).
"""
import argparse, csv, datetime as dt, gzip, html, io, os, re, statistics, sys, time
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from holdout_lock import check_period
from edgar_common import Sec, NotFound, to_dmy, clean_cell

FORMS = {'NPORT-P', 'NPORT-P/A'}
COLS = ['accession_number', 'holding_id', 'sub_type', 'report_ending_period', 'report_date', 'filing_date', 'registrant_cik',
        'registrant_name', 'series_id', 'series_name', 'net_assets', 'issuer_name', 'issuer_lei', 'issuer_title', 'issuer_cusip',
        'balance', 'unit', 'other_unit_desc', 'currency_code', 'currency_value', 'exchange_rate', 'percentage', 'payoff_profile',
        'issuer_type', 'investment_country', 'is_restricted_security', 'fair_value_level', 'isin', 'ticker', 'other_identifiers',
        'maturity_date', 'coupon_type', 'annualized_rate', 'is_default']
QE_COLS = ['report_date', 'fair_value_level', 'issuer_name', 'issuer_title', 'maturity_date', 'annualized_rate', 'n_series',
           'total_balance', 'total_value', 'price_median', 'price_min', 'price_max', 'issuer_lei', 'n_default']


def local(tag):
    return tag.rsplit('}', 1)[-1]


def kid(el, name):
    for c in el:
        if local(c.tag) == name:
            return c
    return None


def txt(el, name):
    c = kid(el, name)
    return (c.text or '').strip() if c is not None and c.text else ''


def parse_nport(data):
    """Return (general info dict, list of holding dicts for asset category LON)."""
    gen, fund, sub_type = {}, {}, ''
    loans = []
    n_all = 0
    for ev, el in ET.iterparse(io.BytesIO(data), events=('end',)):
        t = local(el.tag)
        if t == 'submissionType':
            sub_type = (el.text or '').strip()
        elif t == 'genInfo':
            gen = {local(c.tag): (c.text or '').strip() for c in el}
        elif t == 'fundInfo':
            fund = {'netAssets': txt(el, 'netAssets')}
        elif t == 'invstOrSec':
            n_all += 1
            ac = txt(el, 'assetCat') or (kid(el, 'assetConditional').get('assetCat', '') if kid(el, 'assetConditional') is not None else '')
            if ac == 'LON':
                cur = txt(el, 'curCd')
                xr = ''
                cc = kid(el, 'currencyConditional')
                if not cur and cc is not None:
                    cur, xr = cc.get('curCd', ''), cc.get('exchangeRt', '')
                ic = txt(el, 'issuerCat')
                if not ic and kid(el, 'issuerConditional') is not None:
                    ic = kid(el, 'issuerConditional').get('issuerCat', '')
                ids = kid(el, 'identifiers')
                isin = ticker = ''
                other = []
                if ids is not None:
                    for c in ids:
                        n = local(c.tag)
                        if n == 'isin':
                            isin = c.get('value', '')
                        elif n == 'ticker':
                            ticker = c.get('value', '')
                        elif n == 'other':
                            other.append(f"{c.get('otherDesc', '')}:{c.get('value', '')}")
                ds = kid(el, 'debtSec')
                loans.append({
                    'issuer_name': txt(el, 'name'), 'issuer_lei': txt(el, 'lei'), 'issuer_title': txt(el, 'title'),
                    'issuer_cusip': txt(el, 'cusip'), 'balance': txt(el, 'balance'), 'unit': txt(el, 'units'),
                    'other_unit_desc': txt(el, 'descOthUnits'), 'currency_code': cur, 'currency_value': txt(el, 'valUSD'),
                    'exchange_rate': xr, 'percentage': txt(el, 'pctVal'), 'payoff_profile': txt(el, 'payoffProfile'),
                    'issuer_type': ic, 'investment_country': txt(el, 'invCountry'), 'is_restricted_security': txt(el, 'isRestrictedSec'),
                    'fair_value_level': txt(el, 'fairValLevel'), 'isin': isin, 'ticker': ticker, 'other_identifiers': ';'.join(other),
                    'maturity_date': (lambda m: to_dmy(m) if m[:1].isdigit() else '')(txt(ds, 'maturityDt')) if ds is not None else '',  # 'N/A' is blank in the data sets
                    'coupon_type': txt(ds, 'couponKind') if ds is not None else '',
                    'annualized_rate': txt(ds, 'annualizedRt') if ds is not None else '',
                    'is_default': txt(ds, 'isDefault') if ds is not None else ''})
            el.clear()
    gen['netAssets'] = fund.get('netAssets', '')
    gen['submissionType'] = sub_type
    gen['n_holdings'] = n_all
    return gen, loans


def header_series(header):
    return sorted(set(re.findall(r'<SERIES-ID>\s*(S\d{9})', html.unescape(header))))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def aggregate(rows, period):
    """Aggregate to the nport_qe_loans layout (USD, unit PA, report date = period, latest filing per series)."""
    rd = to_dmy(period)
    rows = [r for r in rows if r['report_date'] == rd and r['unit'] == 'PA' and r['currency_code'] == 'USD']
    latest = {}
    for r in rows:
        fund = r['series_id'] or ('CIK' + r['registrant_cik'])
        k = (fund, r['report_date'])
        cand = (r['_filing_date_iso'], r['accession_number'])
        if k not in latest or cand > latest[k]:
            latest[k] = cand
    rows = [r for r in rows if (r['_filing_date_iso'], r['accession_number']) == latest[(r['series_id'] or ('CIK' + r['registrant_cik']), r['report_date'])]]
    def rate_key(x):        # '9.89392000' and '9.89392' are one line, as in the data-set build
        try:
            return repr(float(x))
        except (TypeError, ValueError):
            return x or ''
    g = {}
    for r in rows:
        k = (r['report_date'], r['fair_value_level'], r['issuer_name'], r['issuer_title'], r['maturity_date'], rate_key(r['annualized_rate']))
        g.setdefault(k, []).append(r)
    out = []
    for k, rs in g.items():
        bal = [fnum(r['balance']) for r in rs]
        val = [fnum(r['currency_value']) for r in rs]
        px = [100 * v / b for b, v in zip(bal, val) if b and v is not None and b > 0]
        lei = next((r['issuer_lei'] for r in rs if r['issuer_lei'] and r['issuer_lei'].upper() != 'N/A'), '')
        # n_series counts holding lines, as in nport_qe_loans (June 2026 test, 7 Nov: the data-set file sums to the number
        # of holding rows, 31,543 Level 2 rows, not to distinct funds)
        out.append(dict(zip(QE_COLS[:6], k), n_series=len(rs),
                        total_balance=sum(b for b in bal if b is not None), total_value=sum(v for v in val if v is not None),
                        price_median=statistics.median(px) if px else '', price_min=min(px) if px else '', price_max=max(px) if px else '',
                        issuer_lei=lei, n_default=sum(1 for r in rs if r['is_default'] == 'Y')))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--period', required=True)
    ap.add_argument('--filed-by', required=True)
    ap.add_argument('--series', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nport_series.csv'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--ua', default='')
    ap.add_argument('--limit-registrants', type=int, default=0)
    ap.add_argument('--allow-sep2026', action='store_true')
    a = ap.parse_args()

    maxp = check_period(a.period, a.allow_sep2026)
    run_tag = dt.date.today().isoformat()
    os.makedirs(a.out, exist_ok=True)
    sec = Sec(a.ua, os.path.join(a.out, 'cache'))
    uni = {}
    with open(a.series, encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            uni.setdefault(int(float(r['registrant_cik'])), set()).add(r['series_id'] or '')
    regs = sorted(uni)
    if a.limit_registrants:
        regs = regs[:a.limit_registrants]
    print(f'{len(regs)} registrants, {sum(len(uni[c]) for c in regs)} series in the universe; period {a.period}; filed by {a.filed_by}', flush=True)

    loans, manifest = [], []
    t0 = time.time()
    for k, cik in enumerate(regs, 1):
        try:
            sub, filings = sec.submissions(cik, run_tag)
        except NotFound:
            manifest.append({'registrant_cik': cik, 'accession_number': '', 'status': 'no submissions file'})
            continue
        sel = [f for f in filings if f.get('form') in FORMS and f.get('reportDate') == a.period and f.get('filingDate', '9') <= a.filed_by]
        for f in sel:
            acc = f['accessionNumber']
            m = {'registrant_cik': cik, 'accession_number': acc, 'form': f['form'], 'filing_date': f['filingDate'],
                 'report_date': f['reportDate'], 'series_id': '', 'status': '', 'n_holdings': 0, 'n_loans': 0}
            try:
                series = header_series(sec.header_text(cik, acc))
                m['series_id'] = ';'.join(series)
                wanted = (set(series) & uni[cik]) or ('' in uni[cik] and not series)
                if not wanted:
                    m['status'] = 'series not in loan universe'
                    manifest.append(m)
                    continue
                data = sec.get(sec.folder(cik, acc) + '/primary_doc.xml', f'filings/{cik}/{acc}/primary_doc.xml')
                gen, ls = parse_nport(data)
                rpd = gen.get('repPdDate', '')
                if not rpd or rpd > maxp:     # lock: a holdings date later than allowed is never written
                    m['status'] = f'report date {rpd or "missing"} outside the lock, dropped'
                    manifest.append(m)
                    continue
                if rpd != a.period:
                    m['status'] = f'report date in the filing is {rpd}, not {a.period}; dropped'
                    manifest.append(m)
                    continue
                base = {'accession_number': acc, 'sub_type': gen.get('submissionType') or f['form'],
                        'report_ending_period': to_dmy(gen.get('repPdEnd', '')), 'report_date': to_dmy(rpd),
                        'filing_date': to_dmy(f['filingDate']), '_filing_date_iso': f['filingDate'],
                        'registrant_cik': str(cik), 'registrant_name': gen.get('regName', ''),
                        'series_id': gen.get('seriesId', '') or (series[0] if series else ''),
                        'series_name': gen.get('seriesName', ''), 'net_assets': gen.get('netAssets', '')}
                for i, l in enumerate(ls, 1):
                    r = dict(base)
                    r.update(l)
                    r['holding_id'] = f'{acc}-{i}'
                    loans.append(r)
                m.update({'status': 'kept', 'n_holdings': gen.get('n_holdings', 0), 'n_loans': len(ls), 'series_id': base['series_id']})
            except NotFound as e:
                m['status'] = f'not found: {e}'
            except ET.ParseError as e:
                m['status'] = f'XML parse error: {e}'
            manifest.append(m)
        if k % 25 == 0 or k == len(regs):
            kept = sum(1 for x in manifest if x.get('status') == 'kept')
            print(f'  {k}/{len(regs)} registrants, {kept} filings kept, {len(loans):,} loan lines, {sec.n_requests} requests, '
                  f'{sec.n_bytes / 1e6:.0f} MB, {time.time() - t0:.0f}s', flush=True)

    with gzip.open(os.path.join(a.out, f'nport_loans_{a.period}.csv.gz'), 'wt', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction='ignore')
        w.writeheader()
        for r in loans:
            w.writerow({c: clean_cell(r.get(c, '')) for c in COLS})
    agg = aggregate(loans, a.period)
    with gzip.open(os.path.join(a.out, f'nport_qe_loans_{a.period}.csv.gz'), 'wt', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=QE_COLS)
        w.writeheader()
        w.writerows(agg)
    keys = sorted({k for m in manifest for k in m})
    with open(os.path.join(a.out, f'nport_manifest_{a.period}.csv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(manifest)
    kept = [x for x in manifest if x.get('status') == 'kept']
    print(f'done: {len(kept)} filings kept ({len({x["series_id"] for x in kept})} series), {len(loans):,} loan lines, '
          f'{len(agg):,} aggregated lines; {len(manifest) - len(kept)} candidate filings skipped (see manifest); '
          f'{sec.n_requests} requests, {sec.n_bytes / 1e6:.0f} MB, {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
