"""Pull BDC schedules of investments filing by filing from EDGAR (XBRL) and write them in the SEC data-set soi.tsv layout,
so that v2_load.py, v2_clean.py and the rest of the v2 pipeline read them without changes.

Why: the SEC BDC data sets are published by filing month, so the file with November 2026 filings comes after the
thesis cut-off. The pre-analysis plan (v4, 'Source of the September 2026 BDC data') therefore pulls the 30 September 2026
10-Qs one by one. Before that, this script must reproduce the 30 June 2026 rows of positions.parquet
(edgar_compare_june.py checks that).

Hold-out lock: --period must be 2026-06-30 or earlier unless the plan is posted (holdout_lock.py). Filings are chosen
by the period of report in the EDGAR filing list, so no September 2026 filing is downloaded while the lock is closed,
and contexts dated after the allowed period are never written.

What it does, per BDC in --ciks:
  1. filing list from data.sec.gov/submissions; keep 10-Q, 10-K (and /A, 10-QT, 10-KT) whose period of report equals
     --period and whose filing date is on or before --filed-by;
  2. the filing folder's index.json; the XBRL instance (the *_htm.xml file EDGAR extracts from inline XBRL) and the
     label linkbase (*_lab.xml) for the labels of the fund's own members;
  3. every context with a typed InvestmentIdentifierAxis member becomes one row per (date, quarters, full set of
     dimensions), as in the SEC soi.tsv; standard axes are columns named by their label without ' [Axis]'; explicit
     members are written as labels (standard members without ' [Member]', the fund's own members as labelled in its
     filing); custom axes are not columns (as in soi.tsv) but are listed in the extra column custom_axes;
  4. the schedule tags the loader reads are columns named by their SOI label.

Output (under --out):
  src/edgar_<period>/soi.tsv   one file for all filings (feed the src folder to v2_load.py)
  manifest_<period>.csv        one line per filing: rows written, contexts, problems
  not_found_<period>.csv       BDCs in --ciks with no matching filing
  cache/                       gzipped raw downloads (instances, label files, filing lists), reused on reruns

Usage (on a machine that can reach sec.gov):
  python edgar_pull_bdc.py --period 2026-06-30 --filed-by 2026-09-30 --ciks bdc_ciks.csv --out OUT --ua "Name email"
  add --limit 5 for a smoke test, --cik 1899017 for one fund.
Standard library only (Python 3.8+).
"""
import argparse, csv, datetime as dt, html, io, os, re, sys, time
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from holdout_lock import check_period
from edgar_common import Sec, NotFound, round_month_end, quarters_between, clean_cell

FORMS = {'10-Q', '10-K', '10-Q/A', '10-K/A', '10-QT', '10-KT', '10-QT/A', '10-KT/A'}
XBRLI = '{http://www.xbrl.org/2003/instance}'
XBRLDI = '{http://xbrl.org/2006/xbrldi}'
XSI_NIL = '{http://www.w3.org/2001/XMLSchema-instance}nil'
LINK = '{http://www.xbrl.org/2003/linkbase}'
XLINK = '{http://www.w3.org/1999/xlink}'
STD_NS = ('http://fasb.org/', 'http://xbrl.sec.gov/', 'http://xbrl.ifrs.org/', 'http://www.xbrl.org/')

# schedule tags the loader reads -> soi.tsv column label (cost and fair value go to the us-gaap/2026 labels;
# v2_load.py fills cost/fv from the 2025 labels when these are empty, so either works)
VAL = {
    'InvestmentInterestRate': 'Investment Interest Rate',
    'InvestmentBasisSpreadVariableRate': 'Investment, Basis Spread, Variable Rate',
    'InvestmentMaturityDate': 'Investment Maturity Date',
    'InvestmentOwnedBalancePrincipalAmount': 'Investment Owned, Balance, Principal Amount',
    'InvestmentOwnedAtCost': 'Adjusted cost basis',
    'InvestmentOwnedAtFairValue': 'Initial fair value of Investment',
    'InvestmentInterestRatePaidInKind': 'Investment, Interest Rate, Paid in Kind',
    'InvestmentInterestRatePaidInCash': 'Investment, Interest Rate, Paid in Cash',
    'InvestmentVariableInterestRateTypeExtensibleEnumeration': 'Investment, Variable Interest Rate, Type [Extensible Enumeration]',
    'InvestmentInterestRateFloor': 'Investment, Interest Rate, Floor',
    'InvestmentNonIncomeProducing': 'Investment, Non-income Producing [true false]',
    'InvestmentTypeExtensibleEnumeration': 'Investment, Type [Extensible Enumeration]',
    'InvestmentIssuerNameExtensibleEnumeration': 'Investment, Issuer Name [Extensible Enumeration]',
    'InvestmentAcquisitionDate': 'Investment, Acquisition Date',
    'InvestmentIndustrySectorExtensibleEnumeration': 'Investment, Industry Sector [Extensible Enumeration]',
    'InvestmentOwnedBalanceShares': 'Investment shares',
    'InvestmentOwnedPercentOfNetAssets': 'Investment Owned, Net Assets, Percentage',
}
IDENT_AXIS = 'InvestmentIdentifierAxis'
META = ['adsh', 'cik', 'name', 'ddate', 'qtrs', 'form', 'filed', 'period', 'inlineurl', 'cstm']
FIRST_AXES = ['Industry Sector Axis', 'Investment, Identifier Axis', 'Investment, Issuer Affiliation Axis', 'Investment Type Axis']
EXTRA = ['custom_axes', 'n_dup_facts']


def is_std(uri):
    return uri.startswith(STD_NS)


def camel(local):
    return re.sub(r'(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])', ' ', local)


def load_std_labels(path):
    lab = {}
    with open(path, encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            lab[row['tag']] = row['tlabel']
    return lab


def strip_suffix(label):
    return re.sub(r'\s*\[(Axis|Member|Domain|Line Items|Table)\]\s*$', '', label)


def parse_labels(xml_bytes):
    """Standard labels of the fund's own elements from its label linkbase: local name -> label."""
    out, pref = {}, {}
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return out
    for ll in root.iter(LINK + 'labelLink'):
        loc, lab, arcs = {}, {}, []
        for e in ll:
            t = e.get(XLINK + 'type')
            if t == 'locator':
                href = e.get(XLINK + 'href', '')
                frag = href.split('#')[-1]
                loc[e.get(XLINK + 'label')] = frag.split('_', 1)[1] if '_' in frag else frag
            elif t == 'resource':
                lab.setdefault(e.get(XLINK + 'label'), []).append((e.get(XLINK + 'role', ''), ''.join(e.itertext()).strip()))
            elif t == 'arc':
                arcs.append((e.get(XLINK + 'from'), e.get(XLINK + 'to')))
        for a, b in arcs:
            el = loc.get(a)
            if not el:
                continue
            for role, text in lab.get(b, []):
                rank = 0 if role.endswith('/role/label') else (1 if role.endswith('terseLabel') else 2)
                if el not in out or rank < pref[el]:
                    out[el], pref[el] = text, rank
    return out


class Instance:
    """Streaming parse of an XBRL instance: contexts with dimensions, and the facts of the tags in VAL."""

    def __init__(self, data):
        self.ns = {}
        self.contexts = {}
        self.facts = []
        self.n_facts_all = 0
        root = None
        depth = 0
        for ev, el in ET.iterparse(io.BytesIO(data), events=('start', 'end', 'start-ns')):
            if ev == 'start-ns':
                self.ns[el[0]] = el[1]
                continue
            if ev == 'start':
                depth += 1
                if root is None:
                    root = el
                continue
            depth -= 1
            if depth != 1:
                continue
            if el.tag == XBRLI + 'context':
                self._context(el)
            elif el.get('contextRef') is not None:
                self.n_facts_all += 1
                uri, local = el.tag[1:].split('}', 1) if el.tag.startswith('{') else ('', el.tag)
                if local in VAL and is_std(uri) and el.get(XSI_NIL) != 'true':
                    self.facts.append((el.get('contextRef'), local, (el.text or '').strip()))
            root.clear()

    def qname(self, s):
        s = (s or '').strip()
        p, _, local = s.rpartition(':')
        return self.ns.get(p, p), local

    def _context(self, el):
        dims = []
        for cont in (el.find(XBRLI + 'entity/' + XBRLI + 'segment'), el.find(XBRLI + 'scenario')):
            if cont is None:
                continue
            for m in cont:
                if m.tag == XBRLDI + 'explicitMember':
                    dims.append(('e',) + self.qname(m.get('dimension')) + self.qname(m.text))
                elif m.tag == XBRLDI + 'typedMember':
                    kids = list(m)
                    val = ''.join(kids[0].itertext()) if kids else ''.join(m.itertext())
                    dims.append(('t',) + self.qname(m.get('dimension')) + ('', val.strip()))
        per = el.find(XBRLI + 'period')
        inst = per.findtext(XBRLI + 'instant') if per is not None else None
        start = per.findtext(XBRLI + 'startDate') if per is not None else None
        end = per.findtext(XBRLI + 'endDate') if per is not None else None
        self.contexts[el.get('id')] = (dims, (inst or '').strip(), (start or '').strip(), (end or '').strip())


def pick_files(items):
    names = [(i.get('name', ''), int(i.get('size') or 0)) for i in items]
    inst = [n for n in names if n[0].lower().endswith('_htm.xml')]
    if not inst:   # filings that are not inline XBRL: the instance is the .xml that is not a linkbase or summary
        inst = [n for n in names if n[0].lower().endswith('.xml') and not re.search(r'(_cal|_def|_lab|_pre)\.xml$|filingsummary|metalinks', n[0].lower())]
    lab = [n for n in names if n[0].lower().endswith('_lab.xml')]
    inst = max(inst, key=lambda x: x[1])[0] if inst else None
    lab = max(lab, key=lambda x: x[1])[0] if lab else None
    return inst, lab


def conformed_name(header, cik):
    """EDGAR conformed name as of the filing (the 'name' field of the SEC data sets)."""
    name, out = None, None
    for line in html.unescape(header).splitlines():
        m = re.search(r'COMPANY CONFORMED NAME:\s*(.+)', line)
        if m:
            name = m.group(1).strip()
        m = re.search(r'CENTRAL INDEX KEY:\s*0*(\d+)', line)
        if m and name and int(m.group(1)) == int(cik):
            out = name
            break
    return out


def rows_from_instance(inst, meta, std_lab, fund_lab, maxp):
    """Pivot an instance into soi.tsv rows. Returns (rows, stats)."""
    def axis_col(uri, local):          # 'Investment, Identifier [Axis]' -> 'Investment, Identifier Axis', as in soi.tsv
        return re.sub(r'\s*\[Axis\]\s*$', ' Axis', std_lab.get(local, camel(local)))

    def member_label(uri, local):
        if is_std(uri):
            return strip_suffix(std_lab.get(local, camel(local)))
        return fund_lab.get(local) or (camel(local[:-6]) + ' [Member]' if local.endswith('Member') else camel(local))

    rows, dup, after = {}, 0, 0
    ctx_used = set()
    for cref, local, val in inst.facts:
        c = inst.contexts.get(cref)
        if c is None:
            continue
        dims, instant, start, end = c
        if not any(d[0] == 't' and d[2] == IDENT_AXIS and is_std(d[1]) for d in dims):
            continue
        if instant:
            ddate, qtrs = round_month_end(instant), 0
        elif end:
            ddate, qtrs = round_month_end(end), quarters_between(start, end) if start else 0
        else:
            continue
        if ddate > maxp:          # lock: never write anything dated after the allowed period
            after += 1
            continue
        segkey = tuple(sorted((d[1], d[2], d[3], d[4]) for d in dims))
        key = (ddate, qtrs, segkey)
        r = rows.get(key)
        if r is None:
            r = dict(meta)
            r['ddate'], r['qtrs'] = ddate, qtrs
            cust = []
            for kind, duri, dloc, muri, mloc in dims:
                if not is_std(duri):
                    cust.append(f'{dloc}={mloc if kind == "e" else "typed:" + mloc}')
                    continue
                r[axis_col(duri, dloc)] = mloc if kind == 't' else member_label(muri, mloc)
            r['cstm'] = '1' if cust else '2'
            r['custom_axes'] = ';'.join(cust)
            r['n_dup_facts'] = 0
            rows[key] = r
            ctx_used.add(cref)
        col = VAL[local]
        if r.get(col, '') != '':
            if r[col] != val:
                dup += 1
                r['n_dup_facts'] += 1
            continue
        r[col] = val
    return list(rows.values()), {'contexts_used': len(ctx_used), 'dup_conflicts': dup, 'facts_after_lock': after}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--period', required=True, help='period of report, e.g. 2026-06-30')
    ap.add_argument('--filed-by', required=True, help='last filing date to include, e.g. 2026-09-30')
    ap.add_argument('--ciks', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bdc_ciks.csv'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--ua', default='', help='SEC User-Agent: "First Last email" (or env SEC_USER_AGENT)')
    ap.add_argument('--limit', type=int, default=0, help='only the first N BDCs (smoke test)')
    ap.add_argument('--cik', type=int, default=0, help='only this BDC')
    ap.add_argument('--allow-sep2026', action='store_true', help='needs the OSF receipt; see holdout_lock.py')
    a = ap.parse_args()

    maxp = check_period(a.period, a.allow_sep2026)
    run_tag = dt.date.today().isoformat()
    os.makedirs(a.out, exist_ok=True)
    sec = Sec(a.ua, os.path.join(a.out, 'cache'))
    std_lab = load_std_labels(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'std_labels.tsv'))

    with open(a.ciks, encoding='utf-8', newline='') as f:
        ciks = [r for r in csv.DictReader(f)]
    if a.cik:
        ciks = [r for r in ciks if int(r['cik']) == a.cik] or [{'cik': str(a.cik), 'name': ''}]
    if a.limit:
        ciks = ciks[:a.limit]
    print(f'{len(ciks)} BDCs; period {a.period}; filed by {a.filed_by}', flush=True)

    all_rows, manifest, missing = [], [], []
    t0 = time.time()
    for k, c in enumerate(ciks, 1):
        cik = int(c['cik'])
        try:
            sub, filings = sec.submissions(cik, run_tag)
        except NotFound:
            missing.append({'cik': cik, 'name': c.get('name', ''), 'reason': 'no submissions file'})
            continue
        sel = [f for f in filings if f.get('form') in FORMS and f.get('reportDate') == a.period and f.get('filingDate', '9') <= a.filed_by]
        sel = [f for f in sel if f['reportDate'] <= maxp]   # lock, belt and braces
        if not sel:
            missing.append({'cik': cik, 'name': c.get('name', '') or sub.get('name', ''), 'reason': 'no 10-Q/10-K for this period filed by the cut-off'})
            continue
        for f in sel:
            acc = f['accessionNumber']
            m = {'cik': cik, 'adsh': acc, 'form': f['form'], 'filed': f['filingDate'], 'accepted': f.get('acceptanceDateTime', ''),
                 'period': f['reportDate'], 'instance': '', 'instance_bytes': 0, 'contexts': 0, 'facts': 0, 'rows': 0,
                 'rows_at_period': 0, 'contexts_used': 0, 'dup_conflicts': 0, 'facts_after_lock': 0, 'name': '', 'problem': ''}
            try:
                items = sec.index(cik, acc)
                inst_name, lab_name = pick_files(items)
                if not inst_name:
                    m['problem'] = 'no XBRL instance in the filing'
                    manifest.append(m)
                    continue
                data = sec.get(sec.folder(cik, acc) + '/' + inst_name, f'filings/{cik}/{acc}/{inst_name}')
                fund_lab = parse_labels(sec.get(sec.folder(cik, acc) + '/' + lab_name, f'filings/{cik}/{acc}/{lab_name}')) if lab_name else {}
                try:
                    name = conformed_name(sec.header_text(cik, acc), cik)
                except NotFound:
                    name = None
                name = (name or sub.get('name') or '').upper()   # the SEC data sets print names in capitals
                inst = Instance(data)
                meta = {'adsh': acc, 'cik': str(cik), 'name': name, 'form': f['form'], 'filed': f['filingDate'],
                        'period': round_month_end(f['reportDate']), 'cstm': '2',
                        'inlineurl': f'https://www.sec.gov/ix?doc=/Archives/edgar/data/{cik}/{acc.replace("-", "")}/{f.get("primaryDocument", "")}'}
                rows, st = rows_from_instance(inst, meta, std_lab, fund_lab, maxp)
                all_rows += rows
                m.update(st)
                m.update({'instance': inst_name, 'instance_bytes': len(data), 'contexts': len(inst.contexts), 'facts': inst.n_facts_all,
                          'rows': len(rows), 'rows_at_period': sum(r['ddate'] == round_month_end(a.period) for r in rows), 'name': name})
                if not rows:
                    m['problem'] = 'no context with a typed InvestmentIdentifierAxis member'
            except NotFound as e:
                m['problem'] = f'not found: {e}'
            except ET.ParseError as e:
                m['problem'] = f'XML parse error: {e}'
            manifest.append(m)
        if k % 10 == 0 or k == len(ciks):
            print(f'  {k}/{len(ciks)} BDCs, {len(manifest)} filings, {len(all_rows):,} rows, {sec.n_requests} requests, '
                  f'{sec.n_bytes / 1e6:.0f} MB, {time.time() - t0:.0f}s', flush=True)

    # write soi.tsv: metadata, the four preset axes, other axes, value columns, diagnostics
    vals = list(VAL.values())
    axes = sorted({k for r in all_rows for k in r} - set(META) - set(vals) - set(EXTRA) - set(FIRST_AXES))
    cols = META + FIRST_AXES + axes + vals + EXTRA
    src = os.path.join(a.out, 'src', f'edgar_{a.period}')
    os.makedirs(src, exist_ok=True)
    with open(os.path.join(src, 'soi.tsv'), 'w', encoding='utf-8', newline='') as f:
        f.write('\t'.join(cols) + '\n')
        for r in all_rows:
            f.write('\t'.join(clean_cell(r.get(c, '')) for c in cols) + '\n')
    for fn, rows in ((f'manifest_{a.period}.csv', manifest), (f'not_found_{a.period}.csv', missing)):
        with open(os.path.join(a.out, fn), 'w', encoding='utf-8', newline='') as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    print(f'done: {len(manifest)} filings from {len({m["cik"] for m in manifest})} BDCs, {len(all_rows):,} rows '
          f'({sum(r["ddate"] == round_month_end(a.period) for r in all_rows):,} dated {a.period}); '
          f'{len(missing)} BDCs without a filing; {sum(1 for m in manifest if m["problem"])} filings with a problem; '
          f'{sec.n_requests} requests, {sec.n_bytes / 1e6:.0f} MB, {time.time() - t0:.0f}s', flush=True)
    print('output:', os.path.join(src, 'soi.tsv'))


if __name__ == '__main__':
    main()
