"""List the BDCs whose September 2026 results were public on EDGAR on or before the PAP posting date (the early-filer list
attached to the pre-analysis plan; v4, 'The hold-out is partly public' and 'Which estimate counts').

It reads ONLY the EDGAR filing lists (form type, filing date, acceptance time, period of report, 8-K item numbers).
It downloads no filing and no document, so it opens no September 2026 mark; it is allowed while the lock is closed.

A fund is on the list (tier A) if, by the posting date, it had
  - a 10-Q or 10-K (or amendment) for the period ending 30 September 2026, or
  - an 8-K with item 2.02 (Results of Operations and Financial Condition) filed on or after 1 October 2026.
Tier B lists 8-Ks filed on or after 1 October 2026 with item 7.01 or 8.01 but not 2.02. Non-traded BDCs use these for
monthly NAV and distribution notices, which can reveal a September NAV but not loan marks. Tier B funds (not in tier A)
stay in the confirmatory estimate; a robustness estimate also drops them (decided 7 Nov 2026, plan v5).
Earnings releases that never reach EDGAR cannot be seen here; the plan says so.

Output: early_filers_<posting date>.csv (one line per qualifying filing) and a count by fund, cohort funds marked.
Usage: python edgar_early_filers.py --posting-date 2026-11-08 --out OUT --ua "Name email"
Standard library only.
"""
import argparse, csv, datetime as dt, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edgar_common import Sec, NotFound

PERIODIC = {'10-Q', '10-K', '10-Q/A', '10-K/A', '10-QT', '10-KT'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--posting-date', required=True, help='date the plan is posted on OSF, YYYY-MM-DD')
    ap.add_argument('--ciks', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bdc_ciks.csv'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--ua', default='')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sec = Sec(a.ua, os.path.join(a.out, 'cache'))
    run_tag = dt.datetime.now().strftime('%Y-%m-%dT%H%M')
    with open(a.ciks, encoding='utf-8', newline='') as f:
        ciks = list(csv.DictReader(f))
    hits = []
    for c in ciks:
        try:
            sub, filings = sec.submissions(int(c['cik']), run_tag)
        except NotFound:
            continue
        for f in filings:
            fd = f.get('filingDate', '')
            if fd > a.posting_date:
                continue
            form = f.get('form', '')
            if form in PERIODIC and f.get('reportDate') == '2026-09-30':
                why = 'periodic report for 30 Sep 2026'
            elif form in ('8-K', '8-K/A') and fd >= '2026-10-01' and '2.02' in (f.get('items') or ''):
                why = 'A: 8-K item 2.02 (results)'
            elif form in ('8-K', '8-K/A') and fd >= '2026-10-01' and any(i in (f.get('items') or '') for i in ('7.01', '8.01')):
                why = 'B: 8-K item 7.01/8.01 (may hold a NAV)'
            else:
                continue
            why = why if why.startswith(('A:', 'B:')) else 'A: ' + why
            hits.append({'cik': c['cik'], 'name': c.get('name', ''), 'held_sep25': c.get('held_sep25', ''), 'form': form,
                         'filing_date': fd, 'accepted': f.get('acceptanceDateTime', ''), 'report_date': f.get('reportDate', ''),
                         'items': f.get('items', ''), 'accession': f.get('accessionNumber', ''), 'reason': why})
    fn = os.path.join(a.out, f'early_filers_{a.posting_date}.csv')
    with open(fn, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['cik', 'name', 'held_sep25', 'form', 'filing_date', 'accepted', 'report_date', 'items', 'accession', 'reason'])
        w.writeheader()
        w.writerows(sorted(hits, key=lambda h: (h['filing_date'], h['name'])))
    for tier in ('A', 'B'):
        hs = [h for h in hits if h['reason'].startswith(tier)]
        fa = {h['cik'] for h in hits if h['reason'].startswith('A')}
        funds = {h['cik'] for h in hs} - (fa if tier == 'B' else set())
        cohort = {h['cik'] for h in hs if str(h['held_sep25']).lower() == 'true'} - (fa if tier == 'B' else set())
        print(f'tier {tier}{" (not already in A)" if tier == "B" else ""}: {len(funds)} of {len(ciks)} BDCs as of {a.posting_date}, '
              f'{len(cohort)} of them held cohort positions in Sep 2025; {len(hs)} filings')
    print(f'File: {fn}. Compiled {dt.datetime.now().isoformat(timespec="minutes")} local time.')


if __name__ == '__main__':
    main()
