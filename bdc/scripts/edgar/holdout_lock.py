"""Hold-out lock for the September 2026 quarter (pre-analysis plan v4, section 'Lock').

Until the plan is posted on OSF, no script in this project may download or load data for a period after
30 June 2026. The lock opens only when BOTH of these hold:
  1. a receipt file OSF-POSTED.txt exists that contains the OSF registration URL (https://osf.io/...) and the
     posting date (YYYY-MM-DD). It is looked for in, in this order: the path in the environment variable
     THESIS_OSF_RECEIPT; the folder of this script; thesis/3-writing/pre-analysis-plan/ in the project folder.
  2. the caller passes the explicit flag --allow-sep2026.
Even when the lock is open, nothing dated after 30 September 2026 is allowed: the plan does not cover it.

Standard library only, so the same file runs on marty's PC and in the cloud.
"""
import os, re, sys

LOCKED_MAX = '2026-06-30'
OPEN_MAX = '2026-09-30'
PROJECT_RECEIPT = '/mnt/project-files/thesis/3-writing/pre-analysis-plan/OSF-POSTED.txt'


def _receipt_paths():
    here = os.path.dirname(os.path.abspath(__file__))
    out = []
    if os.environ.get('THESIS_OSF_RECEIPT'):
        out.append(os.environ['THESIS_OSF_RECEIPT'])
    out += [os.path.join(here, 'OSF-POSTED.txt'), PROJECT_RECEIPT]
    return out


def receipt():
    """Return (path, url, date) of a valid receipt, or None."""
    for p in _receipt_paths():
        if not os.path.isfile(p):
            continue
        txt = open(p, encoding='utf-8', errors='replace').read()
        url = re.search(r'https://osf\.io/\S+', txt)
        date = re.search(r'\b20\d\d-\d\d-\d\d\b', txt)
        if url and date:
            return p, url.group(0), date.group(0)
    return None


def max_period(allow_flag, quiet=False):
    """Latest period end (YYYY-MM-DD) that may be downloaded or loaded in this run."""
    r = receipt()
    if allow_flag and r is None:
        sys.exit('LOCK: --allow-sep2026 was given, but no OSF-POSTED.txt with an osf.io URL and a posting date was found '
                 f'(looked in: {", ".join(_receipt_paths())}). Post the plan first. Nothing was downloaded or loaded.')
    if allow_flag and r is not None:
        if not quiet:
            print(f'LOCK OPEN: plan posted at {r[1]} on {r[2]} (receipt {r[0]}); periods up to {OPEN_MAX} allowed.', flush=True)
        return OPEN_MAX
    if not quiet:
        print(f'LOCK CLOSED: periods up to {LOCKED_MAX} only.', flush=True)
    return LOCKED_MAX


def check_period(period, allow_flag):
    """Stop the script if `period` is after what the lock allows."""
    m = max_period(allow_flag)
    if period > m:
        sys.exit(f'LOCK: period {period} is after {m}. Nothing was downloaded. '
                 + ('The plan does not cover it.' if m == OPEN_MAX else
                    'September 2026 needs the OSF receipt and --allow-sep2026 (see README-edgar-pull.md).'))
    return m


def is_quarter_end(ddate):
    return ddate[5:] in ('03-31', '06-30', '09-30', '12-31')
