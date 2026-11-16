"""Shared helpers for the EDGAR pulls: a polite SEC client with a disk cache, and date helpers.

SEC fair-access rules: a User-Agent with a name and an email, and at most 10 requests per second.
This client waits at least 0.125 s between requests (8 per second) and backs off on 429/5xx.
Standard library only.
"""
import calendar, datetime as dt, gzip, json, os, random, sys, time, urllib.error, urllib.request, zlib


class NotFound(Exception):
    pass


class Sec:
    def __init__(self, ua, cache_dir, min_interval=0.125, log=None):
        ua = ua or os.environ.get('SEC_USER_AGENT', '')
        if '@' not in ua or len(ua.split()) < 2:
            sys.exit('Set the SEC User-Agent to a name and an email, e.g. --ua "First Last name@example.com" '
                     '(or the environment variable SEC_USER_AGENT). The SEC blocks requests without one.')
        self.ua, self.cache, self.min_interval, self.log = ua, cache_dir, min_interval, log
        self.last = 0.0
        self.n_requests = 0
        self.n_bytes = 0
        os.makedirs(cache_dir, exist_ok=True)

    def _wait(self):
        d = time.time() - self.last
        if d < self.min_interval:
            time.sleep(self.min_interval - d)
        self.last = time.time()

    def get(self, url, cache_rel=None):
        """GET url and return bytes. With cache_rel, the body is kept gzipped under the cache folder and reused."""
        path = os.path.join(self.cache, cache_rel + '.gz') if cache_rel else None
        if path and os.path.isfile(path):
            with gzip.open(path, 'rb') as f:
                return f.read()
        err = None
        for attempt in range(7):
            self._wait()
            req = urllib.request.Request(url, headers={'User-Agent': self.ua, 'Accept-Encoding': 'gzip, deflate'})
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    data = r.read()
                    enc = (r.headers.get('Content-Encoding') or '').lower()
                if enc == 'gzip':
                    data = gzip.decompress(data)
                elif enc == 'deflate':
                    data = zlib.decompress(data)
                self.n_requests += 1
                self.n_bytes += len(data)
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    raise NotFound(url)
                if e.code == 403:
                    sys.exit(f'SEC returned 403 Forbidden for {url}. Check the User-Agent; if it is set, wait 10 minutes '
                             '(the SEC blocks clients that went over 10 requests per second) and rerun: finished files are cached.')
                err = e
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                err = e
            time.sleep(min(60, 2 ** attempt + random.random()))
        else:
            raise RuntimeError(f'giving up on {url}: {err}')
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + '.part'
            with gzip.open(tmp, 'wb') as f:
                f.write(data)
            os.replace(tmp, path)
        return data

    def get_json(self, url, cache_rel=None):
        return json.loads(self.get(url, cache_rel).decode('utf-8'))

    def submissions(self, cik, run_tag):
        """Filing list of one registrant (data.sec.gov). Cached per run, because new filings keep arriving."""
        cik = int(cik)
        j = self.get_json(f'https://data.sec.gov/submissions/CIK{cik:010d}.json', f'submissions/{run_tag}/CIK{cik:010d}.json')
        rec = j.get('filings', {}).get('recent', {})
        keys = list(rec.keys())
        rows = [dict(zip(keys, vals)) for vals in zip(*[rec[k] for k in keys])] if keys else []
        return j, rows

    def folder(self, cik, acc):
        return f'https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace("-", "")}'

    def index(self, cik, acc):
        j = self.get_json(self.folder(cik, acc) + '/index.json', f'filings/{int(cik)}/{acc}/index.json')
        return j.get('directory', {}).get('item', [])

    def header_text(self, cik, acc):
        b = self.get(self.folder(cik, acc) + f'/{acc}-index-headers.html', f'filings/{int(cik)}/{acc}/index-headers.html')
        return b.decode('utf-8', errors='replace')


def month_end(y, m):
    return dt.date(y, m, calendar.monthrange(y, m)[1])


def round_month_end(s):
    """'2026-06-30' -> '2026-06-30'; '2026-07-01' -> '2026-06-30'; '2026-06-20' -> '2026-06-30' (nearest month-end)."""
    d = dt.date.fromisoformat(s[:10])
    this_end = month_end(d.year, d.month)
    prev_end = d.replace(day=1) - dt.timedelta(days=1)
    return (this_end if (this_end - d) <= (d - prev_end) else prev_end).isoformat()


def quarters_between(start, end):
    """Rounded number of quarters in a duration context (SEC data-set convention: 0 means an instant)."""
    a = dt.date.fromisoformat(start[:10])
    b = dt.date.fromisoformat(end[:10])
    return int(round(((b - a).days + 1) / (365.25 / 4)))


MON = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']


def to_dmy(s):
    """'2026-06-30' -> '30-JUN-2026' (the date format of the SEC N-PORT data sets). Other strings pass through."""
    if not s or len(s) < 10 or s[4] != '-':
        return s or ''
    try:
        d = dt.date.fromisoformat(s[:10])
    except ValueError:
        return s
    return f'{d.day:02d}-{MON[d.month - 1]}-{d.year}'


def clean_cell(v):
    """No tabs or line breaks inside a TSV/CSV cell."""
    if v is None:
        return ''
    return str(v).replace('\t', ' ').replace('\r', ' ').replace('\n', ' ')
