# EDGAR pull for the September 2026 quarter (and its June 2026 test)

Written 7 Nov 2026. The SEC BDC and N-PORT data sets that would hold the 30 September 2026 schedules come out after the
thesis cut-off, so the pre-analysis plan (v4, "Source of the September 2026 BDC data", "Lock", "N-PORT") pulls them
filing by filing from EDGAR. sec.gov can't be reached from the cloud, so the pulls run on a local PC. Before they are
used, they must reproduce the 30 June 2026 data.

## Files

| File | What it does |
|---|---|
| `holdout_lock.py` | The lock. Periods after 30 June 2026 need an `OSF-POSTED.txt` receipt (OSF URL + posting date) **and** `--allow-sep2026`. Nothing after 30 Sep 2026, ever. |
| `edgar_common.py` | SEC client: User-Agent required, 8 requests per second at most, retries, gzipped cache of every download. |
| `edgar_pull_bdc.py` | 10-Q/10-K XBRL instances to `soi.tsv` in the SEC data-set layout, so `v2_load.py` and `v2_clean.py` run unchanged. |
| `edgar_pull_nport.py` | NPORT-P `primary_doc.xml` to the loan-extract layout and the `nport_qe_loans` aggregate. |
| `edgar_early_filers.py` | The early-filer list for the posting date. Reads filing lists only, downloads no filing. |
| `edgar_compare_june.py` | Cloud side: the June test against `v2/debt.parquet` (same June rows as `positions.parquet`) and `v2/nport_qe_loans.parquet`. |
| `bdc_ciks.csv` | The 166 BDCs with positions on or after 30 Sep 2025 (160 with June 2026 rows, 151 held positions in Sep 2025). |
| `nport_series.csv` | 596 registrants and 1,132 series or no-series funds with loan holdings in the N-PORT data sets 2025q3 to 2026q2, plus the 45 funds with loans in the 30 June 2026 rows of `loans_2026q3` (column `in_jun26`; added 7 Nov after the June test). |
| `std_labels.tsv` | Labels of the standard XBRL axes and members, from the `tag.tsv` files of the BDC data sets. |

The loader `../v2_load.py` now stops before writing if a row is dated 30 September 2026 or comes from a filing for a
period after 30 June 2026 (`--allow-sep2026` plus the receipt opens it). `../v2_clean.py` only changed so that its
closing comparison with a pre-audit file is skipped when that file is missing; the cleaning rules are unchanged.

## Checks already done in the cloud (7 Nov)

- **Parser against the SEC's own pivot.** Four June 2026 10-Qs (Golub Capital BDC, Bain Capital Private Credit,
  Blackstone Private Credit Fund, Audax Private Credit Fund) were rebuilt as XBRL instances from the data-set facts and
  parsed. Three match the SEC `soi.tsv` rows exactly on every column the loader reads (3,591, 901 and 3,967 rows). Audax
  matches on all columns the cleaning uses, and differs only in two enumeration columns: the SEC drops values that point
  to the fund's own members, the pull keeps them. The cleaning doesn't read those columns.
- **Filing choice.** Taking only 10-Q/10-K filings whose period of report is 30 June 2026 and that were filed by
  30 September reproduces all 40,909 June 2026 debt positions in 160 funds, with the same filing chosen for every
  fund (simulated with the data-set rows). So the rule the pull uses loses nothing.
- **Lock.** A synthetic 30 Sep 2026 row stops the loader; the flag without a receipt stops it; a test receipt outside
  the project plus the flag opens it. The pulls refuse `--period 2026-09-30` while the lock is closed.
- **Mock runs** of both pulls from a pre-filled cache (no network) produce the expected files.

What the cloud can't check is real EDGAR files: their exact XML, typed-member text and headers. That is the June test.

## June test on the PC (Remote Control, folder `Downloads\2026_03_bdc`)

```
cd %USERPROFILE%\Downloads\2026_03_bdc\edgar_pull
set SEC_USER_AGENT=<name> <email>
python edgar_pull_bdc.py --period 2026-06-30 --filed-by 2026-09-30 --out ..\edgar_out\2026-06-30\bdc --limit 3
python edgar_pull_bdc.py --period 2026-06-30 --filed-by 2026-09-30 --out ..\edgar_out\2026-06-30\bdc
python edgar_pull_nport.py --period 2026-06-30 --filed-by 2026-09-30 --out ..\edgar_out\2026-06-30\nport --limit-registrants 5
python edgar_pull_nport.py --period 2026-06-30 --filed-by 2026-09-30 --out ..\edgar_out\2026-06-30\nport
```

The first line of each pair is a smoke test; reruns reuse the cache. Then copy back to the project, into
`thesis/2-data-and-code/raw-sec/edgar-pull/2026-06-30/`: `bdc/src/edgar_2026-06-30/soi.tsv`,
`bdc/manifest_2026-06-30.csv`, `bdc/not_found_2026-06-30.csv`, `nport/nport_qe_loans_2026-06-30.csv.gz`,
`nport/nport_loans_2026-06-30.csv.gz`, `nport/nport_manifest_2026-06-30.csv` (not the cache). In the cloud:

```
python3 edgar_compare_june.py --pull /mnt/project-files/thesis/2-data-and-code/raw-sec/edgar-pull/2026-06-30 \
        --out /mnt/project-files/thesis/2-data-and-code/bdc/v2/results_edgar_june_test_<date>.txt
```

**Pass rules, fixed before the test:** BDC at least 99.5% of June positions found both ways on fund, description,
principal and fair value, 99.5% of principal, and 99.5% agreement on spread, coupon, maturity and the software,
first-lien, PIK and non-accrual flags. N-PORT at least 98% of June Level 2 lines found and 98% with the same median
price. A failure is reported, the pull is fixed and the test rerun; every run is kept. The script and the test result
go on OSF as a dated amendment before 20 November (plan v4).

## June test result (7 Nov 2026)

The PC run (lock closed throughout, scripts unchanged) found a 10-Q/10-K for all 160 BDCs with June 2026 rows, with
0 problem filings, and kept 652 NPORT-P filings (700 in the N-PORT rerun of run 4). Files: `raw-sec/edgar-pull/2026-06-30/` (run-1 N-PORT files in `nport_run1/`). Four comparison runs, all
kept in `bdc/v2/results_edgar_june_test_2026-11-07*.txt`:

| Run | Change | BDC | N-PORT |
|---|---|---|---|
| 1 | pull as written | FAIL: 95.3% of positions | FAIL: 97.98% of Level 2 lines (inflated, see run 3) |
| 2 | BDC through `edgar_sec_compat.py`: 4-decimal rounding, descriptions cut at 255 characters, non-ASCII dropped, holdings with the same cut description merged; N-PORT counts holding lines | FAIL: 99.46% | FAIL |
| 3 | accents transliterated instead of dropped; each N-PORT line counted once | **PASS**: 99.51% both ways, 99.81% of principal, fields 99.99-100% | **FAIL**: 97.60% of Level 2 lines; median price equal on 99.48% of lines found |
| 4 | N-PORT re-pulled with the 45 missing funds added to `nport_series.csv` (BDC files unchanged) | **PASS** (same as run 3) | **PASS**: 99.63% of Level 2 lines, median price equal on 99.97% |

What the BDC result means: the September pull has to go through `edgar_sec_compat.py` so the new quarter has the same
conventions as the history. The BDC pass is narrow (99.51% against 99.5%); the remaining 0.5% are descriptions the
SEC wrapped in quotes or cut one character off at the start, which we couldn't reproduce.

What the N-PORT result means: the parser was right from the start, the fund list was not. Comparing holding by holding
with the 30 June 2026 rows of `loans_2026q3.csv.gz`  showed that run 1
missed 45 funds that first held loans by June (16 registrants and 29 series of registrants already listed; 760 Level 2
holdings). With them added, the rerun keeps the same 675 filings as the data set, with the same number of loan holdings
in each of the 658 funds (480,935), and balance and value equal on 100% of holdings filing by filing. The 0.37% of
Level 2 lines still not found come from the aggregation rules, not the pull: the script that built `nport_qe_loans`
is lost, and our rules rebuild 99.73% of its lines even from the data-set rows. The fund list now covers every fund with loans on 30 June 2026.
Funds that first hold loans in September are still left out, which C2 doesn't need since it compares with a June price.

**Data-set defect found on the way.** When two holdings of a fund share the first 255 characters of their description,
the SEC data sets turn them into one row, taking each value from the first holding that has it, so principal and fair
value can come from different loans (example: Goldman Sachs BDC, BCPE HIPH Parent, principal 16.18m from one tranche
and fair value 9.07m from another, a "mark" of 56 for two loans marked at 100). Measured exactly with the pull: 18 of
32,467 clean positions in June 2026 and 50 of 30,926 in December 2025, 2 of them software. Descriptions of 255
characters, the precondition, are 1.8-3.5% of clean positions in other quarters, but 5.7% in March 2026, which can't
be checked without pulling the March filings. Minor for H1; a robustness check that drops 255-character descriptions
is cheap.

## September 2026, only after the plan is posted

1. Save the OSF receipt as `OSF-POSTED.txt` (one line with the `https://osf.io/...` URL and the posting date) in
   `thesis/3-writing/pre-analysis-plan/` and next to these scripts on the PC.
2. On the posting date: `python edgar_early_filers.py --posting-date <date> --out ..\edgar_out\early-filers`, and attach
   the list to the plan. Tier A (a 10-Q/10-K for 30 Sep 2026, or an 8-K with item 2.02 "results") is the list the plan
   describes. Tier B (8-Ks with items 7.01 or 8.01 only) catches the monthly NAV and distribution notices of non-traded
   BDCs, which can show a September NAV but no loan marks. **Decided 7 Nov:** tier B funds stay in the confirmatory
   estimate, and a robustness estimate also drops them (plan v5). Attach both lists to the plan.
3. After the cut-off: `python edgar_pull_bdc.py --period 2026-09-30 --filed-by 2026-11-20 --out ..\edgar_out\2026-09-30\bdc --allow-sep2026`,
   then in the cloud `python3 edgar_sec_compat.py <bdc/src> <bdc/src_sec>` and load `src_sec`
   and, by 30 November, `python edgar_pull_nport.py --period 2026-09-30 --filed-by 2026-11-30 --out ..\edgar_out\2026-09-30\nport --allow-sep2026`.
4. In the cloud, `v2_load.py` needs `--allow-sep2026` for the September files.
