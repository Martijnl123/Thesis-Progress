#!/bin/bash
# Draft 4 (23 Nov 2026): rebuild every number, table and figure of a draft folder from the data, then check the macros.
# Usage: bash run_all.sh <out folder> [scratch folder for the September reruns]
# Runs sequentially: v3_numbers.py rewrites two files in bdc/v2/ with identical content, so never run two copies at once.
_ROOT=${THESIS_DATA:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}  # repo root
set -euo pipefail
OUT=$1; SCR=${2:-$(mktemp -d)}
B=$_ROOT/bdc/scripts
mkdir -p "$OUT"
python3 $B/v3_numbers.py "$OUT"                 # June 2026 results: numbers.tex, numbers-log.txt, tables/, figures/ (draft 3 pipeline, unchanged)
python3 $B/draft4/d4_pap_numbers.py "$OUT"      # numbers-pap.tex: checks listed in PAP v5 (June data only)
python3 $B/draft4/d4_sep_numbers.py "$OUT" "$SCR"  # numbers-sep.tex: registered September 2026 statistics (+ \ifnport)
python3 $B/draft4/d4_labels_numbers.py "$OUT"   # numbers-labels.tex: blind validation and open-model relabel
python3 $B/draft4/d4_tables.py "$OUT"           # draft 4 tables and numbers-tab.tex
python3 $B/draft4/d4_check_macros.py "$OUT"     # needs the .tex chapters in OUT; skip this line when OUT holds numbers only
sha256sum "$OUT"/numbers*.tex
