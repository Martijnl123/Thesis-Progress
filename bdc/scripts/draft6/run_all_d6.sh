#!/bin/bash
# Draft 6 (26 Nov 2026): rebuild every number, table and figure of the draft 6 folder, then check the macros and build the PDF.
# The September builds in bdc/v2/sep2026_d6/ come from draft6/run_build_det.sh (deterministic tie-break, Carlyle added;
# see bdc/v2/sep2026_d6/DEVIATIONS-2026-11-26.md); this script reads them and does not rebuild them.
# Usage: bash run_all_d6.sh <draft6 folder> [scratch folder for the September reruns]
# Runs sequentially: v3_numbers.py rewrites two files in bdc/v2/ with identical content, so never run two copies at once.
_ROOT=${THESIS_DATA:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}  # repo root
set -euo pipefail
OUT=$1; SCR=${2:-$(mktemp -d)}
B=$_ROOT/bdc/scripts
python3 $B/v3_numbers.py "$OUT"                    # June 2026 results: numbers.tex, tables/, figures/ (draft 3 pipeline)
python3 $B/draft4/d4_pap_numbers.py "$OUT"         # numbers-pap.tex: checks listed in the plan (June data only)
python3 $B/draft6/d6_sep_numbers.py "$OUT" "$SCR"  # numbers-sep.tex: registered September statistics on the 26 Nov builds
python3 $B/draft4/d4_labels_numbers.py "$OUT"      # numbers-labels.tex: blind validation and open-model relabel
python3 $B/draft4/d4_tables.py "$OUT"              # draft 4 tables and numbers-tab.tex
python3 $B/draft6/d6_d5numbers.py "$OUT" "$SCR/d5tables"  # numbers-d5.tex on the 26 Nov builds
python3 $B/draft6/d6_numbers.py "$OUT"             # numbers-d6.tex: new June checks (rank variants, root-3, Pluralsight, trims)
python3 $B/draft6/d6_figs.py "$OUT"                # figures and numbers-d6fig.tex (iShares quarter-ends, histogram)
python3 $B/draft6/d6_tables.py "$OUT"              # draft 6 tables and numbers-d6tab.tex
python3 $B/draft6/d6_tabfix.py "$OUT"              # tab_power_d6.tex (minus signs)
python3 $B/draft6/d6_derived.py "$OUT"             # numbers-d6der.tex: word forms, unsigned forms, ratios, NAV scenario
python3 $B/draft6/d6_check_macros.py "$OUT"
# The thesis text (main.tex) is not in this repository: put it in <draft6 folder> to build the PDF and the word count.
if [ -f "$OUT/main.tex" ]; then
(cd "$OUT" && latexmk -pdf -interaction=nonstopmode main.tex >/dev/null)
python3 $B/draft5/d5_wordcount.py "$OUT/main.pdf" "$OUT"   # numbers-wc.tex
(cd "$OUT" && latexmk -pdf -interaction=nonstopmode -g main.tex >/dev/null)
fi
sha256sum "$OUT"/numbers*.tex
