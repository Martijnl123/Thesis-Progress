#!/bin/bash
# Draft 6 (26 Nov 2026): September build with the deterministic tie-break in the position key.
# Same chain as sep2026/run_build.sh (v2_load -> [cik_successor_map] -> v2_clean -> v2_pairs 0.5 -> v2_stablekey -> v2_panel),
# with draft6/v2_stablekey_det.py in place of the stable key. Writes only into <out>, never into bdc/v2.
# Usage: CIKMAP=1 bash run_build_det.sh <src dir with */soi.tsv> <out dir>
_ROOT=${THESIS_DATA:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}  # repo root
set -euo pipefail
SRC=$1; OUT=$2; B=$_ROOT/bdc/scripts
export PYTHONHASHSEED=0
mkdir -p $OUT
python3 $B/v2_load.py $SRC $OUT/raw.parquet --allow-sep2026
if [ -n "${CIKMAP:-}" ]; then python3 $_ROOT/bdc/scripts/draft6/cik_successor_map.py $OUT/raw.parquet; fi
python3 $B/v2_clean.py $OUT/raw.parquet $OUT/debt.parquet
python3 $B/v2_pairs.py $OUT/debt.parquet $OUT 0.5
OUTV=$OUT python3 $B/draft6/v2_stablekey_det.py
OUTV=$OUT python3 $B/draft6/v2_panel_sep.py
echo BUILD DONE $OUT
