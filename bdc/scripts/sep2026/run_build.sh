#!/bin/bash
# usage: run_build.sh <srcdir> <outdir> [--allow-sep2026]
_ROOT=${THESIS_DATA:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}  # repo root
set -e
SRC=$1; OUT=$2; shift 2
mkdir -p $OUT
cd $_ROOT/bdc/scripts/sep2026
python3 $_ROOT/bdc/scripts/v2_load.py $SRC $OUT/raw.parquet "$@"
python3 $_ROOT/bdc/scripts/v2_clean.py $OUT/raw.parquet $OUT/debt.parquet
python3 $_ROOT/bdc/scripts/v2_pairs.py $OUT/debt.parquet $OUT 0.5
OUTV=$OUT python3 $_ROOT/bdc/scripts/sep2026/v2_stablekey.py
OUTV=$OUT python3 $_ROOT/bdc/scripts/sep2026/v2_panel.py
echo BUILD DONE
