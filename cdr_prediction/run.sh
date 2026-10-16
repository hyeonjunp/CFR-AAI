#!/bin/bash
# ABlang + AbLangStyleLayer CDR 예측 — OAS_paired 학습
#   ./run.sh                 # chain=both(heavy+light), fold 0
#   ./run.sh light 0         # light, fold 0
#   ./run.sh both 0 --resume # heavy+light, fold 0, 마지막 checkpoint 부터 재개
#   ./run.sh heavy 0         # heavy 만
#   ./run.sh heavy all       # fold 0~4 전부 (5-fold CV)
set -uo pipefail
PY=${PY:-python}
D="$(cd "$(dirname "$0")" && pwd)"

CHAIN=${1:-both}
FOLD=${2:-0}
[ $# -ge 1 ] && shift
[ $# -ge 1 ] && shift

ARGS=(--config config.yaml --chain "$CHAIN")
[ "$FOLD" != "all" ] && ARGS+=(--fold "$FOLD")
ARGS+=("$@")

mkdir -p "$D/logs"
echo "[$(date '+%F %T')] START ${ARGS[*]}"
( cd "$D" && "$PY" train.py "${ARGS[@]}" ) 2>&1 | tee "$D/logs/${CHAIN}_fold${FOLD}.log"
echo "[$(date '+%F %T')] DONE (exit ${PIPESTATUS[0]})"
