#!/usr/bin/env bash
# OAS Paired 전체 파이프라인: 다운로드 → heavy/light 분리 + IMGT CDR 추출 → 90% 클러스터링 + 5-fold split → 리포트
#   scripts/run_all.sh [다운로드 병렬 수]       (재실행 가능: 받은 파일/만든 산출물은 건너뜀)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY=${PY:-python}
JOBS=${1:-10}

echo "[1/4] 다운로드 (610 units, 3.3GB)"
"$ROOT/scripts/01_download.sh" "$JOBS"
n=$(find "$ROOT/raw" -name '*.csv.gz' | wc -l)
[ "$n" -ge "$(wc -l < "$ROOT/scripts/paired_urls.txt")" ] || { echo "다운로드 누락($n개): 다시 실행하세요. download_fail.log 참고"; exit 1; }

echo "[2/4] CDR 추출"
if [ ! -f "$ROOT/processed/heavy/heavy.csv.gz" ] || [ ! -f "$ROOT/processed/light/light.csv.gz" ]; then
  "$PY" "$ROOT/scripts/02_extract_cdr.py" --raw "$ROOT/raw" --out "$ROOT/processed"
fi

echo "[3/4] cd-hit 90% 클러스터링 + 5-fold split"
for chain in heavy light; do
  [ -f "$ROOT/dataset/$chain/fold_4/test.csv.gz" ] || "$PY" "$ROOT/scripts/04_cluster_split.py" --chain "$chain"
done

echo "[4/4] 리포트"
"$PY" "$ROOT/scripts/05_report.py" --dataset
