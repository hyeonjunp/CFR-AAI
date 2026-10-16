#!/usr/bin/env bash
# OAS Paired 전체 다운로드 (610 units / 21 studies, 압축 3.3GB)
# URL 목록은 https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/ 디렉토리 인덱스를 크롤링해 생성함
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/raw"
URLS="$ROOT/scripts/paired_urls.txt"
JOBS="${1:-8}"

mkdir -p "$RAW"
# 스터디 이름을 디렉토리로 유지: .../paired/<Study>/<csv_dir>/<file>.csv.gz -> raw/<Study>/<file>.csv.gz
cat "$URLS" | xargs -P "$JOBS" -I{} bash -c '
  url="{}"
  study=$(echo "$url" | sed -E "s#.*/paired/([^/]+)/.*#\1#")
  fn=$(basename "$url")
  out="'"$RAW"'/$study/$fn"
  mkdir -p "$(dirname "$out")"
  if [ -s "$out" ]; then exit 0; fi
  curl -sfL --retry 5 --retry-delay 3 --max-time 1800 -o "$out.part" "$url" && mv "$out.part" "$out" || echo "FAIL $url" >> "'"$ROOT"'/download_fail.log"
'
echo "downloaded: $(find "$RAW" -name '*.csv.gz' | wc -l) / $(wc -l < "$URLS")"
