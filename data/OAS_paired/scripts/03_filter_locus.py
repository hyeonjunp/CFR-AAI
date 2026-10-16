#!/usr/bin/env python
"""이미 만들어진 processed/*.csv.gz 에서 locus 오배정 행을 제거한다.

02_extract_cdr.py 를 처음부터 다시 돌리면 동일한 결과가 나오지만(패치 반영됨),
40분 걸리므로 산출물에 필터만 다시 먹인다.
"""
import os, gzip, shutil
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "processed")
CHUNK = 200000

for which, want in (("heavy", {"H"}), ("light", {"K", "L"})):
    src = os.path.join(ROOT, which, which + ".csv.gz")
    dst = src + ".tmp"
    kept = dropped = 0
    with gzip.open(dst, "wt", newline="") as out:
        first = True
        for ch in pd.read_csv(src, keep_default_na=False, na_values=[],
                              low_memory=False, chunksize=CHUNK):
            ok = ch["locus"].astype(str).str.upper().isin(want)
            dropped += int((~ok).sum()); kept += int(ok.sum())
            ch[ok].to_csv(out, index=False, header=first)
            first = False
    shutil.move(dst, src)
    print("%-7s kept=%d dropped=%d" % (which, kept, dropped))

# paired: 양쪽 locus 가 모두 정상인 행만
src = os.path.join(ROOT, "paired", "paired.csv.gz")
dst = src + ".tmp"
kept = dropped = 0
with gzip.open(dst, "wt", newline="") as out:
    first = True
    for ch in pd.read_csv(src, keep_default_na=False, na_values=[],
                          low_memory=False, chunksize=CHUNK):
        ok = (ch["heavy_locus"].astype(str).str.upper() == "H") & \
             (ch["light_locus"].astype(str).str.upper().isin(["K", "L"]))
        dropped += int((~ok).sum()); kept += int(ok.sum())
        ch[ok].to_csv(out, index=False, header=first)
        first = False
shutil.move(dst, src)
print("paired  kept=%d dropped=%d" % (kept, dropped))
