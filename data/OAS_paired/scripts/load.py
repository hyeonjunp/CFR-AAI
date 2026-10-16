"""OAS Paired 처리 결과 로더.

주의: 아미노산 서열 "NA"(Asn-Ala) 가 pandas 기본 설정에서 결측치로 변환된다.
      반드시 keep_default_na=False 로 읽어야 한다. 이 헬퍼가 그걸 처리한다.
"""
import os
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "processed")


def load_oas(which="heavy", **kw):
    """which: 'heavy' | 'light' | 'paired'"""
    path = os.path.join(ROOT, which, which + ".csv.gz")
    return pd.read_csv(path, keep_default_na=False, na_values=[], low_memory=False, **kw)


if __name__ == "__main__":
    for w in ("heavy", "light", "paired"):
        d = load_oas(w, nrows=5)
        print(w, d.shape, list(d.columns)[:8], "...")
