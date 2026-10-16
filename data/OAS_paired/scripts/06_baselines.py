#!/usr/bin/env python
"""두 태스크의 trivial baseline. 모델 점수를 해석하려면 하한선을 알아야 한다.

  A. VH/VL 분류      - 길이 단독 / 아미노산 조성 로지스틱 회귀
  B. CDR 경계 예측   - 위치별 다수결 (서열 내용을 전혀 안 봄)

클러스터 단위로 나눈 fold_0 의 train/test 를 그대로 쓴다 (누수 없음).
"""
import os, argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
READ = dict(keep_default_na=False, na_values=[], low_memory=False)
AA = "ACDEFGHIKLMNPQRSTVWY"


def load_fold(chain, split, n, seed=0):
    p = os.path.join(ROOT, "dataset", chain, "fold_0", split + ".csv.gz")
    d = pd.read_csv(p, usecols=["sequence", "mask", "seq_len",
                                "CDR1_len", "CDR2_len", "CDR3_len"], **READ)
    if n and len(d) > n:
        d = d.sample(n, random_state=seed).reset_index(drop=True)
    return d


def aa_composition(seqs):
    idx = {a: i for i, a in enumerate(AA)}
    X = np.zeros((len(seqs), len(AA)), dtype=np.float32)
    for r, s in enumerate(seqs):
        for ch in s:
            j = idx.get(ch)
            if j is not None:
                X[r, j] += 1.0
        X[r] /= max(len(s), 1)
    return X


def task_vh_vl(n_train, n_test):
    print("\n" + "=" * 62)
    print("A. VH / VL 분류")
    print("=" * 62)
    tr = pd.concat([load_fold("heavy", "train", n_train).assign(y=1),
                    load_fold("light", "train", n_train).assign(y=0)], ignore_index=True)
    te = pd.concat([load_fold("heavy", "test", n_test).assign(y=1),
                    load_fold("light", "test", n_test).assign(y=0)], ignore_index=True)
    print("  train %d / test %d" % (len(tr), len(te)))

    # 1) 길이 단독: train 에서 최적 임계값 하나만 고름
    lens = np.sort(tr.seq_len.unique())
    best = max(((t, accuracy_score(tr.y, (tr.seq_len >= t).astype(int))) for t in lens),
               key=lambda x: x[1])
    thr = best[0]
    acc = accuracy_score(te.y, (te.seq_len >= thr).astype(int))
    print("  [길이 단독]        임계값 seq_len >= %d  ->  test 정확도 %.4f" % (thr, acc))

    # 2) 아미노산 조성 20차원 로지스틱 회귀 (서열 순서 정보 없음)
    clf = LogisticRegression(max_iter=2000, n_jobs=-1)
    clf.fit(aa_composition(tr.sequence.values), tr.y.values)
    pred = clf.predict(aa_composition(te.sequence.values))
    print("  [AA 조성 20차원]   test 정확도 %.4f  F1 %.4f"
          % (accuracy_score(te.y, pred), f1_score(te.y, pred)))
    print("\n  길이 분포: heavy %.1f±%.1f / light %.1f±%.1f"
          % (tr[tr.y == 1].seq_len.mean(), tr[tr.y == 1].seq_len.std(),
             tr[tr.y == 0].seq_len.mean(), tr[tr.y == 0].seq_len.std()))
    # 임계값 하나로 갈랐을 때 어느 쪽이 얼마나 틀리는지 (범위 겹침이 아니라 질량 기준)
    fp = ((te.y == 0) & (te.seq_len >= thr)).sum()
    fn = ((te.y == 1) & (te.seq_len < thr)).sum()
    print("  임계값 %d 오분류: heavy->light %d/%d (%.2f%%),  light->heavy %d/%d (%.2f%%)"
          % (thr, fn, (te.y == 1).sum(), 100.0 * fn / (te.y == 1).sum(),
             fp, (te.y == 0).sum(), 100.0 * fp / (te.y == 0).sum()))
    lo = max(te[te.y == 1].seq_len.min(), te[te.y == 0].seq_len.min())
    hi = min(te[te.y == 1].seq_len.max(), te[te.y == 0].seq_len.max())
    print("  길이 범위는 크게 겹침: 겹치는 구간 [%d, %d] 에 heavy %.1f%% / light %.1f%% 가 들어감"
          % (lo, hi,
             100.0 * ((te.y == 1) & te.seq_len.between(lo, hi)).sum() / (te.y == 1).sum(),
             100.0 * ((te.y == 0) & te.seq_len.between(lo, hi)).sum() / (te.y == 0).sum()))


def positional_majority(train, test, chain):
    """위치 인덱스별로 train 다수결 라벨을 그대로 예측. 서열 내용은 안 봄."""
    L = int(max(train.seq_len.max(), test.seq_len.max()))
    ones = np.zeros(L, dtype=np.int64)
    tot = np.zeros(L, dtype=np.int64)
    for m in train["mask"].values:
        arr = np.frombuffer(m.encode(), dtype=np.uint8) - ord("0")
        ones[:len(arr)] += arr
        tot[:len(arr)] += 1
    prior = np.where(tot > 0, ones / np.maximum(tot, 1), 0.0)
    rule = (prior >= 0.5).astype(np.uint8)

    y_true, y_pred = [], []
    exact = np.zeros(3, dtype=np.int64)
    n = 0
    for m in test["mask"].values:
        arr = np.frombuffer(m.encode(), dtype=np.uint8) - ord("0")
        p = rule[:len(arr)]
        y_true.append(arr); y_pred.append(p)
        # CDR 별 스팬 완전일치 여부
        import re
        t_sp = [x.span() for x in re.finditer(r"1+", m)]
        p_sp = [x.span() for x in re.finditer(r"1+", "".join(map(str, p)))]
        for k in range(3):
            if k < len(p_sp) and k < len(t_sp) and p_sp[k] == t_sp[k]:
                exact[k] += 1
        n += 1
    y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
    print("  [%s] 토큰 정확도 %.4f   CDR토큰 F1 %.4f"
          % (chain, accuracy_score(y_true, y_pred), f1_score(y_true, y_pred)))
    print("       스팬 완전일치:  CDR1 %.4f   CDR2 %.4f   CDR3 %.4f"
          % (exact[0] / n, exact[1] / n, exact[2] / n))


def task_cdr(n_train, n_test):
    print("\n" + "=" * 62)
    print("B. CDR 경계 예측 — 위치별 다수결 (서열 내용 미사용)")
    print("=" * 62)
    for chain in ("heavy", "light"):
        tr = load_fold(chain, "train", n_train)
        te = load_fold(chain, "test", n_test)
        positional_majority(tr, te, chain)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=200000)
    ap.add_argument("--n-test", type=int, default=50000)
    args = ap.parse_args()
    task_vh_vl(args.n_train, args.n_test)
    task_cdr(args.n_train, args.n_test)


if __name__ == "__main__":
    main()
