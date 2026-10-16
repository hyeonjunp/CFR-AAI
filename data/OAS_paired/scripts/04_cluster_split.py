#!/usr/bin/env python
"""중복 제거 -> cd-hit 90% 클러스터링 -> 클러스터 단위 5-fold split.

같은 클론(거의 동일한 서열)이 train/test 양쪽에 들어가면 성능이 부풀려지므로,
서열 유사도 90% 클러스터를 그룹으로 묶어 그룹 단위로 fold 를 나눈다.

단계
  1. exact dedup      동일 `sequence` 는 1행만 남기고 redundancy 로 개수 보존
  2. cd-hit -c 0.9    유니크 서열을 90% identity 로 클러스터링
  3. GroupKFold       클러스터 id 를 그룹으로 5-fold (train/valid/test)

사용
  python 04_cluster_split.py --chain heavy
  python 04_cluster_split.py --chain light
"""
import os
import shutil, sys, gzip, argparse, subprocess, collections
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CDHIT = shutil.which("cd-hit") or os.path.join(ROOT, "bin", "cd-hit")


def read_chain(chain):
    """NA(Asn-Ala) 가 결측치로 변환되지 않도록 keep_default_na=False 필수."""
    path = os.path.join(ROOT, "processed", chain, chain + ".csv.gz")
    return pd.read_csv(path, keep_default_na=False, na_values=[], low_memory=False)


def dedup(df):
    """동일 서열은 1행만 남기고 중복 수를 redundancy 컬럼에 기록."""
    counts = df["sequence"].value_counts()
    out = df.drop_duplicates(subset="sequence", keep="first").copy()
    out["redundancy"] = out["sequence"].map(counts).astype(int)
    return out.reset_index(drop=True)


def write_fasta(df, path):
    with open(path, "w") as fh:
        for i, seq in enumerate(df["sequence"].values):
            fh.write(">%d\n%s\n" % (i, seq))


def run_cdhit(fasta, out_prefix, identity=0.9, threads=16, mem_mb=64000, word=5):
    cmd = [CDHIT, "-i", fasta, "-o", out_prefix,
           "-c", str(identity), "-n", str(word),
           "-T", str(threads), "-M", str(mem_mb), "-d", "0"]
    print("  $ " + " ".join(cmd)); sys.stdout.flush()
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL)
    return out_prefix + ".clstr"


def parse_clstr(path, n_seq):
    """cd-hit .clstr -> 서열 인덱스별 클러스터 id 배열."""
    cluster = np.full(n_seq, -1, dtype=np.int64)
    cid = -1
    with open(path) as fh:
        for line in fh:
            if line.startswith(">Cluster"):
                cid = int(line.split()[1])
            else:
                # 예: "0	118aa, >12345... *"
                tag = line.split(">", 1)[1].split("...")[0]
                cluster[int(tag)] = cid
    if (cluster < 0).any():
        raise RuntimeError("클러스터 배정 누락: %d" % int((cluster < 0).sum()))
    return cluster


def group_kfold(groups, n_splits=5, seed=0):
    """클러스터를 크기 내림차순으로 가장 작은 fold 에 배정 (fold 크기 균형)."""
    rng = np.random.RandomState(seed)
    sizes = collections.Counter(groups)
    order = sorted(sizes, key=lambda g: (-sizes[g], rng.rand()))
    load = np.zeros(n_splits, dtype=np.int64)
    assign = {}
    for g in order:
        k = int(load.argmin())
        assign[g] = k
        load[k] += sizes[g]
    return np.array([assign[g] for g in groups]), load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", required=True, choices=["heavy", "light"])
    ap.add_argument("--identity", type=float, default=0.9)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = args.out or os.path.join(ROOT, "dataset", args.chain)
    work = os.path.join(out_dir, "cdhit")
    os.makedirs(work, exist_ok=True)

    print("[1/4] 로드 + 중복 제거")
    df = read_chain(args.chain)
    n_raw = len(df)
    df = dedup(df)
    print("      %d -> %d (유니크 %.1f%%)" % (n_raw, len(df), 100.0 * len(df) / n_raw))

    print("[2/4] cd-hit %.0f%% 클러스터링" % (args.identity * 100))
    fasta = os.path.join(work, "input.fasta")
    write_fasta(df, fasta)
    clstr = run_cdhit(fasta, os.path.join(work, "cdhit_%d" % int(args.identity * 100)),
                      identity=args.identity, threads=args.threads)
    df["cluster_id"] = parse_clstr(clstr, len(df))
    n_clusters = df["cluster_id"].nunique()
    print("      클러스터 %d 개 (서열 %d 개, 평균 %.2f)"
          % (n_clusters, len(df), len(df) / n_clusters))

    print("[3/4] 클러스터 단위 %d-fold 배정" % args.folds)
    df["fold"], load = group_kfold(df["cluster_id"].values, args.folds)
    print("      fold 크기: " + ", ".join("%d" % x for x in load))

    print("[4/4] train/valid/test 저장")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "all_with_cluster_fold.csv.gz"), index=False)
    for k in range(args.folds):
        test_f, valid_f = k, (k + 1) % args.folds
        split = np.where(df["fold"] == test_f, "test",
                np.where(df["fold"] == valid_f, "valid", "train"))
        fdir = os.path.join(out_dir, "fold_%d" % k)
        os.makedirs(fdir, exist_ok=True)
        for name in ("train", "valid", "test"):
            df[split == name].to_csv(os.path.join(fdir, name + ".csv.gz"), index=False)
        print("      fold_%d  train=%d valid=%d test=%d"
              % (k, (split == "train").sum(), (split == "valid").sum(),
                 (split == "test").sum()))

    # 누수 점검: fold 간 클러스터 교집합이 없어야 한다
    per_fold = df.groupby("fold")["cluster_id"].apply(set)
    overlap = sum(len(per_fold[i] & per_fold[j])
                  for i in range(args.folds) for j in range(i + 1, args.folds))
    print("\n클러스터 누수: %d 건 (0 이어야 정상)" % overlap)


if __name__ == "__main__":
    main()
