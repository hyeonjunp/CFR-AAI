#!/usr/bin/env python
"""데이터 분포를 계산해 Markdown 리포트(REPORT.md)로 저장한다.

python 05_report.py              # processed 단계만
python 05_report.py --dataset    # dataset(클러스터/fold) 단계까지 포함
"""
import os, argparse, datetime
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
READ = dict(keep_default_na=False, na_values=[], low_memory=False)


def load(which):
    return pd.read_csv(os.path.join(ROOT, "processed", which, which + ".csv.gz"), **READ)


def md_table(df, floatfmt="%.2f"):
    """DataFrame -> GitHub Markdown 표."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return floatfmt % v
        if isinstance(v, (int, np.integer)):
            return "{:,}".format(v)
        return str(v)
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |"
            for r in df.itertuples(index=False, name=None)]
    return "\n".join([head, sep] + rows)


def dist_table(s, bins):
    """값 분포를 구간별 표로."""
    cut = pd.cut(s, bins=bins, right=False)
    vc = cut.value_counts().sort_index()
    return pd.DataFrame({"구간": [str(i) for i in vc.index],
                         "행 수": vc.values,
                         "비율(%)": 100.0 * vc.values / len(s)})


def length_stats(d, chain):
    rows = []
    for col, label in (("seq_len", "전체 서열"), ("CDR1_len", "CDR1"),
                       ("CDR2_len", "CDR2"), ("CDR3_len", "CDR3")):
        s = d[col]
        rows.append(dict(항목=label, 평균=s.mean(), 표준편차=s.std(),
                         최소=int(s.min()), P25=int(s.quantile(.25)),
                         중앙값=int(s.median()), P75=int(s.quantile(.75)),
                         P99=int(s.quantile(.99)), 최대=int(s.max())))
    return pd.DataFrame(rows)


def counts_table(s, name, top=None):
    vc = s.value_counts()
    if top:
        vc = vc.head(top)
    return pd.DataFrame({name: vc.index, "행 수": vc.values,
                         "비율(%)": 100.0 * vc.values / len(s)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="store_true", help="클러스터/fold 결과까지 포함")
    ap.add_argument("--out", default=os.path.join(ROOT, "REPORT.md"))
    args = ap.parse_args()

    L = []
    A = L.append
    A("# OAS Paired 항체 CDR 데이터셋 — 분포 리포트\n")
    A("생성: %s\n" % datetime.date.today().isoformat())
    A("소스: OAS Paired 전체 (21 studies / 610 data units), IMGT CDR 정의\n")

    heavy, light = load("heavy"), load("light")

    A("\n## 1. 규모\n")
    A(md_table(pd.DataFrame([
        dict(구분="heavy", 행수=len(heavy), 유니크서열=heavy.sequence.nunique(),
             유니크CDR3=heavy.CDR3.nunique()),
        dict(구분="light", 행수=len(light), 유니크서열=light.sequence.nunique(),
             유니크CDR3=light.CDR3.nunique())])))

    for name, d in (("heavy", heavy), ("light", light)):
        A("\n## 2-%s. %s 체인\n" % ("H" if name == "heavy" else "L", name))
        A("\n### 길이 통계 (잔기)\n")
        A(md_table(length_stats(d, name)))
        A("\n### CDR3 길이 분포\n")
        bins = list(range(0, 36, 3)) + [100]
        A(md_table(dist_table(d.CDR3_len, bins)))
        A("\n### CDR1 / CDR2 길이 분포 (상위 8)\n")
        for c in ("CDR1_len", "CDR2_len"):
            A("\n**%s**\n" % c)
            A(md_table(counts_table(d[c].astype(str), "길이", top=8)))
        A("\n### locus\n")
        A(md_table(counts_table(d.locus, "locus")))
        A("\n### 품질 플래그\n")
        A(md_table(counts_table(d.quality_flag, "quality_flag")))
        A("\n### V 유전자 패밀리 (상위 10)\n")
        fam = d.v_call.str.split(",").str[0].str.split("-").str[0].str.split("*").str[0]
        A(md_table(counts_table(fam, "V family", top=10)))

    A("\n## 3. 메타데이터 분포 (heavy 기준)\n")
    for col, label in (("species", "종"), ("study", "study"),
                       ("btype", "B세포 타입"), ("bsource", "조직"),
                       ("disease", "질환")):
        A("\n### %s\n" % label)
        A(md_table(counts_table(heavy[col], label, top=12)))

    A("\n### study x subject 그룹 (split key 후보)\n")
    g = heavy.groupby(["study", "subject"]).size().sort_values(ascending=False)
    A("- 그룹 수: **%d**\n" % len(g))
    A("- subject 가 `no`/`None` 인 비율: **%.1f%%** (해당 study 는 study 통째로 한 그룹)\n"
      % (100 * heavy.subject.isin(["no", "None", ""]).mean()))
    A(md_table(pd.DataFrame({"study": [i[0] for i in g.index[:12]],
                             "subject": [i[1] for i in g.index[:12]],
                             "행 수": g.values[:12],
                             "비율(%)": 100.0 * g.values[:12] / len(heavy)})))

    if args.dataset:
        A("\n## 4. 중복 제거 + cd-hit 90% 클러스터 + 5-fold\n")
        for name in ("heavy", "light"):
            p = os.path.join(ROOT, "dataset", name, "all_with_cluster_fold.csv.gz")
            if not os.path.exists(p):
                continue
            d = pd.read_csv(p, **READ)
            n_raw = len(heavy) if name == "heavy" else len(light)
            A("\n### %s\n" % name)
            A("- 중복 제거: %s -> %s (%.1f%%)\n"
              % ("{:,}".format(n_raw), "{:,}".format(len(d)), 100.0 * len(d) / n_raw))
            A("- 클러스터 수: **%s** (평균 %.2f 서열/클러스터)\n"
              % ("{:,}".format(d.cluster_id.nunique()), len(d) / d.cluster_id.nunique()))
            cs = d.groupby("cluster_id").size()
            A("- 클러스터 크기: 중앙값 %d, P99 %d, 최대 %s\n"
              % (cs.median(), cs.quantile(.99), "{:,}".format(cs.max())))
            A("\n**fold 별 행 수**\n")
            fold = d.groupby("fold").agg(행수=("sequence", "size"),
                                         클러스터=("cluster_id", "nunique")).reset_index()
            fold["비율(%)"] = 100.0 * fold["행수"] / len(d)
            A(md_table(fold))
            per = d.groupby("fold")["cluster_id"].apply(set)
            ov = sum(len(per[i] & per[j]) for i in per.index for j in per.index if i < j)
            A("\n- fold 간 클러스터 중복: **%d 건** (0 이어야 정상)\n" % ov)
            A("\n**fold 별 CDR3 평균 길이** (분포가 고른지 확인)\n")
            t = d.groupby("fold")["CDR3_len"].agg(["mean", "std", "median"]).reset_index()
            A(md_table(t))

    with open(args.out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
