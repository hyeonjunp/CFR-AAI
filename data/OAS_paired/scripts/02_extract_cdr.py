#!/usr/bin/env python
"""
OAS Paired 원본(csv.gz) -> heavy / light 분리 + IMGT CDR/FR 추출

- OAS의 fwr*_aa / cdr*_aa 컬럼은 ANARCI IMGT 넘버링 기반 (OAS 문서 기준).
- OAS에는 fwr4 컬럼이 없어서 sequence_alignment_aa 의 잔여 접미사로 복원한다.
- 인덱스 규약: 0-based start, end-exclusive  (CDR_prediction/Dataset 과 동일)
- locus 가 체인과 불일치하는 OAS 오배정 행은 제거한다.

[읽을 때 주의] 아미노산 서열 "NA"(Asn-Ala) 를 pandas 가 결측치로 바꿔버린다. 반드시
    pd.read_csv(path, keep_default_na=False, na_values=[])
로 읽을 것. (scripts/load.py 의 load_oas() 사용 권장)

출력 (data/OAS_paired/processed/)
    heavy/heavy.csv.gz    체인 단위 long 포맷 (heavy만)
    light/light.csv.gz    체인 단위 long 포맷 (light만)
    paired/paired.csv.gz  pair 단위 wide 포맷 (H/L 한 행)
    unit_summary.csv      데이터유닛별 통계
"""
import os, sys, csv, gzip, json, ast, glob, argparse
import pandas as pd

AA = set("ACDEFGHIKLMNPQRSTVWY")

REGION_COLS = ["fwr1_aa", "cdr1_aa", "fwr2_aa", "cdr2_aa", "fwr3_aa", "cdr3_aa"]
BASE_COLS = ["sequence_id", "locus", "stop_codon", "vj_in_frame", "productive",
             "v_call", "d_call", "j_call", "sequence_alignment_aa",
             "junction_aa", "ANARCI_status"] + REGION_COLS

CHAIN_FIELDS = ["sample_id", "pair_id", "study", "unit", "sequence_id",
                "chain_type", "locus", "species", "disease", "btype", "bsource",
                "vaccine", "subject", "age", "isotype",
                "sequence", "seq_len",
                "FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4",
                "CDR1_len", "CDR2_len", "CDR3_len",
                "CDR1_start", "CDR1_end", "CDR2_start", "CDR2_end",
                "CDR3_start", "CDR3_end",
                "mask", "v_call", "d_call", "j_call", "junction_aa",
                "ANARCI_status", "quality_flag"]

PAIR_FIELDS = (["pair_id", "study", "unit", "species", "disease", "btype",
                "bsource", "vaccine", "subject", "age", "isotype"]
               + ["heavy_" + f for f in ["sequence", "FR1", "CDR1", "FR2", "CDR2",
                                         "FR3", "CDR3", "FR4", "v_call", "j_call",
                                         "locus", "mask"]]
               + ["light_" + f for f in ["sequence", "FR1", "CDR1", "FR2", "CDR2",
                                         "FR3", "CDR3", "FR4", "v_call", "j_call",
                                         "locus", "mask"]])


def read_metadata(path):
    """csv.gz 첫 줄에 JSON 메타데이터가 따옴표로 감싸여 들어있다."""
    with gzip.open(path, "rt") as fh:
        first = next(fh)
    try:
        return json.loads(next(csv.reader([first]))[0])
    except Exception:
        try:
            return ast.literal_eval(next(csv.reader([first]))[0])
        except Exception:
            return {}


def build_chain(df, chain):
    """chain('heavy'/'light') 접미사 컬럼만 뽑아 표준 컬럼명으로 되돌린다."""
    cols = {c + "_" + chain: c for c in BASE_COLS if c + "_" + chain in df.columns}
    out = df[list(cols)].rename(columns=cols).copy()
    out["row"] = df.index
    return out


def annotate(sub, chain_kind):
    """FR4 복원 + 인덱스/마스크 계산 + 품질 필터. 통과한 행만 돌려준다."""
    for c in REGION_COLS + ["sequence_alignment_aa"]:
        if c not in sub.columns:
            sub[c] = None
    sub = sub.dropna(subset=REGION_COLS + ["sequence_alignment_aa"])
    if sub.empty:
        return sub

    # 생산적 서열만 + locus 가 해당 체인과 맞는 행만
    #   (OAS 원본에 heavy 컬럼인데 locus 가 K/L 인 오배정 행이 극소수 섞여 있다)
    keep = (sub["productive"].astype(str).str.upper() == "T")
    if "locus" in sub.columns:
        want = ["H"] if chain_kind == "heavy" else ["K", "L"]
        keep &= sub["locus"].astype(str).str.upper().isin(want)
    if "stop_codon" in sub:
        keep &= (sub["stop_codon"].astype(str).str.upper() != "T")
    if "vj_in_frame" in sub:
        keep &= (sub["vj_in_frame"].astype(str).str.upper() != "F")
    sub = sub[keep]
    if sub.empty:
        return sub

    fr1, cdr1 = sub["fwr1_aa"].astype(str), sub["cdr1_aa"].astype(str)
    fr2, cdr2 = sub["fwr2_aa"].astype(str), sub["cdr2_aa"].astype(str)
    fr3, cdr3 = sub["fwr3_aa"].astype(str), sub["cdr3_aa"].astype(str)
    vdj = sub["sequence_alignment_aa"].astype(str).str.replace(".", "", regex=False)

    head = fr1 + cdr1 + fr2 + cdr2 + fr3 + cdr3
    # sequence_alignment_aa 가 FR1..CDR3 로 시작해야 FR4 를 잘라낼 수 있다
    ok = [v.startswith(h) for v, h in zip(vdj, head)]
    sub, fr1, cdr1, fr2, cdr2, fr3, cdr3, head, vdj = [
        x[ok] for x in (sub, fr1, cdr1, fr2, cdr2, fr3, cdr3, head, vdj)]
    if sub.empty:
        return sub

    fr4 = pd.Series([v[len(h):] for v, h in zip(vdj, head)], index=sub.index)
    seq = head + fr4

    # 알파벳 검증 (X, *, 결측 없음)
    valid = [bool(s) and set(s) <= AA for s in seq]
    sub, fr1, cdr1, fr2, cdr2, fr3, cdr3, fr4, seq = [
        x[valid] for x in (sub, fr1, cdr1, fr2, cdr2, fr3, cdr3, fr4, seq)]
    if sub.empty:
        return sub

    l1, l2, l3 = cdr1.str.len(), cdr2.str.len(), cdr3.str.len()
    s1 = fr1.str.len()
    s2 = s1 + l1 + fr2.str.len()
    s3 = s2 + l2 + fr3.str.len()

    sub["FR1"], sub["CDR1"], sub["FR2"], sub["CDR2"] = fr1, cdr1, fr2, cdr2
    sub["FR3"], sub["CDR3"], sub["FR4"] = fr3, cdr3, fr4
    sub["sequence"] = seq
    sub["seq_len"] = seq.str.len()
    sub["CDR1_len"], sub["CDR2_len"], sub["CDR3_len"] = l1, l2, l3
    sub["CDR1_start"], sub["CDR1_end"] = s1, s1 + l1
    sub["CDR2_start"], sub["CDR2_end"] = s2, s2 + l2
    sub["CDR3_start"], sub["CDR3_end"] = s3, s3 + l3
    sub["mask"] = [
        "0" * a + "1" * b + "0" * c + "1" * d + "0" * e + "1" * f + "0" * g
        for a, b, c, d, e, f, g in zip(fr1.str.len(), l1, fr2.str.len(), l2,
                                       fr3.str.len(), l3, fr4.str.len())]
    # FR4 가 짧거나 CDR3 길이가 비정상이면 표시만 해둔다 (제거는 하지 않음)
    sub["quality_flag"] = "ok"
    sub.loc[(fr4.str.len() < 8) | (l3 < 1) | (l3 > 40), "quality_flag"] = "check"
    return sub


def process_unit(path, study):
    unit = os.path.basename(path).replace(".csv.gz", "")
    meta = read_metadata(path)
    df = pd.read_csv(path, skiprows=1, low_memory=False)
    df = df.reset_index(drop=True)

    heavy = annotate(build_chain(df, "heavy"), "heavy")
    light = annotate(build_chain(df, "light"), "light")

    common = dict(
        study=study, unit=unit,
        species=meta.get("Species", ""), disease=meta.get("Disease", ""),
        btype=meta.get("BType", ""), bsource=meta.get("BSource", ""),
        vaccine=meta.get("Vaccine", ""), subject=meta.get("Subject", ""),
        age=meta.get("Age", ""), isotype=meta.get("Isotype", ""))

    frames = {}
    for name, sub in (("heavy", heavy), ("light", light)):
        if sub.empty:
            frames[name] = pd.DataFrame(columns=CHAIN_FIELDS)
            continue
        sub = sub.copy()
        sub["pair_id"] = ["%s|%s|%d" % (study, unit, i) for i in sub["row"]]
        sub["sample_id"] = sub["pair_id"] + ("|H" if name == "heavy" else "|L")
        sub["chain_type"] = name
        for k, v in common.items():
            sub[k] = v
        for c in CHAIN_FIELDS:
            if c not in sub.columns:
                sub[c] = ""
        frames[name] = sub[CHAIN_FIELDS]

    # pair: 양쪽 다 살아남은 행만
    hi = heavy.set_index("row") if not heavy.empty else None
    li = light.set_index("row") if not light.empty else None
    if hi is None or li is None:
        pair = pd.DataFrame(columns=PAIR_FIELDS)
    else:
        rows = hi.index.intersection(li.index)
        hi, li = hi.loc[rows], li.loc[rows]
        pair = pd.DataFrame({"pair_id": ["%s|%s|%d" % (study, unit, i) for i in rows]})
        for k, v in common.items():
            pair[k] = v
        for pfx, src in (("heavy_", hi), ("light_", li)):
            for f in ["sequence", "FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3",
                      "FR4", "v_call", "j_call", "locus", "mask"]:
                pair[pfx + f] = src[f].values
        pair = pair[PAIR_FIELDS]

    stat = dict(study=study, unit=unit, raw_rows=len(df),
                heavy_kept=len(frames["heavy"]), light_kept=len(frames["light"]),
                pair_kept=len(pair), species=common["species"],
                disease=common["disease"], btype=common["btype"])
    return frames["heavy"], frames["light"], pair, stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(os.path.dirname(__file__), "..", "raw"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "processed"))
    ap.add_argument("--limit", type=int, default=0, help="유닛 N개만 처리 (테스트용)")
    args = ap.parse_args()

    raw, out = os.path.abspath(args.raw), os.path.abspath(args.out)
    for d in ("heavy", "light", "paired"):
        os.makedirs(os.path.join(out, d), exist_ok=True)

    files = sorted(glob.glob(os.path.join(raw, "*", "*.csv.gz")))
    if args.limit:
        files = files[:args.limit]
    print("units: %d" % len(files))

    handles = {
        "heavy": gzip.open(os.path.join(out, "heavy", "heavy.csv.gz"), "wt", newline=""),
        "light": gzip.open(os.path.join(out, "light", "light.csv.gz"), "wt", newline=""),
        "paired": gzip.open(os.path.join(out, "paired", "paired.csv.gz"), "wt", newline=""),
    }
    written = {k: False for k in handles}
    stats = []
    for n, path in enumerate(files, 1):
        study = os.path.basename(os.path.dirname(path))
        try:
            h, l, p, st = process_unit(path, study)
        except Exception as e:
            print("  !! %s/%s : %s" % (study, os.path.basename(path), e))
            stats.append(dict(study=study, unit=os.path.basename(path),
                              raw_rows=-1, heavy_kept=0, light_kept=0,
                              pair_kept=0, species="", disease="", btype=""))
            continue
        for key, frame in (("heavy", h), ("light", l), ("paired", p)):
            frame.to_csv(handles[key], index=False, header=not written[key])
            written[key] = True
        stats.append(st)
        print("[%4d/%d] %-18s %-34s raw=%-7d H=%-7d L=%-7d pair=%-7d"
              % (n, len(files), study, st["unit"][:34], st["raw_rows"],
                 st["heavy_kept"], st["light_kept"], st["pair_kept"]))
        sys.stdout.flush()

    for fh in handles.values():
        fh.close()
    sdf = pd.DataFrame(stats)
    sdf.to_csv(os.path.join(out, "unit_summary.csv"), index=False)
    print("\n=== 합계 ===")
    print("units      : %d" % len(sdf))
    print("heavy rows : %d" % sdf["heavy_kept"].sum())
    print("light rows : %d" % sdf["light_kept"].sum())
    print("paired rows: %d" % sdf["pair_kept"].sum())


if __name__ == "__main__":
    main()
