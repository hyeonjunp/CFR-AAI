"""
antibody.csv (ab_id, ab_seq) → ANARCI IMGT 넘버링으로 FR/CDR 추출

사용 환경: conda env `AbTLAS` (ANARCI 1.3 + hmmer 설치돼 있음)

    conda activate AbTLAS
    python extract_imgt_cdr.py                       # antibody.csv → antibody_imgt_cdr.csv
    python extract_imgt_cdr.py --flank 2             # CDR 양옆 2잔기씩 붙여서 추출
    python extract_imgt_cdr.py -i x.csv -o y.csv --seq_col seq --id_col id

IMGT 구간 정의 (Lefranc et al. 2003):
    FR1  1-26 | CDR1  27-38 | FR2  39-55 | CDR2  56-65 | FR3  66-104 | CDR3 105-117 | FR4 118-128
IMGT는 gap을 넣어 길이를 맞추는 스킴이라 넘버링 결과에는 '-'가 섞여 있다. 실제 서열만 필요하므로
gap은 버리고 residue만 모은다. insertion code(111A, 112A ...)는 CDR3 길이 확장용이라 그대로 포함된다.

출력 컬럼:
    ab_id, ab_seq            원본
    chain_type               H / K / L  (VHH도 H로 나옴)
    fr1..fr4, cdr1..cdr3     IMGT 구간별 서열
    cdr_mask                 ab_seq와 같은 길이의 0/1 문자열 (1 = CDR)
    domain_start/domain_end  ab_seq 상에서 ANARCI가 잡은 가변도메인 범위 (end는 inclusive)
    status                   ok / no_domain (도메인 못 잡음) / missing_cdr (CDR 구간이 빔)
"""

import argparse
import os
import sys

import pandas as pd
from anarci import run_anarci

HERE = os.path.dirname(os.path.abspath(__file__))

# IMGT 구간 경계 (start, end) — 둘 다 inclusive
IMGT_REGIONS = [
    ("fr1", 1, 26),
    ("cdr1", 27, 38),
    ("fr2", 39, 55),
    ("cdr2", 56, 65),
    ("fr3", 66, 104),
    ("cdr3", 105, 117),
    ("fr4", 118, 128),
]
CDR_REGIONS = ("cdr1", "cdr2", "cdr3")


def region_of(pos):
    for name, lo, hi in IMGT_REGIONS:
        if lo <= pos <= hi:
            return name
    return None


def parse_domain(domain_numbering, start, seq):
    """ANARCI 한 도메인의 넘버링을 (구간별 서열, 구간별 서열 인덱스)로 변환.

    domain_numbering: [((pos, ins_code), aa), ...]  — aa가 '-'면 IMGT gap
    start: 원본 서열에서 도메인이 시작하는 인덱스
    gap이 아닌 잔기는 원본 서열 상에서 start부터 순서대로 1:1 대응한다.
    """
    regions = {name: [] for name, _, _ in IMGT_REGIONS}
    cdr_idx = []
    i = start
    for (pos, _ins), aa in domain_numbering:
        if aa == "-":
            continue
        name = region_of(pos)
        if name is not None:
            regions[name].append(aa)
            if name in CDR_REGIONS:
                cdr_idx.append(i)
        i += 1
    return {k: "".join(v) for k, v in regions.items()}, cdr_idx


def extend(seq, sub_start, sub_len, flank):
    """서열 상 [sub_start, sub_start+sub_len) 구간을 양쪽으로 flank 만큼 늘려서 반환."""
    lo = max(0, sub_start - flank)
    hi = min(len(seq), sub_start + sub_len + flank)
    return seq[lo:hi]


def extract_cdrs(seqs, ncpu=1, species=None, flank=0, id_col="ab_id", seq_col="ab_seq"):
    """서열 리스트 → IMGT FR/CDR 레코드 리스트.

    seqs: [(id, sequence), ...]
    반환: 입력과 같은 순서의 dict 리스트. status 는 ok / no_domain / missing_cdr.
    build_il6_dataset.py 가 손상 서열을 걸러내려고 이 함수를 그대로 쓴다.
    """
    print(f"[info] {len(seqs):,} sequences → ANARCI (imgt, ncpu={ncpu})", flush=True)
    # run_anarci → (입력 서열, 넘버링, 도메인 정보, hmmer hit table)
    _, numbering, details, _ = run_anarci(
        seqs, scheme="imgt", ncpu=ncpu, allowed_species=species
    )

    rows = []
    for (_id, seq), num, det in zip(seqs, numbering, details):
        rec = {
            id_col: _id,
            seq_col: seq,
            "chain_type": "",
            **{name: "" for name, _, _ in IMGT_REGIONS},
            "cdr_mask": "0" * len(seq),
            "domain_start": -1,
            "domain_end": -1,
            "status": "no_domain",
        }
        if num:
            # 도메인이 여러 개면 (scFv 등) 가장 긴 것 하나만 사용
            k = max(range(len(num)), key=lambda j: num[j][2] - num[j][1])
            domain_numbering, start, end = num[k]
            regions, cdr_idx = parse_domain(domain_numbering, start, seq)

            if flank:
                # CDR이 원본 서열에서 차지하는 구간을 양옆으로 flank 만큼 확장
                offsets = cdr_offsets(domain_numbering, start)
                for name in CDR_REGIONS:
                    if regions[name]:
                        sub_start, sub_len = offsets[name]
                        regions[name] = extend(seq, sub_start, sub_len, flank)

            mask = ["0"] * len(seq)
            for i in cdr_idx:
                mask[i] = "1"
            rec.update(regions)
            rec["cdr_mask"] = "".join(mask)
            rec["chain_type"] = det[k]["chain_type"]
            rec["domain_start"] = start
            rec["domain_end"] = end
            # 도메인은 잡혔어도 서열이 잘렸거나 프레임시프트면 CDR 구간이 비어서 나온다
            rec["status"] = "ok" if all(regions[c] for c in CDR_REGIONS) else "missing_cdr"
        rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser(description="ANARCI IMGT 넘버링으로 CDR 추출")
    ap.add_argument("-i", "--input", default=os.path.join(HERE, "antibody.csv"))
    ap.add_argument("-o", "--output", default=os.path.join(HERE, "antibody_imgt_cdr.csv"))
    ap.add_argument("--id_col", default="ab_id")
    ap.add_argument("--seq_col", default="ab_seq")
    ap.add_argument("--ncpu", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--flank", type=int, default=0,
                    help="CDR 양옆에 붙일 잔기 수 (기본 0 = 순수 IMGT 구간)")
    ap.add_argument("--species", default=None,
                    help="쉼표로 구분된 ANARCI allowed_species (예: alpaca,human). 기본은 전체")
    ap.add_argument("--drop_failed", action="store_true",
                    help="CDR을 못 뽑은 서열(no_domain / missing_cdr)을 출력에서 제외")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    for col in (args.id_col, args.seq_col):
        if col not in df.columns:
            sys.exit(f"[error] '{col}' 컬럼이 {args.input} 에 없음. 있는 컬럼: {list(df.columns)}")

    seqs = [(str(i), str(s)) for i, s in zip(df[args.id_col], df[args.seq_col])]
    species = args.species.split(",") if args.species else None
    rows = extract_cdrs(seqs, ncpu=args.ncpu, species=species, flank=args.flank,
                        id_col=args.id_col, seq_col=args.seq_col)

    full = pd.DataFrame(rows)
    out = full[full["status"] == "ok"].reset_index(drop=True) if args.drop_failed else full
    out.to_csv(args.output, index=False)

    counts = full["status"].value_counts()
    n_ok = int(counts.get("ok", 0))
    print(f"[done] {args.output}   rows: {len(out):,}")
    print(f"  CDR 추출 성공: {n_ok:,} / {len(full):,} ({n_ok / len(full) * 100:.2f}%)")
    for st in ("no_domain", "missing_cdr"):
        if counts.get(st):
            print(f"  {st}: {counts[st]:,}" + ("  (출력에서 제외됨)" if args.drop_failed else ""))
    for name in CDR_REGIONS:
        s = out.loc[out["status"] == "ok", name]
        if len(s):
            print(f"  {name}: len {s.str.len().min()}-{s.str.len().max()} "
                  f"(median {int(s.str.len().median())}), unique {s.nunique():,}")


def cdr_offsets(domain_numbering, start):
    """각 CDR이 원본 서열에서 시작하는 인덱스와 길이."""
    out = {}
    i = start
    for (pos, _ins), aa in domain_numbering:
        if aa == "-":
            continue
        name = region_of(pos)
        if name in CDR_REGIONS:
            if name not in out:
                out[name] = [i, 0]
            out[name][1] += 1
        i += 1
    return {k: (v[0], v[1]) for k, v in out.items()}


if __name__ == "__main__":
    main()
