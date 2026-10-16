"""
data/il6 데이터셋 빌더

원본 il6_aai_dataset.csv (VHH_sequence, label, Ag_label, Ag_sequence — 573,891 pairs,
38,599 unique VHH, 31 unique 항원)를 AVIDa_hIL6와 동일한 폴더 구조로 변환:

    antibody.csv          ab_id, ab_seq            (VHH_sequence 중복 제거)
    antigen.csv            ag_id, ag_seq, Ag_label  (Ag_sequence 중복 제거)
    ab_ag_pair.csv          label, Ag_label, ab_id, ag_id   (원본 순서 유지, 중복 정리 후 index = select_index 기준)
    ag_unseen/train_seed_N.csv / val_seed_N.csv / unseen_seed_N.csv   (N=0..4)

── 태그 제거 (--keep_tags 로 끌 수 있음, 기본은 제거) ────────────────────────
원본 서열은 발현 컨스트럭트 전체라 가변도메인 바깥의 상수 구간이 붙어 있다:

    VHH  : MKYLLPTAAAGLLLLAAQPAMA | QVQLQESGGG ... QVTVSS | HHHHHH
           └─ PelB signal peptide(22aa)              └─ His6
    IL-6 : MNSFSTSAFGPVAFSLGLLLVLPAAFPAP | VPPGEDSKD ... RALRQM | HHHHHH
           └─ native signal peptide(29aa)                     └─ His6

모든 서열에 동일하게 붙어 있어 결합 여부와 무관하고 ESM2 입력 길이만 잡아먹으므로 제거한다.
(AVIDa_hIL6 폴더는 ANARCI로 CDR만 추출한 축소판이라 이미 태그가 없다 — il6 원본만 해당.)

단순 str.removeprefix()로는 부족하다: VHH 38,599개 중 527개(1.4%)는 signal peptide 안에 점돌연변이가
있어(MKYLLPTAAAGLLLLAAQPA[L]A 등) 문자열이 정확히 일치하지 않는다. 그래서 앞 22aa를 canonical PelB와
hamming 거리로 비교해 MAX_MISMATCH 이하면 잘라낸다. His-tag는 C말단 H 연속 구간으로 찾는다.

주의: 태그를 떼면 signal peptide 변이로만 달랐던 VHH들이 같은 서열로 합쳐진다(38,599 → 38,109).
그 결과 같은 (ab_id, ag_id)가 중복되므로 분할 전에 정리한다 — label 일치 중복은 하나만 남기고,
label이 충돌하는 조합은 통째로 버린다(--keep_dupes로 끌 수 있음). 573,891 → 569,070 rows.

── 손상 서열 제거 (--keep_broken 으로 끌 수 있음, 기본은 제거) ──────────────
태그를 뗀 VHH 38,109개 중 186개(0.49%)는 ANARCI IMGT 넘버링으로 CDR을 뽑을 수 없다:

    no_domain    84개 — 가변도메인 자체가 안 잡힘. 58개는 길이는 정상(121~126aa)인데
                        N말단 ~75잔기가 통째로 다른 서열(QVQLQIHRVLGVVSGDGESALHG...)로
                        치환된 프레임시프트 아티팩트라 FR1~CDR2가 소실됐다.
                        나머지 26개는 중간이 결실된 truncated 서열(72~126aa).
    missing_cdr 102개 — 도메인은 잡혔지만 CDR 구간이 비어 나옴 (cdr1 76 / cdr2 65 / cdr3 26).

CDR3만 정규식으로 건질 수 있는 것들도 있지만 FR1/CDR1/CDR2는 어차피 복구 불가라 통째로 버린다.
이 186개를 빼면 참조가 끊기는 pair 2,516개(label 0이 2,500, label 1이 16)도 함께 제거된다.

주의: AVIDa_hIL6 폴더는 CDR 추출 + homology 필터링을 거친 축소판이지만, 여기서는 homology
필터링 툴이 없으므로 CDR 추출 가능 여부까지만 거른다.

ag_unseen 분할 로직 (AVIDa_hIL6/ag_unseen 기존 파일 분석으로 역산):
    - WT 항원(IL-6_WTs)은 항상 seen(train/val)에 포함
    - 나머지 30개 mutant 항원을 seed별로 반씩(15/15) seen/unseen으로 무작위 분할
    - seen 항원의 pair들을 다시 train_test_split(test_size=0.1, random_state=seed)로 train/val 분할
    - unseen 항원의 pair는 그대로 unseen_seed_N.csv (원본 순서 유지, shuffle 없음)
    - select_index 컬럼 = ab_ag_pair.csv 상의 원래 row index
"""

import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from extract_imgt_cdr import extract_cdrs

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "il6_aai_dataset.csv")
WT_LABEL = "IL-6_WTs"
N_SEEDS = 5
AG_SEEN_FRAC = 0.5      # WT 제외 mutant 항원 중 seen 비율
VAL_FRAC = 0.1          # seen 항원 pair 중 val 비율

# ── 태그 제거 설정 ──────────────────────────────────────────────────────────
PELB = "MKYLLPTAAAGLLLLAAQPAMA"              # VHH N말단 signal peptide (22aa)
IL6_SIGNAL = "MNSFSTSAFGPVAFSLGLLLVLPAAFPAP"  # IL-6 native signal peptide (29aa)
MIN_HIS = 6             # C말단 His-tag로 인정할 최소 H 연속 길이
MAX_MISMATCH = 3        # signal peptide 내 점돌연변이 허용치 (관측된 최대치는 1)


def strip_signal(seq: str, ref: str, max_mismatch: int = MAX_MISMATCH) -> str:
    """seq 앞부분이 ref와 max_mismatch 이하로 일치하면 ref 길이만큼 잘라낸다."""
    head = seq[:len(ref)]
    if len(head) < len(ref):
        return seq
    if sum(a != b for a, b in zip(head, ref)) <= max_mismatch:
        return seq[len(ref):]
    return seq


def strip_his(seq: str, min_his: int = MIN_HIS) -> str:
    """C말단 His-tag(H가 min_his개 이상 연속) 제거."""
    n = 0
    while n < len(seq) and seq[len(seq) - 1 - n] == "H":
        n += 1
    return seq[:len(seq) - n] if n >= min_his else seq


def strip_tags(seq: str, ref: str) -> str:
    return strip_his(strip_signal(str(seq).strip().upper(), ref))


def _strip_report(name: str, seqs: pd.Series, ref: str) -> pd.Series:
    """태그를 제거하고 단계별 통계를 출력한 뒤 정제된 서열을 반환."""
    before = seqs.astype(str).str.strip().str.upper()
    no_sig = before.map(lambda s, r=ref: strip_signal(s, r))
    after = no_sig.map(strip_his)

    n = len(before)
    sig_removed = no_sig.str.len() < before.str.len()
    his_removed = after.str.len() < no_sig.str.len()

    print(f"  {name}: {n} seqs "
          f"| signal peptide 제거 {int(sig_removed.sum())} ({sig_removed.mean():.1%})"
          f" | His-tag 제거 {int(his_removed.sum())} ({his_removed.mean():.1%})")
    print(f"    length  raw {before.str.len().min()}/{int(before.str.len().median())}/{before.str.len().max()}"
          f"  →  stripped {after.str.len().min()}/{int(after.str.len().median())}/{after.str.len().max()}"
          f"  (min/median/max)")
    print(f"    unique  {before.nunique()} → {after.nunique()}")

    # signal peptide가 안 떨어진 서열 = ref와 너무 많이 다름 → 앞부분이 잘린 손상 레코드일 가능성
    leftover = before[~sig_removed]
    if len(leftover):
        print(f"    ! signal peptide 미제거 {len(leftover)}건 — 앞부분이 잘린 손상 레코드로 보임:")
        for x in leftover.head(3):
            print(f"      {x[:60]}...")

    return after


# ── main ────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
parser.add_argument("--keep_tags", action="store_true",
                    help="PelB signal peptide / His-tag를 제거하지 않고 원본 서열 그대로 사용")
parser.add_argument("--keep_dupes", action="store_true",
                    help="중복 (ab_id, ag_id) pair를 정리하지 않고 그대로 둔다")
parser.add_argument("--keep_broken", action="store_true",
                    help="ANARCI로 CDR을 못 뽑는 손상 VHH를 제거하지 않고 그대로 둔다")
parser.add_argument("--ncpu", type=int, default=os.cpu_count() or 1,
                    help="ANARCI 넘버링에 쓸 CPU 수")
args = parser.parse_args()

print("Loading", SRC)
raw = pd.read_csv(SRC)
print("raw rows:", len(raw))

if args.keep_tags:
    print("\n[--keep_tags] 태그 제거 건너뜀 — 원본 서열 그대로 사용")
else:
    print("\n태그 제거 (PelB signal peptide + His-tag):")
    for col, ref, name in [("VHH_sequence", PELB, "VHH"),
                           ("Ag_sequence", IL6_SIGNAL, "IL-6")]:
        raw[col] = _strip_report(name, raw[col], ref)

# ── antibody.csv ────────────────────────────────────────────────────────────
ab_unique = raw["VHH_sequence"].drop_duplicates().reset_index(drop=True)

# CDR을 못 뽑는 손상 서열 제거. ab_id 부여 '전에' 걸러야 id가 0..N-1로 연속된다.
if args.keep_broken:
    print("\n[--keep_broken] 손상 VHH 제거 건너뜀")
else:
    print("\nANARCI IMGT 넘버링으로 CDR 추출 가능 여부 확인:")
    cdr_rows = extract_cdrs(list(zip(ab_unique.index.astype(str), ab_unique)), ncpu=args.ncpu)
    status = pd.Series([r["status"] for r in cdr_rows], index=ab_unique.index)
    n_before = len(ab_unique)
    ab_unique = ab_unique[status == "ok"].reset_index(drop=True)
    counts = status.value_counts()
    for st in ("no_domain", "missing_cdr"):
        if counts.get(st):
            print(f"  {st} 제거: {counts[st]}")
    print(f"  VHH: {n_before} → {len(ab_unique)} "
          f"({(n_before - len(ab_unique)) / n_before:.2%} 제거)")

ab_df = pd.DataFrame({"ab_id": range(len(ab_unique)), "ab_seq": ab_unique})
ab_map = dict(zip(ab_df["ab_seq"], ab_df["ab_id"]))
print("\nantibody.csv:", len(ab_df), "unique VHH")

# ── antigen.csv ─────────────────────────────────────────────────────────────
ag_unique = raw.drop_duplicates(subset=["Ag_sequence"])[["Ag_sequence", "Ag_label"]].reset_index(drop=True)
ag_df = pd.DataFrame({
    "ag_id": range(len(ag_unique)),
    "ag_seq": ag_unique["Ag_sequence"],
    "Ag_label": ag_unique["Ag_label"],
})
ag_map = dict(zip(ag_df["ag_seq"], ag_df["ag_id"]))
print("antigen.csv:", len(ag_df), "unique antigens")

# ── ab_ag_pair.csv ──────────────────────────────────────────────────────────
pair_df = pd.DataFrame({
    "label": raw["label"],
    "Ag_label": raw["Ag_label"],
    "ab_id": raw["VHH_sequence"].map(ab_map),
    "ag_id": raw["Ag_sequence"].map(ag_map),
})
assert pair_df["ag_id"].isna().sum() == 0

# 제거된 VHH를 참조하는 pair는 ab_id가 NaN이 된다 → 같이 버린다
n_orphan = int(pair_df["ab_id"].isna().sum())
if n_orphan:
    orphan_labels = pair_df.loc[pair_df["ab_id"].isna(), "label"].value_counts().to_dict()
    pair_df = pair_df[pair_df["ab_id"].notna()].reset_index(drop=True)
    print(f"  손상 VHH를 참조해 제거된 pair: {n_orphan} (label별 {orphan_labels})")
pair_df["ab_id"] = pair_df["ab_id"].astype(int)

print("ab_ag_pair.csv:", len(pair_df))

# ── 중복 pair 정리 ──────────────────────────────────────────────────────────
# 태그를 떼면 signal peptide 변이로만 달랐던 VHH들이 한 서열로 합쳐지면서(38,599 → 38,109)
# 같은 (ab_id, ag_id) 조합이 여러 번 나타난다. 반드시 train/val 분할 '전에' 정리해야 한다 —
# 그대로 두면 동일 샘플이 train과 val 양쪽에 들어가 val 성능이 부풀려진다.
#
#   label 일치 중복 → 하나만 남긴다 (정보 손실 없음)
#   label 충돌      → 조합 전체를 버린다. 충돌의 절반 이상이 정확히 동점(label 평균 0.5)이라
#                     다수결이 성립하지 않고, 나머지도 2~3 rows 근거뿐이라 신뢰할 수 없다.
if args.keep_dupes:
    print("  [--keep_dupes] 중복 pair 정리 건너뜀")
else:
    n_before = len(pair_df)
    grp = pair_df.groupby(["ab_id", "ag_id"])["label"]
    n_dup_combo = int((grp.size() > 1).sum())
    conflict = grp.nunique() > 1
    conflict_keys = conflict[conflict].index

    if len(conflict_keys):
        keys = pd.MultiIndex.from_frame(pair_df[["ab_id", "ag_id"]])
        pair_df = pair_df[~keys.isin(conflict_keys)]
    n_after_conflict = len(pair_df)

    pair_df = pair_df.drop_duplicates(subset=["ab_id", "ag_id"]).reset_index(drop=True)

    print(f"  중복 (ab_id, ag_id) 조합: {n_dup_combo}"
          f"  |  label 충돌 조합: {len(conflict_keys)}")
    print(f"  충돌 조합 제거   : -{n_before - n_after_conflict} rows")
    print(f"  중복 rows 제거   : -{n_after_conflict - len(pair_df)} rows")
    print(f"  ab_ag_pair.csv   : {n_before} → {len(pair_df)} rows "
          f"({(n_before - len(pair_df)) / n_before:.2%} 감소)")

ab_df.to_csv(os.path.join(HERE, "antibody.csv"), index=False)
ag_df.to_csv(os.path.join(HERE, "antigen.csv"), index=False)
pair_df.to_csv(os.path.join(HERE, "ab_ag_pair.csv"), index=False)
print("Saved antibody.csv / antigen.csv / ab_ag_pair.csv")

# ── ag_unseen splits ────────────────────────────────────────────────────────
out_dir = os.path.join(HERE, "ag_unseen")
os.makedirs(out_dir, exist_ok=True)

mutant_labels = sorted(l for l in pair_df["Ag_label"].unique() if l != WT_LABEL)
print(f"\n{len(mutant_labels)} mutant antigens (+ {WT_LABEL} always seen)")

for seed in range(N_SEEDS):
    seen_mutants, unseen_mutants = train_test_split(
        mutant_labels, test_size=AG_SEEN_FRAC, random_state=seed
    )
    seen_ags = set(seen_mutants) | {WT_LABEL}
    unseen_ags = set(unseen_mutants)

    seen_df = pair_df[pair_df["Ag_label"].isin(seen_ags)]
    unseen_df = pair_df[pair_df["Ag_label"].isin(unseen_ags)]

    train_df, val_df = train_test_split(seen_df, test_size=VAL_FRAC, random_state=seed)

    def with_select_index(df):
        out = df.reset_index().rename(columns={"index": "select_index"})
        return out[["select_index", "label", "Ag_label", "ab_id", "ag_id"]]

    train_out = with_select_index(train_df)
    val_out = with_select_index(val_df)
    unseen_out = with_select_index(unseen_df)

    train_out.to_csv(os.path.join(out_dir, f"train_seed_{seed}.csv"), index=False)
    val_out.to_csv(os.path.join(out_dir, f"val_seed_{seed}.csv"), index=False)
    unseen_out.to_csv(os.path.join(out_dir, f"unseen_seed_{seed}.csv"), index=False)

    print(f"seed {seed}: train={len(train_out)} val={len(val_out)} unseen={len(unseen_out)} "
          f"(seen_ags={len(seen_ags)}, unseen_ags={len(unseen_ags)})")

print("\nDone.")
