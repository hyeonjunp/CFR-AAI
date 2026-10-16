"""
data/AVIDa-SARS-CoV-2 데이터셋 빌더 (il6/build_il6_dataset.py 와 동일 패턴)

antibody.csv (ab_id="ab_N", ab_seq), antigen.csv (ag_id="ag_N", ag_label, ag_seq),
ab_ag_pairs.csv (label, Ag_label, ab_id, ag_id) 는 이미 만들어져 있다(PelB/His 태그
안 뗀 원본 VHH). 여기서는:

  1. ANARCI IMGT 넘버링으로 CDR 추출 불가 VHH 제거 (ab_id 를 0..N-1 로 재부여)
  2. ab_id/ag_id 를 "ab_0" 문자열에서 antibody.csv/antigen.csv 의 순서를 가리키는
     0-based 정수로 변환 (DeepInterAware PairDataset 이 .iloc[ab_id] 로 접근하므로 필수)
  3. ag_unseen 분할: WT_cell 은 항상 seen, 나머지 16개 항원을 seed 별로 8/8 seen/unseen

출력: antibody.csv(ab_id,ab_seq,ab_cdr), antigen.csv(ag_id,ag_seq,Ag_label),
      ab_ag_pair.csv(label,Ag_label,ab_id,ag_id),
      ag_unseen/{train,val,unseen}_seed_{0..4}.csv
"""
import argparse
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "il6"))
from extract_imgt_cdr import extract_cdrs

WT_LABEL = "WT_cell"
N_SEEDS = 5
AG_SEEN_FRAC = 0.5
VAL_FRAC = 0.1

parser = argparse.ArgumentParser()
parser.add_argument("--ncpu", type=int, default=os.cpu_count() or 1)
args = parser.parse_args()

ab_raw = pd.read_csv(os.path.join(HERE, "antibody.csv"))
ag_raw = pd.read_csv(os.path.join(HERE, "antigen.csv"))
pair_raw = pd.read_csv(os.path.join(HERE, "ab_ag_pairs.csv"))
print("raw antibody:", len(ab_raw), "raw antigen:", len(ag_raw), "raw pairs:", len(pair_raw))

# ── ANARCI CDR 추출 ──────────────────────────────────────────────────────────
print("\nANARCI IMGT 넘버링으로 CDR 추출 가능 여부 확인:")
cdr_rows = extract_cdrs(list(zip(ab_raw["ab_id"], ab_raw["ab_seq"])), ncpu=args.ncpu)
cdr_df = pd.DataFrame(cdr_rows)
cdr_df["ab_cdr"] = cdr_df["cdr1"] + cdr_df["cdr2"] + cdr_df["cdr3"]

status = cdr_df["status"]
counts = status.value_counts()
for st in ("no_domain", "missing_cdr"):
    if counts.get(st):
        print(f"  {st} 제거: {counts[st]}")
ok = cdr_df[status == "ok"].reset_index(drop=True)
print(f"  VHH: {len(ab_raw)} -> {len(ok)} ({(len(ab_raw) - len(ok)) / len(ab_raw):.2%} 제거)")
print(f"  ab_cdr length min/median/max: "
      f"{ok.ab_cdr.str.len().min()}/{int(ok.ab_cdr.str.len().median())}/{ok.ab_cdr.str.len().max()}")

# ── antibody.csv (ab_id: 0..N-1 정수, 기존 순서 유지) ────────────────────────
ab_df = ok[["ab_id", "ab_seq", "ab_cdr"]].reset_index(drop=True)
old_to_new_ab = {old: new for new, old in enumerate(ab_df["ab_id"])}
ab_df["ab_id"] = range(len(ab_df))
print("\nantibody.csv:", len(ab_df))

# ── antigen.csv (ag_id: 0..M-1 정수) ─────────────────────────────────────────
ag_df = ag_raw.rename(columns={"ag_label": "Ag_label"})[["ag_id", "ag_seq", "Ag_label"]].reset_index(drop=True)
old_to_new_ag = {old: new for new, old in enumerate(ag_df["ag_id"])}
ag_df["ag_id"] = range(len(ag_df))
print("antigen.csv:", len(ag_df))

# ── ab_ag_pair.csv (정수 id로 remap, 제거된 항체 참조 pair는 버림) ───────────
pair_df = pair_raw.copy()
pair_df["ab_id"] = pair_df["ab_id"].map(old_to_new_ab)
pair_df["ag_id"] = pair_df["ag_id"].map(old_to_new_ag)
n_orphan = int(pair_df["ab_id"].isna().sum())
if n_orphan:
    orphan_labels = pair_df.loc[pair_df["ab_id"].isna(), "label"].value_counts().to_dict()
    pair_df = pair_df[pair_df["ab_id"].notna()].reset_index(drop=True)
    print(f"  손상 VHH 참조로 제거된 pair: {n_orphan} (label별 {orphan_labels})")
pair_df["ab_id"] = pair_df["ab_id"].astype(int)
pair_df["ag_id"] = pair_df["ag_id"].astype(int)
pair_df = pair_df[["label", "Ag_label", "ab_id", "ag_id"]]

n_before = len(pair_df)
grp = pair_df.groupby(["ab_id", "ag_id"])["label"]
n_dup_combo = int((grp.size() > 1).sum())
conflict = grp.nunique() > 1
conflict_keys = conflict[conflict].index
if len(conflict_keys):
    keys = pd.MultiIndex.from_frame(pair_df[["ab_id", "ag_id"]])
    pair_df = pair_df[~keys.isin(conflict_keys)]
pair_df = pair_df.drop_duplicates(subset=["ab_id", "ag_id"]).reset_index(drop=True)
print(f"  중복 (ab_id, ag_id) 조합: {n_dup_combo} | 충돌 조합: {len(conflict_keys)}")
print(f"ab_ag_pair.csv: {n_before} -> {len(pair_df)}")

ab_df.to_csv(os.path.join(HERE, "antibody.csv"), index=False)
ag_df.to_csv(os.path.join(HERE, "antigen.csv"), index=False)
pair_df.to_csv(os.path.join(HERE, "ab_ag_pair.csv"), index=False)
print("Saved antibody.csv / antigen.csv / ab_ag_pair.csv")

# ── ag_unseen splits ──────────────────────────────────────────────────────
out_dir = os.path.join(HERE, "ag_unseen")
os.makedirs(out_dir, exist_ok=True)

mutant_labels = sorted(l for l in pair_df["Ag_label"].unique() if l != WT_LABEL)
print(f"\n{len(mutant_labels)} non-WT antigens (+ {WT_LABEL} always seen)")

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

    with_select_index(train_df).to_csv(os.path.join(out_dir, f"train_seed_{seed}.csv"), index=False)
    with_select_index(val_df).to_csv(os.path.join(out_dir, f"val_seed_{seed}.csv"), index=False)
    with_select_index(unseen_df).to_csv(os.path.join(out_dir, f"unseen_seed_{seed}.csv"), index=False)

    print(f"seed {seed}: train={len(train_df)} val={len(val_df)} unseen={len(unseen_df)} "
          f"(seen_ags={len(seen_ags)}, unseen_ags={len(unseen_ags)})")

print("\nDone.")
