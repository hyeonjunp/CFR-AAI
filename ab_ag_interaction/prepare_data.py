"""
Ab-Ag 데이터 로더

각 데이터셋의 기존 split 파일(train/val/test)을 읽어
antibody.csv + antigen.csv 와 join하여
vh_seq, vl_seq, ag_seq, label DataFrame을 반환.

데이터셋별 split 구조:
    SAbDab    : data/SAbDab/kfold/train_fold_N.csv  / test_fold_N.csv        (N=0~4)
    HIV       : data/HIV/{split_type}/train_seed_N.csv / val_seed_N.csv / unseen_seed_N.csv
                split_type ∈ {unseen, ag_unseen, ab_unseen}
    CoVAbDab  : data/CoVAbDab/transfer/train_seed_N.csv / val_seed_N.csv / unseen_seed_N.csv
    il6       : data/il6/ag_unseen/train_seed_N.csv / val_seed_N.csv / unseen_seed_N.csv
                나노바디(VHH) — light chain 없음, vl_seq는 빈 문자열(모델에서 자동으로 마스킹되어 무시됨)
"""

import os
import pandas as pd

# <repo>/data/ab_ag — resolved from THIS file's location, not the CWD, so
# `python train.py` works from any directory and survives moving the tree off
# the old /HDD1 mount.  config.yaml's data.data_root overrides it.
DATA_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "ab_ag")
)

# 데이터셋 키와 실제 폴더명이 다른 경우만 등록한다 (없으면 키를 그대로 폴더명으로 쓴다).
DATASET_DIRS = {
    "AVIDa_SARS_CoV_2": "AVIDa-SARS-CoV-2",
    # hIL6 는 데이터셋 키와 폴더명이 둘 다 "il6" 라 매핑이 필요 없다.
    # (full VHH 버전 = 중앙값 124. 구버전 CDR concat = 46 은 "ㅁㄴㅇㅁ" 폴더에 남아있다.)
}

# 나노바디(VHH) 데이터셋 — light chain이 없다. 모델/데이터셋을 use_vl=False로
# 만들어 VL stream 자체를 빼야 한다 (빈 서열을 넣으면 p_vl이 상수 0이 되어
# vl_iip와 classifier의 p_vl 구간이 학습되지 않는다).
VHH_DATASETS = {"il6", "AVIDa_SARS_CoV_2"}


def dataset_has_vl(dataset: str) -> bool:
    """이 데이터셋에 light chain이 있는가."""
    return dataset not in VHH_DATASETS


# ── AB / AG 마스터 테이블 로더 ─────────────────────────────────────────────────

def _load_ab_ag_master(dataset: str, data_root: str):
    """antibody.csv + antigen.csv를 읽어 (ab_df, ag_df) 반환."""
    base = os.path.join(data_root, DATASET_DIRS.get(dataset, dataset))
    ab = pd.read_csv(os.path.join(base, "antibody.csv"))
    ag = pd.read_csv(os.path.join(base, "antigen.csv"))
    return ab, ag


def _join(pairs: pd.DataFrame, ab: pd.DataFrame, ag: pd.DataFrame,
          dataset: str) -> pd.DataFrame:
    """pairs(ab_id, ag_id, label) + ab + ag → vh_seq, vl_seq, ag_seq, label"""

    if dataset == "AVIDa_SARS_CoV_2":
        # The master tables and split CSVs use different pandas-inferred dtypes
        # (e.g. 0 vs "ab_0"). IDs are opaque keys, so compare them as strings.
        pairs["ab_id"] = pairs["ab_id"].astype(str).str.strip()
        pairs["ag_id"] = pairs["ag_id"].astype(str).str.strip()
        ab["ab_id"] = "ab_" + ab["ab_id"].astype(str).str.strip().str.removeprefix("ab_")
        ag["ag_id"] = "ag_" + ag["ag_id"].astype(str).str.strip().str.removeprefix("ag_")

    if dataset == "SAbDab":
        # ab: ab_id, H, L, ...  /  ag: ag_id, ag_seq, ...
        df = pairs.merge(ab[["ab_id", "H", "L"]], on="ab_id", how="inner")
        df = df.merge(ag[["ag_id", "ag_seq"]], on="ag_id", how="inner")
        df = df.rename(columns={"H": "vh_seq", "L": "vl_seq"})

    elif dataset in ("HIV", "CoVAbDab"):
        # ab: ab_id(=row index), heavy, light, ...  /  ag: ag_id(=row index), ag_seq, ...
        ab = ab.reset_index().rename(columns={"index": "ab_id"})
        ag = ag.reset_index().rename(columns={"index": "ag_id"})
        df = pairs.merge(ab[["ab_id", "heavy", "light"]], on="ab_id", how="inner")
        df = df.merge(ag[["ag_id", "ag_seq"]], on="ag_id", how="inner")
        df = df.rename(columns={"heavy": "vh_seq", "light": "vl_seq"})

    elif dataset in ("il6", "AVIDa_SARS_CoV_2"):
        # (AVIDa_SARS_CoV_2의 antigen.csv에는 ag_label 컬럼도 있지만 여기서는 쓰지 않는다)
        # 나노바디(VHH): light chain 없음. ab: ab_id, ab_seq  /  ag: ag_id, ag_seq
        df = pairs.merge(ab[["ab_id", "ab_seq"]], on="ab_id", how="inner")
        df = df.merge(ag[["ag_id", "ag_seq"]], on="ag_id", how="inner")
        df = df.rename(columns={"ab_seq": "vh_seq"})
        df["vl_seq"] = ""   # sentinel — AbAgESMDataset/모델에서 빈 서열은 자동으로 마스킹되어 무시됨

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # 공통 정제
    has_vl = dataset_has_vl(dataset)
    for col in ["vh_seq", "vl_seq", "ag_seq"]:
        df[col] = df[col].astype(str).str.strip()

    mask = (
        df["vh_seq"].str.len().gt(4) &
        df["ag_seq"].str.len().gt(4) &
        ~df["vh_seq"].eq("nan") &
        ~df["ag_seq"].eq("nan")
    )
    if has_vl:
        mask &= df["vl_seq"].str.len().gt(4) & ~df["vl_seq"].eq("nan")
    df = df[mask].reset_index(drop=True)

    return df[["vh_seq", "vl_seq", "ag_seq", "label"]]


def _read_split(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # ab_id가 없으면 row index로 생성
    if "ab_id" not in df.columns:
        df["ab_id"] = df.index
    if "ag_id" not in df.columns:
        df["ag_id"] = df.index
    return df


# ── Public API ─────────────────────────────────────────────────────────────────

def load_sabdab_fold(fold: int, data_root: str = DATA_ROOT):
    """
    SAbDab kfold 분할.
    train_fold_N / test_fold_N (val = None)
    Returns: (df_train, df_val=None, df_test)
    """
    base = os.path.join(data_root, "SAbDab", "kfold")
    ab, ag = _load_ab_ag_master("SAbDab", data_root)

    train = _join(_read_split(os.path.join(base, f"train_fold_{fold}.csv")), ab, ag, "SAbDab")
    test  = _join(_read_split(os.path.join(base, f"test_fold_{fold}.csv")),  ab, ag, "SAbDab")
    return train, None, test


def load_hiv_seed(seed: int, split_type: str = "unseen", data_root: str = DATA_ROOT):
    """
    HIV seed 분할.
    split_type ∈ {unseen, ag_unseen, ab_unseen}
    Returns: (df_train, df_val, df_test)
    """
    base = os.path.join(data_root, "HIV", split_type)
    ab, ag = _load_ab_ag_master("HIV", data_root)

    train = _join(_read_split(os.path.join(base, f"train_seed_{seed}.csv")),  ab, ag, "HIV")
    val   = _join(_read_split(os.path.join(base, f"val_seed_{seed}.csv")),    ab, ag, "HIV")
    test  = _join(_read_split(os.path.join(base, f"unseen_seed_{seed}.csv")), ab, ag, "HIV")
    return train, val, test


def load_covabdab_seed(seed: int, data_root: str = DATA_ROOT):
    """
    CoVAbDab transfer 분할.
    Returns: (df_train, df_val, df_test)
    """
    base = os.path.join(data_root, "CoVAbDab", "transfer")
    ab, ag = _load_ab_ag_master("CoVAbDab", data_root)

    train = _join(_read_split(os.path.join(base, f"train_seed_{seed}.csv")),  ab, ag, "CoVAbDab")
    val   = _join(_read_split(os.path.join(base, f"val_seed_{seed}.csv")),    ab, ag, "CoVAbDab")
    test  = _join(_read_split(os.path.join(base, f"unseen_seed_{seed}.csv")), ab, ag, "CoVAbDab")
    return train, val, test


def load_avida_seed(seed: int, data_root: str = DATA_ROOT):
    """
    il6 ag_unseen 분할 (나노바디, light chain 없음).
    Returns: (df_train, df_val, df_test)
    """
    base = os.path.join(data_root, DATASET_DIRS.get("il6", "il6"), "ag_unseen")
    ab, ag = _load_ab_ag_master("il6", data_root)

    train = _join(_read_split(os.path.join(base, f"train_seed_{seed}.csv")),  ab, ag, "il6")
    val   = _join(_read_split(os.path.join(base, f"val_seed_{seed}.csv")),    ab, ag, "il6")
    test  = _join(_read_split(os.path.join(base, f"unseen_seed_{seed}.csv")), ab, ag, "il6")
    return train, val, test


def load_avida_sars_cov2_seed(seed: int, data_root: str = DATA_ROOT):
    """
    AVIDa-SARS-CoV-2 5-fold 분할 (나노바디, light chain 없음).

    train/val 은 원본 train 파일을 (Ag_label, label) 층화 5-fold 로 나눈 것이고,
    test 는 원본 공식 홀드아웃이라 **seed 0~4 에서 모두 동일한 파일 내용**이다.
    따라서 seed 간 test 성능 차이는 split 차이가 아니라 학습 분산만 반영한다
    (HIV 처럼 split seed 가 아니다 — 해석할 때 주의).

    주의: split 파일의 컬럼 **순서**가 train/val (label, ab_id, ag_id, Ag_label) 과
    test (label, Ag_label, ab_id, ag_id) 에서 다르다. pandas 가 헤더로 읽으므로
    문제는 없지만, awk 등으로 위치 기반 파싱을 하면 틀린다.

    Returns: (df_train, df_val, df_test)
    """
    base = os.path.join(data_root, DATASET_DIRS["AVIDa_SARS_CoV_2"],
                        "AVIDa_SARS_CoV_2_5fold")
    ab, ag = _load_ab_ag_master("AVIDa_SARS_CoV_2", data_root)

    ds = "AVIDa_SARS_CoV_2"
    train = _join(_read_split(os.path.join(base, f"train_seed_{seed}.csv")), ab, ag, ds)
    val   = _join(_read_split(os.path.join(base, f"val_seed_{seed}.csv")),   ab, ag, ds)
    test  = _join(_read_split(os.path.join(base, f"test_seed_{seed}.csv")),  ab, ag, ds)
    return train, val, test


# dataset 이름 → 로더 함수 매핑 (train.py에서 import)
DATASET_LOADERS = {
    "SAbDab":     load_sabdab_fold,
    "HIV":        load_hiv_seed,
    "CoVAbDab":   load_covabdab_seed,
    "il6": load_avida_seed,
    "AVIDa_SARS_CoV_2": load_avida_sars_cov2_seed,
}

N_SPLITS = 5   # 모든 데이터셋 공통 (seed 0~4 또는 fold 0~4)


# ── CLI: 병합 CSV 저장 ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=DATA_ROOT)
    parser.add_argument("--out_csv",   default="../data/abag_all.csv")
    args = parser.parse_args()

    rows = []
    for ds, loader in DATASET_LOADERS.items():
        print(f"{ds}...", end=" ", flush=True)
        try:
            train, val, test = loader(0, data_root=args.data_root) if ds != "SAbDab" else loader(0, data_root=args.data_root)
            for df in [train, val, test]:
                if df is not None:
                    df["dataset"] = ds
                    rows.append(df)
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")

    merged = pd.concat(rows, ignore_index=True)
    print(f"\nTotal: {len(merged)}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    print(f"Saved → {args.out_csv}")
