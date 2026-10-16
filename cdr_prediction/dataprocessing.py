"""
ABlang 기반 CDR 예측 데이터 처리 — OAS_paired 데이터셋, 7-class region + VH/VL.

토큰/라벨 정렬 규약
--------------------
  pos 0        : '<' (시작) → label -100
  pos 1..L     : AA           → region label 0..6
  pos L+1      : '>' (끝)    → label -100
  pos L+2..max : '-' (pad)   → label -100

라벨 체계 (7-class, `model_OAS/dataprocessing.py`와 동일). 사용자 표기 1~7과 텐서 인덱스 0~6:

  사용자 표기   1     2     3     4     5     6     7
  텐서 인덱스   0     1     2     3     4     5     6
  이름         FR1  CDR1  FR2   CDR2  FR3   CDR3  FR4

CrossEntropyLoss 가 0-based 를 요구하므로 텐서에는 0..6 을 담고, 출력/리포트에는
REGION_NAMES 로 이름을 붙인다. 여기에 더해 서열 단위 chain 라벨(heavy=1, light=0)도 낸다.

model_esmlayer_oas 대비 바뀐 것
----------------------------------
1. 토크나이저가 `EsmTokenizer` 하나가 아니라 **heavy/light 전용 `ABtokenizer` 두 개**다.
   이 모듈은 토크나이저를 직접 만들지 않고, `ABlangCDRModel`이 이미 로드해 둔
   `model.tokenizers` ( {"heavy": ABtokenizer, "light": ABtokenizer} )를
   `ABlangCDRDataset`에 그대로 넘겨 받는다 — 체크포인트를 두 번 내려받거나 두 번
   메모리에 올리지 않기 위함이다.
2. 특수 토큰이 `<cls>`/`<eos>` 가 아니라 `'<'`/`'>'` 이다 (라벨 정렬 규약은 동일한 상대
   위치를 따른다: 시작/끝 토큰 -100, 나머지 AA에 region 라벨).
3. `collate_dynamic`도 heavy/light 로 pad_token_id 가 다를 수 있어(대부분 같지만 보장은
   없음) 토크나이저 dict를 받아 샘플마다 자기 chain의 pad_token으로 패딩한다.
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# 7-class region 라벨. 인덱스 i 가 사용자 표기 i+1 에 해당한다.
REGION_NAMES = ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]
NUM_REGIONS = len(REGION_NAMES)
# CDR 에 해당하는 인덱스 (CDR1, CDR2, CDR3) — binary 지표를 함께 뽑을 때 쓴다
CDR_REGION_IDS = [1, 3, 5]

CHAIN_TO_ID = {"heavy": 1, "light": 0}

# CSV 에서 실제로 쓰는 컬럼만 읽는다 (fold 당 150만 행 → 전체 컬럼은 낭비)
OAS_USECOLS = ["sample_id", "chain_type", "locus", "sequence", "seq_len",
               "mask", "quality_flag"]


def mask_to_regions(mask_str: str) -> list:
    """0/1 CDR 마스크 -> 7-class region 인덱스(0..6) 리스트.

    항체 V 도메인은 FR1-CDR1-FR2-CDR2-FR3-CDR3-FR4 순서가 고정이므로,
    마스크의 '1' 구간이 몇 번째인지만 세면 region 이 결정된다.

        FR1=0  CDR1=1  FR2=2  CDR2=3  FR3=4  CDR3=5  FR4=6
    """
    out = []
    cdr_seen = 0
    prev = "0"
    for ch in mask_str:
        if ch == "1":
            if prev != "1":
                cdr_seen += 1
            out.append(2 * cdr_seen - 1)
        else:
            out.append(2 * cdr_seen)
        prev = ch
    if out and max(out) > NUM_REGIONS - 1:
        raise ValueError(
            f"CDR 구간이 3개를 넘습니다 (region index {max(out)}): {mask_str[:60]}...")
    return out


# ── OAS_paired 로더 (model_esmlayer_oas와 동일) ──────────────────────────────

def _read_oas_csv(path: str, usecols=None) -> pd.DataFrame:
    """OAS CSV 를 안전하게 읽는다 ("NA" 서열 보존, 전부 문자열로)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"OAS fold 파일이 없습니다: {path}")
    return pd.read_csv(
        path,
        usecols=usecols,
        dtype=str,
        keep_default_na=False,
        na_values=[],
    )


def load_oas_fold(
    dataset_dir: str,
    chain: str,
    fold: int,
    split: str,
    quality_filter: str = "ok",
    dedup: bool = False,
    max_samples: int = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    dataset_dir    : .../data/OAS_paired/dataset
    chain          : "heavy" | "light" | "both"
    fold           : 0..4
    split          : "train" | "valid" | "test"
    quality_filter : quality_flag 가 이 값인 행만 사용 (None = 전체)
    dedup          : (sequence, mask) 중복 제거
    max_samples    : 이 개수로 랜덤 서브샘플 (None = 전체)
    """
    chains = ["heavy", "light"] if chain == "both" else [chain]
    if any(c not in ("heavy", "light") for c in chains):
        raise ValueError(f"chain must be 'heavy', 'light' or 'both', got {chain!r}")

    frames = []
    for c in chains:
        path = os.path.join(dataset_dir, c, f"fold_{fold}", f"{split}.csv.gz")
        df = _read_oas_csv(path, usecols=OAS_USECOLS)
        print(f"  [load] {path} → {len(df):,} rows")
        frames.append(df)

    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    if quality_filter is not None and "quality_flag" in df.columns:
        before = len(df)
        df = df[df["quality_flag"] == quality_filter]
        if before != len(df):
            print(f"  [filter] quality_flag=={quality_filter!r}: {before:,} → {len(df):,}")

    if dedup:
        before = len(df)
        df = df.drop_duplicates(subset=["sequence", "mask"])
        if before != len(df):
            print(f"  [dedup] {before:,} → {len(df):,}")

    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed)
        print(f"  [subsample] → {len(df):,}")

    return df.reset_index(drop=True)


# ── 인코딩 ───────────────────────────────────────────────────────────────────

def ablang_cdr_encode(
    seq: str,
    mask_str: str,
    tokenizer,
    max_len: int      = 160,
    chain_type: str   = None,
    pad_to_max: bool  = True,
) -> dict:
    """
    Parameters
    ----------
    seq        : amino acid sequence string (single-letter, no spaces)
    mask_str   : CDR mask string of same length as seq, e.g. "000111000"
    tokenizer  : ablang.tokenizers.ABtokenizer (해당 chain 전용)
    max_len    : total token length including '<' and '>'
    chain_type : "heavy" | "light" (None 이면 chain_label = -100 → 손실에서 제외)
    pad_to_max : False 면 패딩 없이 실제 길이만 낸다 (collate_dynamic과 짝)

    Returns
    -------
    dict with:
        input_ids      : (L,) or (max_len,) long
        attention_mask : (L,) or (max_len,) long
        labels         : 같은 길이 long — region 0..6, 특수/패딩은 -100
        aa_mask        : 같은 길이 bool — 실제 아미노산 토큰만 True (mean-pool 용)
        chain_label    : () long — heavy=1, light=0, 미상=-100
        sequence       : str  (truncated)
        true_mask      : str  (truncated)
    """
    aa_tokens = list(str(seq).strip().upper())
    mask_list = list(str(mask_str).strip())

    if len(mask_list) < len(aa_tokens):
        raise ValueError(
            f"mask가 sequence보다 짧습니다: seq={len(aa_tokens)}, mask={len(mask_list)}"
        )
    mask_list = mask_list[:len(aa_tokens)]

    # truncate to fit inside max_len ('<' + AA... + '>')
    max_aa = max_len - 2
    aa_tokens = aa_tokens[:max_aa]
    mask_list = mask_list[:max_aa]
    seq_str = "".join(aa_tokens)

    ids = tokenizer.encode(seq_str)                     # (L+2,) long — '<', AA..., '>'
    pad_id = tokenizer.pad_token

    if pad_to_max:
        pad_len = max_len - ids.shape[0]
        input_ids = torch.cat([ids, ids.new_full((pad_len,), pad_id)]) if pad_len > 0 else ids[:max_len]
    else:
        input_ids = ids
    attention_mask = (input_ids != pad_id).long()

    # Labels: -100 for '<', region 0..6 for AAs, -100 for '>' and pad
    regions = mask_to_regions("".join(mask_list))
    labels  = [-100] + regions + [-100]
    if pad_to_max:
        labels += [-100] * (max_len - len(labels))
        labels  = labels[:max_len]
    labels_t = torch.tensor(labels, dtype=torch.long)

    # 실제 아미노산 토큰 위치만 True — '<'/'>'/pad 를 뺀 mean-pooling 에 쓴다
    aa_mask = labels_t != -100

    chain_label = CHAIN_TO_ID.get(chain_type, -100) if chain_type else -100

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels_t,
        "aa_mask":        aa_mask,
        "chain_label":    torch.tensor(chain_label, dtype=torch.long),
        "sequence":       seq_str,
        "true_mask":      "".join(mask_list),
    }


class ABlangCDRDataset(Dataset):
    """
    Dataset for ABlang-based CDR prediction (7-class region + VH/VL).

    DataFrame columns required:
        seq_col   (default "sequence")   : amino acid sequence
        mask_col  (default "mask")       : CDR mask string (0/1)
        chain_col (default "chain_type") : "heavy" | "light" — 토크나이저/라우팅/chain 라벨에 사용

    tokenizers : {"heavy": ABtokenizer, "light": ABtokenizer} — `ABlangCDRModel`이 이미
                 로드해 둔 것을 그대로 받는다 (여기서 새로 다운로드/로드하지 않는다).

    __init__ 에서 필요한 컬럼만 numpy bytes 배열로 복사하고 DataFrame 참조는 버린다
    (fold 하나가 150만 행이라 num_workers>0 로 fork 할 때 copy-on-write 복제를 피하기 위함).
    """

    def __init__(
        self,
        df,
        tokenizers: dict,
        max_len: int    = 160,
        seq_col: str    = "sequence",
        mask_col: str   = "mask",
        chain_col: str  = "chain_type",
        id_col: str     = None,
        pad_to_max: bool = True,
    ):
        for col in [seq_col, mask_col, chain_col]:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        self.max_len    = max_len
        self.pad_to_max = pad_to_max
        self.tokenizers = tokenizers
        # 고정폭 bytes ('S') — 워커 fork 시 copy-on-write 로 실제 공유된다
        self.seqs   = df[seq_col].to_numpy(dtype="S").astype("S")
        self.masks  = df[mask_col].to_numpy(dtype="S").astype("S")
        self.chains = df[chain_col].to_numpy(dtype="S").astype("S")
        if id_col and id_col in df.columns:
            self.ids = df[id_col].to_numpy(dtype="S").astype("S")
        else:
            self.ids = None

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq      = self.seqs[idx].decode()
        mask_str = self.masks[idx].decode()
        chain    = self.chains[idx].decode()

        item = ablang_cdr_encode(seq, mask_str, self.tokenizers[chain], self.max_len,
                                 chain_type=chain, pad_to_max=self.pad_to_max)
        item["chain_type"] = chain
        item["seq_id"] = self.ids[idx].decode() if self.ids is not None else str(idx)
        return item


def collate_dynamic(batch, tokenizers: dict):
    """배치 안에서 가장 긴 서열까지만 패딩하는 collate_fn.

    고정 max_len 패딩은 실제 평균 길이보다 훨씬 길어 계산 대부분을 패딩 토큰에 쓴다
    (model_OAS/dataprocessing.py의 collate_dynamic과 같은 이유). heavy/light 는 서로
    다른 ABtokenizer 인스턴스라 pad_token_id 가 다를 수 있으므로, 샘플마다 자기 chain의
    pad_token으로 패딩한다 (전역 상수 하나를 가정하지 않는다).
    """
    T = max(x["input_ids"].size(0) for x in batch)
    B = len(batch)

    input_ids      = torch.zeros((B, T), dtype=torch.long)   # 아래서 chain별 pad_id로 채움
    attention_mask = torch.zeros((B, T), dtype=torch.long)
    labels         = torch.full((B, T), -100, dtype=torch.long)
    aa_mask        = torch.zeros((B, T), dtype=torch.bool)

    for i, x in enumerate(batch):
        pad_id = tokenizers[x["chain_type"]].pad_token
        input_ids[i, :] = pad_id
        L = x["input_ids"].size(0)
        input_ids[i, :L]      = x["input_ids"]
        attention_mask[i, :L] = x["attention_mask"]
        labels[i, :L]         = x["labels"]
        aa_mask[i, :L]        = x["aa_mask"]

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
        "aa_mask":        aa_mask,
        "chain_label":    torch.stack([x["chain_label"] for x in batch]),
        "sequence":       [x["sequence"]   for x in batch],
        "true_mask":      [x["true_mask"]  for x in batch],
        "seq_id":         [x["seq_id"]     for x in batch],
        "chain_type":     [x["chain_type"] for x in batch],
    }
