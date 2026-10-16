"""
Data processing for AbLang-based Ab-Ag interaction prediction.

이 모듈은 캐시된 임베딩만 다룬다 — 토크나이징은 build_embeddings.py 에서 이미 끝났고
(embedding_cache.EsmEncoder / AbLangEncoder), 학습 때는 (서열, max_len) → (hidden, mask)
딕셔너리 조회만 한다. 그래서 여기에는 토크나이저가 아예 없다 (CDR_ESM150M 과 다른 점).

항원용 ESM2 토크나이저는 build_embeddings.py 가 직접 만든다.
"""

import torch
from torch.utils.data import Dataset


class AbAgCachedDataset(Dataset):
    """
    VH/VL/AG 가 전부 미리 계산된 frozen 인코더 hidden state 에서 온다
    (embedding_cache.EmbeddingCache.state_dict()). 순수 CPU 딕셔너리 조회라 GPU/모델
    접근이 없어 num_workers>0 에서도 안전하다.

    VH/VL 은 **AbLang 백본을 CDR attn_block에 통과시킨 post-CDR 출력**(768-dim)이다.
    CDR logits는 저장하지 않으며, 학습 시 attn_block을 다시 적용하지 않는다.
    AG 는 ESM2 출력(640-dim)이고 위에 얹은 레이어가 없어 캐시된 값을 그대로 쓴다.

    `embeddings` 에는 이 데이터셋이 요청할 (seq, max_len) 이 전부 들어 있어야 한다 —
    여기서 KeyError 가 나면 호출자가 prewarm/require 를 빠뜨린 것이다.

    DataFrame columns required:
        vh_col  (default "vh_seq")  : VH chain sequence
        vl_col  (default "vl_seq")  : VL chain sequence (use_vl=False 면 안 읽음)
        ag_col  (default "ag_seq")  : Antigen sequence
        label_col (default "label") : interaction label (0/1 for binary)
    """

    def __init__(
        self,
        df,
        embeddings: dict,
        vh_max_len:  int = 160,
        vl_max_len:  int = 160,
        ag_max_len:  int = 512,
        vh_col:      str = "vh_seq",
        vl_col:      str = "vl_seq",
        ag_col:      str = "ag_seq",
        label_col:   str = "label",
        binary:      bool = True,
        use_vl:      bool = True,
    ):
        # 나노바디(VHH) 데이터셋은 light chain이 없다. use_vl=False면 VL을 아예 쓰지 않는다 —
        # 빈 서열의 임베딩을 만들지도, 배치에 넣지도 않는다 (어차피 exclude_mask에서 전부
        # 제외되므로 결과는 같고, 쓸데없는 임베딩/forward만 사라진다).
        self.use_vl     = use_vl
        self.df         = df.reset_index(drop=True)
        self.embeddings = embeddings
        self.vh_max     = vh_max_len
        self.vl_max     = vl_max_len
        self.ag_max     = ag_max_len
        self.vh_col     = vh_col
        self.vl_col     = vl_col
        self.ag_col     = ag_col
        self.label_col  = label_col
        self.binary     = binary

    def __len__(self):
        return len(self.df)

    def _get(self, seq: str, max_len: int):
        key = (str(seq), max_len)
        try:
            return self.embeddings[key]
        except KeyError:
            raise KeyError(
                f"No cached embedding for seq[:20]={str(seq)[:20]!r} max_len={max_len}. "
                "build_embeddings.py 를 이 데이터셋에 대해 먼저 실행했는지 확인하세요.")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        vh_hidden, vh_mask = self._get(row[self.vh_col], self.vh_max)
        ag_emb,    ag_mask = self._get(row[self.ag_col], self.ag_max)

        label_val = float(row[self.label_col]) if self.binary \
                    else int(row[self.label_col])
        label = torch.tensor(label_val, dtype=torch.float if self.binary else torch.long)

        item = {
            "vh_hidden":    vh_hidden,
            "vh_attn_mask": vh_mask,
            "ag_emb":       ag_emb,
            "ag_attn_mask": ag_mask,
            "label":        label,
        }
        if self.use_vl:
            vl_hidden, vl_mask = self._get(row[self.vl_col], self.vl_max)
            item["vl_hidden"]    = vl_hidden
            item["vl_attn_mask"] = vl_mask
        return item
