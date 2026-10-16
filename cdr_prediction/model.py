"""
ABlang 기반 CDR 예측 모델 — 7-class region + VH/VL 체인 보조 헤드.

frozen ABlang (heavy/light 전용 백본, chain_type별 라우팅)
    → per-token hidden states (B, T, hidden_size)
    → AbLangStyleLayer (ABlang 자신의 EncoderBlock과 같은 구조, 새로 초기화 / trainable)
    → CDRHead → region_logits (B, T, 7)   [FR1/CDR1/FR2/CDR2/FR3/CDR3/FR4, CrossEntropyLoss]
    → ChainHead(AA 토큰 mean-pool) → chain_logits (B,)  [heavy=1/light=0, BCEWithLogitsLoss]

label: 0..6 = region,  -100 = '<'/'>'/pad (마스크)  (`model_OAS`와 동일한 라벨 체계)

`model_OAS`(ESM2 버전, 7-class + VH/VL) 대비 바뀐 것
------------------------------------------------------
* 인코더: ESM2(범용, 단일 모델) → ABlang. ABlang은 heavy/light를 **따로** 사전학습했기
  때문에(`ablang.pretrained(chain="heavy")` / `chain="light")` 가 서로 다른 체크포인트),
  ESM2처럼 하나의 frozen 모델을 그냥 공유할 수 없다. `chain="both"`로 학습할 때는 frozen
  백본을 heavy/light 각각 들고, 배치 안의 각 서열을 (OAS 데이터의 `chain_type` 컬럼 기준)
  맞는 백본에 태운다. 그 위에 새로 쌓는 trainable 레이어 1개, CDRHead, ChainHead는
  heavy/light가 공유한다.

* 새로 쌓는 레이어 구조: ESM2는 레이어 자신이 RoPE로 상대 위치를 본다. ABlang은 그와 달리
  **학습된 절대 위치 임베딩**을 입력 임베딩 단계에서만 한 번 더하고(`AbEmbeddings`), 그 뒤
  encoder 레이어들 자체(`ablang.encoderblocks.EncoderBlock`)는 위치 정보가 없는 순수
  post-LN softmax self-attention + FFN이다. 그래서 여기서 새로 쌓는 `AbLangStyleLayer`도
  ABlang의 EncoderBlock과 같은 구조(post-LN, 위치 인코딩 없음)로 새로 초기화한다.

* ChainHead는 사실 ABlang에서는 다소 잉여스러운 태스크다 — heavy/light를 맞히는 것 자체가
  이미 라우팅에 쓰이는 정답(`chain_types`)이기 때문이다(ESM2는 chain 정보 없이 서열만 보고
  맞혀야 해서 의미가 있었다). 그래도 `model_OAS`와 산출물/로그 포맷을 맞추기 위해 그대로
  두고, `hidden`은 chain_types로 라우팅된 값이 아니라 그 위에 새로 쌓은 레이어의 출력이라
  여전히 학습 신호로 의미가 있다(라우팅 정답을 그대로 복사하는 지름길이 없다는 뜻).

* hidden_size, num_attention_heads, intermediate_size, layer_norm_eps 등은 전부 다운로드된
  ABlang 체크포인트의 `hparams`에서 그대로 읽어온다(하드코딩하지 않음).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import ablang


class AbLangStyleLayer(nn.Module):
    """
    ABlang 자신의 `ablang.encoderblocks.EncoderBlock`과 구조가 같은 단일 트랜스포머 블록.

    post-LN self-attention + post-LN FFN, 각각 residual 포함. 위치 인코딩은 없다
    (ABlang 레이어 자체가 원래 위치 정보를 갖지 않고, 위치는 입력 임베딩 단계에서만 더해진다).

    ABlang 원본은 fairseq의 `MultiheadAttention`을 쓰지만(`encoderblocks.ThirdMultiHeadAttention`),
    같은 파일의 `OldMultiHeadAttention`이 보여주듯 표준 `nn.MultiheadAttention`으로도 같은
    구조를 만들 수 있어 여기서는 그것을 쓴다(추가 의존성 없이 구조만 그대로 복제).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float        = 0.1,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True,
        )
        self.attn_dropout    = nn.Dropout(dropout)
        self.attn_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.expand_dense = nn.Linear(hidden_size, intermediate_size)
        self.dense_dense  = nn.Linear(intermediate_size, hidden_size)
        self.ffn_dropout    = nn.Dropout(dropout)
        self.ffn_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, x: Tensor, key_padding_mask: Tensor = None, need_weights: bool = False):
        """
        x                : (B, T, H)
        key_padding_mask : (B, T) bool, True = 무시할 위치(pad)
        """
        attn_out, attn_w = self.self_attn(
            x, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        x = self.attn_layer_norm(x + self.attn_dropout(attn_out))

        ffn_out = self.dense_dense(F.gelu(self.expand_dense(x)))
        x = self.ffn_layer_norm(x + self.ffn_dropout(ffn_out))

        return (x, attn_w) if need_weights else x


class CDRHead(nn.Module):
    """Token-level region classification head. num_labels=7 → position별 7-class logit (CE)."""

    def __init__(self, hidden_size: int, num_labels: int = 7, dropout: float = 0.1):
        super().__init__()
        self.dense      = nn.Linear(hidden_size, hidden_size)
        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.out_proj   = nn.Linear(hidden_size, num_labels)

    def forward(self, x: Tensor) -> Tensor:
        x = F.gelu(self.dense(x))
        x = self.dropout(x)
        x = self.layer_norm(x)
        return self.out_proj(x)


class ChainHead(nn.Module):
    """서열 단위 VH/VL 분류 head.

    실제 아미노산 토큰('<'/'>'/pad 제외) 전체를 mean-pooling 한 뒤 이진 분류한다.
    logit 1개 → BCEWithLogitsLoss. heavy=1, light=0.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dense      = nn.Linear(hidden_size, hidden_size)
        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.out_proj   = nn.Linear(hidden_size, 1)

    def forward(self, hidden: Tensor, aa_mask: Tensor) -> Tensor:
        """hidden: (B, T, D),  aa_mask: (B, T) bool — 아미노산 토큰만 True"""
        m = aa_mask.unsqueeze(-1).to(hidden.dtype)            # (B, T, 1)
        pooled = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)   # (B, D)
        x = F.gelu(self.dense(pooled))
        x = self.dropout(x)
        x = self.layer_norm(x)
        return self.out_proj(x).squeeze(-1)                   # (B,)


class ABlangCDRModel(nn.Module):
    """
    frozen ABlang(heavy/light) 위에 7-class region head + VH/VL chain head를 얹은 모델.

    Parameters
    ----------
    chain            : "heavy" | "light" | "both". "both"면 frozen 백본을 heavy/light 둘 다
                        들고 있다가, forward 시 배치 원소별 `chain_types`로 라우팅한다.
    ablang_model_dir : `ablang.pretrained(model_folder=...)`에 넘기는 경로. "download"면
                        ablang 패키지 폴더 안에 자동 다운로드(체인당 한 번만, 이후 캐시 재사용).
    num_labels       : region head 출력 크기 (7 = FR1/CDR1/FR2/CDR2/FR3/CDR3/FR4, CrossEntropy)
    dropout          : head / 새 레이어의 dropout
    freeze_ablang    : ABlang 가중치 전부 freeze (작은 데이터셋에 권장)
    unfreeze_last_n  : freeze 상태에서 마지막 N개 ABlang EncoderBlock을 unfreeze (0=전체 frozen)
    num_attn_heads   : 새로 쌓는 레이어의 head 수. None이면 ABlang 자신의 num_attention_heads 사용
    use_chain_head   : 전체 mean-pooling → VH/VL head 사용 여부
    device           : ABlang 가중치를 로드할 디바이스. 학습용 GPU 이동은 Trainer가 따로 수행하므로
                        기본값 "cpu"로 두고 모델 전체를 한 번에 옮기는 편을 권장.

    Forward inputs
    --------------
    input_ids      : (B, T) long  — ABtokenizer 출력 ('<' 시작, AA..., '>' 끝, '-' 패딩)
    attention_mask : (B, T) long  — 1=token, 0=pad
    chain_types    : list[str] 길이 B, 각 원소 "heavy" | "light". chain="both"일 때 필수.
    aa_mask        : (B, T) bool — mean-pool 용. None이면 attention_mask에서 시작/끝 토큰을
                     뺀 값으로 근사한다.

    Returns
    -------
    region_logits : (B, T, num_labels)
    chain_logits  : (B,) 또는 None (use_chain_head=False 면 None)
    need_weights=True 이면 (region_logits, chain_logits, attn)
    """

    def __init__(
        self,
        chain: str            = "both",
        ablang_model_dir: str = "download",
        num_labels: int       = 7,
        dropout: float        = 0.1,
        freeze_ablang: bool   = True,
        unfreeze_last_n: int  = 0,
        num_attn_heads: int   = None,
        use_chain_head: bool  = True,
        device: str           = "cpu",
    ):
        super().__init__()
        if chain not in ("heavy", "light", "both"):
            raise ValueError(f"chain must be 'heavy', 'light' or 'both', got {chain!r}")
        self.chains = ["heavy", "light"] if chain == "both" else [chain]

        self.backbones  = nn.ModuleDict()
        self.tokenizers = {}
        hparams = None
        for c in self.chains:
            pt = ablang.pretrained(chain=c, model_folder=ablang_model_dir, device=device)
            self.backbones[c]  = pt.AbRep
            self.tokenizers[c] = pt.tokenizer
            if hparams is None:
                hparams = pt.hparams
            elif pt.hparams.hidden_size != hparams.hidden_size:
                raise ValueError(
                    f"heavy/light hidden_size mismatch: "
                    f"{hparams.hidden_size} vs {pt.hparams.hidden_size}"
                )

        self.hidden_size             = hparams.hidden_size
        self.max_position_embeddings = hparams.max_position_embeddings

        if freeze_ablang:
            for backbone in self.backbones.values():
                for p in backbone.parameters():
                    p.requires_grad = False
            print("[ABlangCDRModel] ABlang fully frozen")

            if unfreeze_last_n > 0:
                for backbone in self.backbones.values():
                    for layer in backbone.EncoderBlocks.Layers[-unfreeze_last_n:]:
                        for p in layer.parameters():
                            p.requires_grad = True
                print(f"[ABlangCDRModel] Unfroze last {unfreeze_last_n} ABlang EncoderBlocks (per chain)")
        else:
            print("[ABlangCDRModel] ABlang fully trainable")

        self.attn_block = AbLangStyleLayer(
            hidden_size       = self.hidden_size,
            num_heads         = num_attn_heads or hparams.num_attention_heads,
            intermediate_size = hparams.intermediate_size,
            dropout           = dropout,
            layer_norm_eps    = hparams.layer_norm_eps,
        )
        self.classifier = CDRHead(self.hidden_size, num_labels, dropout)
        self.chain_head = ChainHead(self.hidden_size, dropout) if use_chain_head else None

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"[ABlangCDRModel] backbones: {self.chains} | hidden_size={self.hidden_size} | "
              f"stacked layer heads={num_attn_heads or hparams.num_attention_heads}")
        print(f"[ABlangCDRModel] region head: {num_labels}-class token classification")
        print(f"[ABlangCDRModel] chain head : {'ON (전체 mean-pool → VH/VL)' if use_chain_head else 'OFF'}")
        print(f"[ABlangCDRModel] Trainable: {trainable:,} / {total:,} params")

    def _encode(self, input_ids: torch.Tensor, chain_types) -> Tensor:
        """chain_types에 맞는 frozen 백본으로 각 샘플을 라우팅해 (B, T, H) hidden state를 만든다."""
        if len(self.chains) == 1:
            return self.backbones[self.chains[0]](input_ids).last_hidden_states

        if chain_types is None:
            raise ValueError("chain='both' 모델은 forward에 chain_types가 필요합니다")

        per_chain = {}
        for c in self.chains:
            idx = [i for i, ct in enumerate(chain_types) if ct == c]
            if not idx:
                continue
            idx_t = torch.as_tensor(idx, device=input_ids.device, dtype=torch.long)
            out = self.backbones[c](input_ids.index_select(0, idx_t)).last_hidden_states
            per_chain[c] = (idx_t, out)

        B, T = input_ids.shape
        any_out = next(iter(per_chain.values()))[1]
        hidden = any_out.new_zeros(B, T, any_out.shape[-1])
        for idx_t, out in per_chain.values():
            hidden.index_copy_(0, idx_t, out)
        return hidden

    @torch.no_grad()
    def embed(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        chain_types=None,
        after_stacked_layer: bool = True,
    ) -> Tensor:
        """토큰별 임베딩 (B, T, H)만 꺼낸다 — head를 태우지 않는다.

        after_stacked_layer=True  : frozen ABlang → 학습된 AbLangStyleLayer 출력
                                    (= CDR 태스크로 학습된 표현. 다른 다운스트림에 넘길 때 기본값)
        after_stacked_layer=False : frozen ABlang 출력 그대로
                                    (= `ablang.pretrained(...)(seq, mode='rescoding')`와 같은 값.
                                       단 detach/numpy 변환이 없다)
        """
        hidden = self._encode(input_ids, chain_types)
        if not after_stacked_layer:
            return hidden

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        return self.attn_block(hidden, key_padding_mask)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        chain_types=None,
        aa_mask: torch.Tensor = None,
        need_weights: bool = False,
    ):
        hidden = self._encode(input_ids, chain_types)          # (B, T, H)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0             # True = 무시할 위치

        attn = None
        if need_weights:
            hidden, attn = self.attn_block(hidden, key_padding_mask, need_weights=True)
        else:
            hidden = self.attn_block(hidden, key_padding_mask)  # (B, T, H)

        region_logits = self.classifier(hidden)                 # (B, T, num_labels)

        chain_logits = None
        if self.chain_head is not None:
            if aa_mask is None:
                # aa_mask가 없으면 attention_mask에서 시작/끝 토큰을 빼서 근사한다
                aa_mask = attention_mask.bool().clone()
                aa_mask[:, 0] = False
                last = attention_mask.sum(dim=1) - 1
                aa_mask[torch.arange(aa_mask.size(0), device=aa_mask.device), last] = False
            chain_logits = self.chain_head(hidden, aa_mask)     # (B,)

        if need_weights:
            return region_logits, chain_logits, attn
        return region_logits, chain_logits
