# CDR_Ablang_BAN 입력 파이프라인 — 원본 서열부터 `model.forward()`까지

`model.py`(`AbLangAbAgModel`)와 `ban.py`(`BANLayer`) 자체의 구조는 `MODEL.md`(V7)에
정리되어 있다. 이 문서는 그 앞 단계, 즉 **원본 항체/항원 서열이
어떤 과정을 거쳐 `model.forward()`의 6개 텐서 인자가 되는지**를 순서대로 정리한다.

```text
data/<dataset>/{antibody,antigen}.csv + split csv     (prepare_data.py)
        │  join
        ▼
DataFrame[vh_seq, vl_seq, ag_seq, label]
        │
        │  1회, GPU 필요(build_embeddings.py)
        ▼
frozen AbLang(VH/VL) / frozen ESM2(AG) 인코딩
        │
        ▼
emb/<dataset>/{vh,vl,ag}_emb.npz   (디스크 캐시, float16)
        │
        │  학습 때는 읽기만 함 (train.py → EmbeddingCache.require + state_dict)
        ▼
AbAgCachedDataset[idx] → {vh_hidden, vh_attn_mask, vl_hidden, vl_attn_mask,
                          ag_emb, ag_attn_mask, label}
        │  DataLoader가 배치로 묶음 (B, T, D) / (B, T)
        ▼
AbLangAbAgModel.forward(vh_hidden, vh_attn_mask,
                        vl_hidden, vl_attn_mask,
                        ag_emb, ag_attn_mask)
```

---

## 1. 원본 데이터 로딩 — `prepare_data.py`

데이터셋마다 `antibody.csv` + `antigen.csv`(마스터 테이블)와 train/val/test split csv
(ab_id, ag_id, label)가 따로 있고, `_join()`이 둘을 합쳐 공통 스키마로 통일한다.

| dataset | split 구조 | 비고 |
|---|---|---|
| `SAbDab` | `kfold/train_fold_N.csv` / `test_fold_N.csv` | val 없음 |
| `HIV` | `{split_type}/train,val,unseen_seed_N.csv` | split_type ∈ {unseen, ag_unseen, ab_unseen} |
| `CoVAbDab` | `transfer/train,val,unseen_seed_N.csv` | |
| `il6` | `ag_unseen/train,val,unseen_seed_N.csv` | 나노바디(VHH), light 없음 |
| `AVIDa_SARS_CoV_2` | `AVIDa_SARS_CoV_2_5fold/train,val,test_seed_N.csv` | 나노바디(VHH), light 없음 |

출력은 항상 `DataFrame[vh_seq, vl_seq, ag_seq, label]` 4컬럼이다. 나노바디 데이터셋은
`vl_seq = ""` (빈 문자열)로 채워 넣는 대신, `dataset_has_vl(dataset) == False`일 때
모델/데이터셋 쪽에서 **VL 스트림 자체를 빼버린다** (`use_vl=False`) — 빈 서열의 임베딩을
만들거나 forward에 넣지 않는다. 정제 단계에서 서열 길이 4 이하이거나 `"nan"`인 행은
버린다.

---

## 2. Frozen 인코더로 임베딩 사전 계산 — `build_embeddings.py`, `embedding_cache.py`

학습 전에 **한 번만** GPU로 실행해서 결과를 npz에 저장해 둔다. 이후 `train.py`는
인코더를 전혀 로드하지 않고 캐시만 읽는다.

체인(role)마다 인코더가 다르다:

| role | 인코더 | 출력 차원 | max_len 한도 |
|---|---|---|---|
| `vh` | AbLang **heavy** `AbRep` (frozen) → CDR checkpoint의 `attn_block` (frozen) | 768 | ≤160 (실토큰 ≤157) |
| `vl` | AbLang **light** `AbRep` (frozen) → 같은 방식의 `attn_block` (frozen) | 768 | ≤160 (실토큰 ≤157) |
| `ag` | ESM2 `facebook/esm2_t30_150M_UR50D` (frozen) | 640 | config의 `ag_max_len` (예: 1024) |

- **왜 항체/항원 인코더가 다른가**: AbLang의 vocab·위치임베딩은 항체 V-도메인 전용이라
  항원 서열을 표현하지 못한다. 그래서 항원은 범용 단백질 언어모델(ESM2)로 인코딩한다.
- **왜 VH/VL npz에 "post-CDR" 값이 저장되는가**: 여기 저장되는 VH/VL 표현은 AbLang
  backbone 출력을 그대로가 아니라, `CDR_prediction`에서 학습한 CDR checkpoint
  (`pretrained.cdr_ckpt`)의 `attn_block.*`(frozen `AbLangStyleLayer`)까지 통과시킨
  결과다. `model.py`는 forward 중에 이 레이어를 다시 적용하지 않는다 — 이미 끝난
  전처리이기 때문이다.
- **토큰 배치**: ESM/AbLang 모두 `pos 0 = 시작 토큰(<cls> 또는 '<')`,
  `pos 1..L = 아미노산`, `pos L+1 = 끝 토큰(<eos> 또는 '>')`, 그 뒤는 padding으로 동일한
  레이아웃을 쓴다. 그래서 `model.py`의 `special_token_mask`/`exclude_mask`가 두 인코더
  출력에 똑같이 적용된다.
- **AbLang 제약 처리** (`AbLangEncoder._clean`): vocab 밖 문자(X 등)는 길이를 보존하기
  위해 삭제 대신 마스크 문자 `'*'`로 치환하고, `min(max_len-2, 157)` 잔기를 넘으면
  N-말단부터 157잔기까지만 남기고 뒤를 자른다 (V 도메인 FR1~FR4/CDR1~3은 N-말단
  ~120잔기 안에 다 들어있고, 그 뒤는 불변부위라 AbLang/CDR 모델 모두 학습한 적 없는
  분포이므로 잘라도 안전하다는 전제).

### 캐시 파일 포맷 (`emb/<dataset>/{vh,vl,ag}_emb.npz`)

```text
"__max_len__"                 -> 이 파일을 만들 때 쓴 max_len (int32)
"__representation__"          -> "ablang_post_cdr_attn_block" | "esm2"
"<sha1(seq|max_len)>_hidden"  -> (max_len, D) float16   # D=768(vh/vl) or 640(ag)
"<sha1(seq|max_len)>_mask"    -> (max_len,) bool
"<sha1(seq|max_len)>_seq"     -> 원본 서열 문자열 (역참조용)
```

키는 `sha1(seq|max_len)`이라서 서열이 같으면(train/val/test가 흔히 서열을 공유) 한 번만
인코딩된다. `max_len`이 config와 다르면 hidden state가 padding 길이에 의존하므로 조용히
섞어 쓰지 않고 즉시 에러를 낸다.

---

## 3. 학습 시 캐시 읽기 — `dataprocessing.py` (`AbAgCachedDataset`)

이 단계에는 **토크나이저가 없다** — 토크나이징은 2단계에서 이미 끝났고, 여기서는
`(서열, max_len)` 키로 미리 메모리에 올라온 딕셔너리(`EmbeddingCache.state_dict()`)를
조회할 뿐이다. GPU/모델 참조가 없는 순수 CPU dict 조회라 `num_workers > 0`인
`DataLoader`에서도 안전하다. `train.py`는 학습을 시작하기 전에
`EmbeddingCache.require(...)`로 이 split에 필요한 모든 서열이 npz에 이미 있는지
확인하고, 빠진 게 있으면 "build_embeddings.py를 먼저 실행하라"는 에러로 즉시 죽는다.

`Dataset.__getitem__(idx)`가 돌려주는 딕셔너리:

```python
{
    "vh_hidden":    Tensor(vh_max_len, 768),
    "vh_attn_mask": Tensor(vh_max_len)  # bool,
    "ag_emb":       Tensor(ag_max_len, 640),
    "ag_attn_mask": Tensor(ag_max_len)  # bool,
    "label":        Tensor()  # scalar, float(binary) or long,
    # use_vl=True 인 경우만:
    "vl_hidden":    Tensor(vl_max_len, 768),
    "vl_attn_mask": Tensor(vl_max_len)  # bool,
}
```

`DataLoader`가 이를 배치로 묶어 `(B, T, D)` / `(B, T)` 텐서로 만들고, `train.py`의 학습
루프가 이 배치를 그대로 `AbLangAbAgModel.forward()`에 넘긴다.

---

## 4. 모델 입력 — `AbLangAbAgModel.forward(...)`

```python
def forward(self, vh_hidden, vh_attn_mask,
                  vl_hidden, vl_attn_mask,
                  ag_emb, ag_attn_mask, ...):
```

| 인자 | shape | 의미 |
|---|---|---|
| `vh_hidden` | (B, T_vh, 768) | post-CDR AbLang heavy residue embedding (frozen, 캐시에서 옴) |
| `vh_attn_mask` | (B, T_vh) | 1=실토큰(시작/끝 토큰 포함), 0=pad |
| `vl_hidden` | (B, T_vl, 768) or None | post-CDR AbLang light residue embedding (`use_vl=False`면 None) |
| `vl_attn_mask` | (B, T_vl) or None | |
| `ag_emb` | (B, T_ag, 640) | ESM2 residue embedding (frozen, 캐시에서 옴) |
| `ag_attn_mask` | (B, T_ag) | |

forward 내부에서 가장 먼저 하는 일은 이 raw hidden state를 다시 인코딩하는 게 **아니라**
(`_encode_antibody_chain`은 캐시된 값을 그대로 통과시키는 항등함수다), `exclude_mask`로
"pad 이거나 시작/끝 특수 토큰"인 위치를 걸러낸 뒤, 공유 `ab_proj`(768→256, VH/VL 공용)와
`ag_proj`(640→256)로 두 표현을 같은 차원의 공간에 투영하는 것이다. 이후 진행되는
BAN residue-pair 상호작용 맵 생성, cross-context 계산, pooling, 최종 binding logit
산출 과정은 `MODEL.md`(V7: BAN 게이트 포함)에
자세히 정리되어 있다.

---

## 요약: 각 단계에서 "얼려진(frozen)" 것과 "학습되는" 것

| 구성 요소 | 상태 | 근거 |
|---|---|---|
| AbLang heavy/light `AbRep` 백본 | frozen, 캐시 생성 시에만 사용 | `build_ablang_encoder`가 `requires_grad=False`로 로드, 학습 그래프에 없음 |
| CDR checkpoint의 `attn_block` | frozen, 캐시 생성 시에만 사용 | `build_cdr_layer`가 eval + `requires_grad=False` |
| ESM2 (항원) | frozen, 캐시 생성 시에만 사용 | `build_esm_encoder`가 `requires_grad=False`로 로드 |
| `ab_proj` / `ag_proj` (Linear→ReLU→MaskedBatchNorm) | **학습됨** | `model.py`의 `nn.Module` 파라미터 |
| `BANLayer` (v_net/q_net/h_mat 등) | **학습됨** | |
| `embedding_classifier` / `map_classifier` | **학습됨** | |

즉 이 모델에서 실제로 gradient가 흐르는 부분은 2단계(임베딩 캐싱)가 끝난 **뒤**,
projection부터 최종 classifier까지뿐이다. 그래서 `build_embeddings.py`를 한 번만
돌려두면 학습(`train.py`)은 AbLang/ESM2 forward 없이 훨씬 가볍게 반복할 수 있다.
