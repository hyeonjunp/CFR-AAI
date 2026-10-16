# CDR_Ablang_V7 모델 구조

V7은 V6의 구조를 유지하면서, 기존 BAN prediction인 `ban_logits`를 attention map에 곱해 cross-context를 만드는 변형이다. IIP는 사용하지 않는다.

차원은 `CDR-SCA_BAN_V2/config.yaml`에 맞춰 기본적으로 projection/BAN hidden 256, classifier hidden 256, classifier bottleneck 128을 사용한다. 단, AbLang 입력 차원은 768이고 항원 cached embedding 기본 차원은 640이므로 입력 차원 자체는 V2의 ESM2 설정과 다르다.

## 핵심 흐름

```text
P_ag, P_ab
   │
   └─ BAN low-rank bilinear map
          │
          ├─ masked global softmax → A_ban
          │                         │
          │                         ├─ BAN bilinear pooling → ban_logits
          │                         │
          │                         └─ sigmoid(ban_logits) → gate
          │
          └─ A_gate = A_ban × gate
                         │
                 ┌───────┴────────┐
                 │                │
        C_ag = A_gate P_ab  C_ab = A_gateᵀ P_ag
                 │                │
                 └───────┬────────┘
                         │
                 gated residual + masked mean pooling
                         │
                    embedding_logits

final_logits = 0.5 × (embedding_logits + ban_logits)
```

## V6와 V7의 차이

| 항목 | V6 | V7 |
|---|---|---|
| IIP | 사용하지 않음 | 사용하지 않음 |
| BAN map | cross-context에 사용 | cross-context에 사용 |
| BAN prediction | `ban_logits` 계산 | `ban_logits` 계산 |
| map gate | 사용하지 않음 | `sigmoid(ban_logits)` |
| gate gradient | 해당 없음 | BAN gate로 역전파함 |
| projection dimension | 256 | 256 |
| BAN hidden dimension | 256 | 256 |
| classifier hidden/bottleneck | 256 / 128 | 256 / 128 |

## 수식

BAN이 생성한 normalized residue-pair map을 다음처럼 둔다.

```text
A_ban ∈ R^(B × H × T_ag × T_ab)
```

BAN bilinear pooling으로 binding logit을 계산한다.

```text
f_ban = BAN_bilinear_pool(A_ban, V, Q)
ban_logits = map_classifier(f_ban)
```

V7에서는 각 sample의 BAN logit을 scalar gate로 변환한다.

```text
g = sigmoid(ban_logits)
A_gate[h,i,j] = A_ban[h,i,j] × g
```

이후 head 평균으로 최종 map을 만들고 양방향 context를 계산한다.

```text
interaction_map = mean_h(A_gate)
C_ag = interaction_map × P_ab
C_ab = interaction_mapᵀ × P_ag
```

따라서 BAN이 binding 가능성이 높다고 예측하는 sample은 residue-pair map의 영향이 커지고, 낮게 예측하는 sample은 cross-context 영향이 작아진다.

## 중요한 구현 사항

- `ban_logits`는 raw logit이므로 map에 직접 곱하지 않고 `sigmoid(ban_logits)`를 사용한다.
- V7에서는 `ban_logits.detach()`를 사용하지 않는다.
- 따라서 embedding branch의 loss도 BAN classifier와 BAN map에 gradient를 전달한다.
- map branch loss도 기존처럼 `ban_logits`에 직접 적용된다.
- IIP, CNN, top-k pooling은 없다.

## 반환값

```python
logits, info = model(..., return_attn=True)
```

| key | 의미 |
|---|---|
| `base_head_maps` | BAN normalized map, gate 적용 전 |
| `ban_map_feature` | BAN bilinear pooling feature |
| `ban_logits` | 기존 BAN binding prediction |
| `interaction_gate` | `sigmoid(ban_logits)` |
| `head_maps` | `base_head_maps × interaction_gate` |
| `interaction_map` | head 평균 후 cross-context에 사용되는 map |
| `c_ag`, `c_ab` | V7 gated attention map으로 만든 context |

## 사용

V7의 `model.py`는 V6 model을 기반으로 하며, `detach_ban_gate=False`만 추가한 최소 변형이다. V6의 데이터/Trainer 코드와 함께 사용할 수 있다.

```python
from model import AbLangAbAgModel

model = AbLangAbAgModel(
    ag_embed_dim=640,
    proj_dim=256,
    co_attn_heads=1,
    use_raw_ablang=True,
    use_cdr_chain_features=False,
)
```
