# model_ablang — ABlang + AbLangStyleLayer CDR 예측 (OAS_paired, 7-class + VH/VL)

`../model_OAS`(ESM2 + sigmoid attention, 7-class region + VH/VL 보조 헤드)와 같은
라벨 체계/손실 구성을 쓴다. 인코더만 ESM2 대신 ABlang(heavy/light 전용 체크포인트 2개)이다.

```
frozen ABlang(heavy/light) → (B,T,768) → 새 레이어 1개 (ABlang 자신의 EncoderBlock과
같은 구조, post-LN softmax attn, 새로 초기화)
    ├─ CDRHead     → region_logits (B,T,7)  [FR1/CDR1/FR2/CDR2/FR3/CDR3/FR4, CrossEntropy]
    └─ ChainHead   → chain_logits  (B,)     [heavy=1/light=0, mean-pool, BCE]

loss = region_CE(class_weight: CDR1/2/3=pos_weight) + chain_loss_weight * chain_BCE
label: 0..6 = region, -100 = '<'·'>'·pad
```

## `model_OAS` 대비 바뀐 것

| | model_OAS (ESM2) | model_ablang |
|---|---|---|
| 인코더 | ESM2-650M (범용, chain 무관 단일 모델) | ABlang (heavy 전용 / light 전용 체크포인트 2개) |
| `chain="both"` 처리 | frozen ESM2 하나를 그냥 공유 | frozen 백본을 heavy/light 각각 로드, 배치 내 `chain_type`으로 샘플별 라우팅. 새로 쌓는 레이어 1개 + 두 head는 공유 |
| 위치 정보 | ESM2 자신의 RoPE (레이어가 직접 봄) | ABlang의 학습된 절대 위치 임베딩 (입력 임베딩 단계에서만 더해짐 — frozen hidden state에 이미 녹아있음). 새로 쌓는 레이어 자체는 위치 인코딩 없음 |
| 토크나이저 | `EsmTokenizer` 하나 | `ABtokenizer` 두 개 (heavy/light). `ABlangCDRModel`이 로드해 둔 것을 `model.tokenizers`로 Dataset/collate에 넘김 |
| hidden_size | 1280 | 768 (ABlang `hparams.hidden_size`, 동적으로 읽음) |
| special token | `<cls>` / `<eos>` | `<` (시작) / `>` (끝) |
| max_len 한도 | 256 (여유 있음, ESM2 자체 한도가 훨씬 큼) | **160** (ABlang `max_position_embeddings` 실측값, 하드 리밋) |
| ChainHead 의의 | 서열만 보고 VH/VL을 맞혀야 해서 의미 있는 보조 태스크 | heavy/light가 이미 백본 라우팅의 정답이라 다소 잉여스럽지만, `model_OAS`와 산출물 포맷을 맞추기 위해 유지 |

region label 유도(`mask_to_regions`), 7-class CrossEntropy + class weight, VH/VL BCE,
동적 패딩(`collate_dynamic`), metric(7x7 혼동행렬 + CDR 묶음 binary F1 + chain acc) 로직은
그대로 이식했다.

## 실행

```bash
./run.sh                    # 기본값: chain=both(heavy+light), fold 0
./run.sh both  0 --resume   # 마지막 checkpoint 에서 재개
./run.sh both  all          # fold 0~4 (5-fold CV) → runs/.../kfold_summary.csv
./run.sh heavy 0            # heavy 만 (ABlang heavy 백본 하나만 로드)
./run.sh light 0            # light 만

# 직접 호출
python train.py --config config.yaml --fold 0 --chain both [--resume]
```

산출물: `runs/ablang_cdr_7class_<chain>/fold_<k>/`
`best_model.pt`, `checkpoint_epoch*.pt`, `training_log.csv`
(`loss/cdr_loss/chain_loss/chain_acc/acc/prec/rec/f1` 컬럼), `test_predictions.csv.gz`
(`true_region/pred_region`, `true_mask/pred_mask`, `true_chain/pred_chain` 등).

## ABlang 가중치

`ablang.pretrained(chain=...)`가 첫 실행 시 자동으로 다운로드한다
(`<ablang 패키지 경로>/model-weights-{heavy,light}/`, 체인당 한 번만, ~300MB).
opig.stats.ox.ac.uk 서버 다운로드 속도가 느릴 수 있으니(환경에 따라 <100KB/s로 느려지는
경우 관측됨) 최초 실행 전에 미리 받아두는 것을 권장한다.

heavy 체크포인트 실측 hparams: `hidden_size=768, num_hidden_layers=12,
num_attention_heads=12, intermediate_size=3072, layer_norm_eps=1e-12,
max_position_embeddings=160, pad_token_id=21, vocab_size=24`.

## 데이터

`config.dataset_dir/<chain>/fold_<k>/{train,valid,test}.csv.gz` (CD-HIT 90% 클러스터 기반 fold),
로딩 로직은 `model_OAS`와 동일. `sequence`와 같은 길이의 0/1 문자열 `mask`에서
`mask_to_regions()`로 7-class 라벨을 유도한다(마스크의 '1' 구간이 정확히 3개여야 함 —
`train.py`의 `check_dataframe`이 fold 진입 시 확인). `chain_type` 컬럼이 heavy/light
라우팅·토크나이저 선택·VH/VL 라벨에 모두 쓰이므로 필수다 (OAS_paired 데이터셋에는 이미 있음).

## max_len 주의

ABlang의 위치 임베딩 테이블 크기(`hparams.max_position_embeddings`, 실측 160)를 넘는
`max_len`을 config에 주면 `train.py`가 fold 진입 시 바로 에러를 낸다. OAS_paired 최대
길이는 heavy 156 / light 123이라 `'<'+AA+'>'` 기준 heavy에 158 토큰이 필요해 160 한도
안에 여유 2토큰으로 들어온다.

## 학습된 임베딩을 DeepInterAware에 넘기기

`extract_features.py` 가 학습된 모델의 토큰별 임베딩을 DeepInterAware 포맷으로 저장한다.
DeepInterAware 는 항체 feature 를 `{cfg.set.ab_model}_token_ft.pt` 이름으로 읽으므로
(`DeepInterAware/dataloader.py`의 PairDataset), 파일명을 맞춰 두고 yml 의 `ab_model` 만
바꾸면 DeepInterAware 코드를 고치지 않고 임베딩을 교체할 수 있다.

```bash
# 학습된 레이어를 통과한 임베딩 (CDR 태스크로 학습된 표현)
python extract_features.py \
    --checkpoint runs/ablang_cdr_7class_both/fold_0/best_model.pt \
    --antibody-csv ../DeepInterAware/data/SAbDab/antibody.csv \
    --heavy-col H_cdr --light-col L_cdr --max-ab-len 110 \
    --out-dir ../DeepInterAware/data/SAbDab --name cdrablang

# 대조군: 같은 파이프라인에서 frozen ABlang 출력만 (원본 rescoding 과 동등)
python extract_features.py ... --name ablang --raw
```

그 다음 `DeepInterAware/configs/SAbDab.yml` 에서 `set.ab_model: cdrablang`.

저장 포맷은 원본 `utools/feature_encoder.py`의 `process_ablang_token_ft` 규약을 그대로 맞췄다
(`[XUZOBJ?*]` 제거, 체인당 157 잔기 절단, heavy→light concat, `manual_seed(42)` 패딩 벡터).

**주의**: 이 모델은 OAS 의 **전체 V 도메인**(FR1~FR4, 평균 118잔기)으로 학습했다. SAbDab 처럼
**CDR 만 이어붙인 문자열**(`H_cdr`+`L_cdr`, 평균 80잔기)을 넣으면 학습 분포와 다르다 —
동작은 하지만 "CDR 태스크로 학습된 표현"의 이점이 그대로 옮겨간다는 보장은 없다.
`--raw` 대조군과 비교해서 실제로 이득이 있는지 확인하는 편이 좋다.

## 빠른 확인 (스모크)

```bash
# config 를 복사해 max_train_samples: 512 / max_valid_samples: 256 / num_epochs: 1 로
python train.py --config smoke.yaml --fold 0
```
