# CFR-AAI

**CDR-aware 항체–항원 상호작용(Antibody–Antigen Interaction) 예측**

항체 언어모델(AbLang)에 CDR 예측 과제를 먼저 학습시켜 CDR 정보를 담은 항체 표현을 만들고,
이 표현과 항원 단백질 언어모델(ESM2) 임베딩을 Bilinear Attention Network(BAN)으로 결합해
항체–항원 결합 여부를 예측합니다.

```
 OAS paired ──▶ ① CDR 예측 학습 ──▶ CDR 체크포인트
                (AbLang + attention 레이어,        │
                 7-class region)                   ▼
                                    ② 임베딩 추출 (1회)
                                       항체 VH/VL : AbLang + CDR 레이어 (768-d)
                                       항원       : ESM2-150M            (640-d)
                                                   │  emb/<dataset>/{vh,vl,ag}_emb.npz
                                                   ▼
                                    ③ 항원–항체 상호작용 학습 (BAN)  ──▶ 결합 확률
```

| 단계 | 하는 일 | 위치 |
|---|---|---|
| ① CDR 예측 | 서열의 각 잔기를 FR1/CDR1/FR2/CDR2/FR3/CDR3/FR4 로 분류 (+ VH/VL 보조 과제) | `cdr_prediction/` |
| ② 임베딩 추출 | 학습된 CDR 모델과 ESM2 로 항체·항원 잔기 임베딩을 만들어 캐시 | `ab_ag_interaction/build_embeddings.py` |
| ③ 상호작용 학습 | 캐시된 임베딩만 읽어 BAN 기반 분류기 학습 (인코더 불필요) | `ab_ag_interaction/train.py` |

## 지원 데이터셋

| 데이터셋 | 항체 종류 | 설정 파일 | 비고 |
|---|---|---|---|
| AVIDa-SARS-CoV-2 | 나노바디 (VHH) | `configs/AVIDa_SARS_CoV_2.yaml` | 5-fold |
| HIV | VH + VL | `configs/HIV.yaml` | `unseen` / `ag_unseen` / `ab_unseen` split |
| hIL6 | 나노바디 (VHH) | `configs/il6.yaml` | 원본 데이터는 별도 생성 필요 ([안내](data/ab_ag/il6/README.md)) |

나노바디 데이터셋은 light chain 이 없어 VL stream 을 자동으로 제외합니다.

## 저장소 구조

```
cdr_prediction/       ① OAS paired 로 CDR 예측 모델 학습         (README 참고)
ab_ag_interaction/    ②③ 임베딩 추출 + BAN 상호작용 학습
  ├─ configs/           데이터셋별 설정 (yaml)
  └─ docs/              MODEL.md (모델 구조), INPUT_PIPELINE.md (입력 파이프라인)
data/OAS_paired/      OAS 다운로드 · CDR 추출 · 클러스터링 split 스크립트와 통계 리포트
data/ab_ag/           항원–항체 상호작용 데이터 (CSV, split 포함)
scripts/              setup_ablang_weights.py, ckpt_tool.py (체크포인트 슬림화/HF 업로드)
run_pipeline.sh       전체 파이프라인 실행
environment.yml       conda 환경
```

## 빠른 시작

### 1. 환경 설치
```bash
conda env create -p ./.env -f environment.yml
conda activate ./.env
python scripts/setup_ablang_weights.py   # AbLang 사전학습 가중치 설치
```
- PyTorch 는 CUDA 12.8 빌드(RTX 50xx 대응)입니다. 다른 GPU 라면 `environment.yml` 의 `--extra-index-url` 을 환경에 맞게 바꾸세요.
- ESM2 가중치는 첫 실행 때 Hugging Face 에서 자동으로 받습니다 (`HF_HOME=./.hf_cache`).

### 2. 전체 파이프라인
```bash
./run_pipeline.sh                  # data → cdr → embed → train 전부
./run_pipeline.sh embed train      # 일부 단계만 (data | cdr | embed | train)
DATASETS="AVIDa-SARS-CoV-2" ./run_pipeline.sh embed train   # 데이터셋 지정 (기본: AVIDa-SARS-CoV-2 HIV)
```
환경변수: `PY`(python 경로), `CDR_CONFIG`, `AG_CONFIG_DIR`, `DATASETS`

### 3. 단계별 실행

**① CDR 예측 학습**
```bash
data/OAS_paired/scripts/run_all.sh        # 다운로드(3.3GB) → CDR 추출 → cd-hit 90% 클러스터링 + 5-fold split
cd cdr_prediction && ./run.sh both 0      # chain=both, fold 0
cp cdr_prediction/runs/ablang_cdr_7class_both/fold_0/best_model.pt checkpoints/cdr_ablang_best_model.pt
```
각 `configs/*.yaml` 의 `pretrained.cdr_ckpt` 가 `checkpoints/cdr_ablang_best_model.pt` 를 가리킵니다.

**② 임베딩 추출** (학습 전 1회, 결과는 `emb/<dataset>/`)
```bash
cd ab_ag_interaction
python build_embeddings.py --config configs/AVIDa_SARS_CoV_2.yaml --dataset AVIDa-SARS-CoV-2
python build_embeddings.py --config configs/HIV.yaml               --dataset HIV
```

**③ 상호작용 학습** (결과는 `ab_ag_interaction/runs/<dataset>/`)
```bash
cd ab_ag_interaction
python train.py --config configs/AVIDa_SARS_CoV_2.yaml --dataset AVIDa-SARS-CoV-2
python train.py --config configs/HIV.yaml --dataset HIV/unseen      # HIV/ag_unseen, HIV/ab_unseen 도 동일
```
`--seed`, `--fold`, `--all` 로 split 을 선택하며, 생략하면 전체 seed/fold 를 순서대로 실행합니다.

## 모델 개요

- **항체 인코더**: AbLang(heavy/light 각각, frozen) → CDR 예측으로 학습한 attention 레이어 → 768-d 잔기 임베딩
- **항원 인코더**: ESM2-150M (frozen) → 640-d 잔기 임베딩
- **상호작용**: 공통 차원(256)으로 투영 후 BAN 으로 잔기쌍 attention map 과 결합 logit 계산
- **학습**: 인코더가 모두 frozen 이므로 캐시된 임베딩만으로 빠르게 학습 (early stopping, AUROC 기준 best 저장)

자세한 구조는 [`ab_ag_interaction/docs/MODEL.md`](ab_ag_interaction/docs/MODEL.md),
CDR 예측 모델은 [`cdr_prediction/README.md`](cdr_prediction/README.md) 를 참고하세요.

## 저장소에 포함되지 않는 것

용량 문제로 아래 항목은 git 에서 제외되어 있습니다 (`.gitignore`).

| 항목 | 얻는 방법 |
|---|---|
| `*.pt` 체크포인트 (`checkpoints/`, `runs/`) | ①을 직접 학습하거나 별도 호스팅된 체크포인트를 `checkpoints/` 에 배치. `scripts/ckpt_tool.py` 로 슬림화/Hugging Face 업로드·다운로드 가능 |
| `emb/` 임베딩 캐시 (데이터셋당 수 GB) | ② 단계로 생성 |
| OAS 원본/가공 데이터 (`data/OAS_paired/{raw,processed,dataset}`) | `data/OAS_paired/scripts/run_all.sh` 로 재생성 ([OAS paired](https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/)) |
| hIL6 원본 등 50MB 초과 CSV | [`data/ab_ag/il6/README.md`](data/ab_ag/il6/README.md) 참고 |
