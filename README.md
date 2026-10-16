# CFR-AAI

CDR 예측 모델(ABlang + 추가 어텐션 레이어)을 먼저 학습하고, 그 모델이 만든 항체 임베딩과 ESM2 항원 임베딩으로
항원-항체 상호작용(BAN)을 예측한다.

```
OAS paired ─▶ ① CDR 예측 학습 ─▶ CDR 체크포인트 ─▶ ② 임베딩 추출 (항체: AbLang+CDR 레이어 / 항원: ESM2)
                                                          │  emb/<dataset>/{vh,vl,ag}_emb.npz
                                                          ▼
                                                   ③ 항원-항체 상호작용 학습 (BAN)
```

```
cdr_prediction/      # ① OAS paired 로 CDR(7-class region) 예측 모델 학습
ab_ag_interaction/   # ②③ 임베딩 추출(build_embeddings.py) + BAN 상호작용 학습(train.py)  — configs/, docs/
data/OAS_paired/     # OAS 다운로드·CDR 추출·클러스터 split 스크립트 (데이터 본체는 제외)
data/ab_ag/          # SAbDab / HIV / CoVAbDab / AVIDa 등 상호작용 데이터 (50MB 초과 파일 제외)
scripts/             # setup_ablang_weights.py
run_pipeline.sh      # 전체 파이프라인 실행
environment.yml
```

## 0. 환경
```bash
conda env create -p ./.env -f environment.yml   # 환경을 폴더 안(./.env)에 설치 — run_pipeline.sh 가 자동으로 이 python 을 쓴다
cp -r ~/.cache/huggingface/hub/models--facebook--esm2_t30_150M_UR50D .hf_cache/hub/   # (선택) 이미 받은 ESM2 재사용. 없으면 HF_HOME=./.hf_cache 로 자동 다운로드
```
`run_pipeline.sh` 는 `HF_HOME=./.hf_cache` 로 고정하고, 시작할 때 `scripts/setup_ablang_weights.py`(AbLang 가중치를 `checkpoints/ablang` 에서 환경으로 복사)를 실행한다.
수동 실행 시: `conda activate ./.env && python scripts/setup_ablang_weights.py`.
torch 는 cu128 빌드(RTX 50xx 대응)다. 다른 GPU 환경이면 `environment.yml` 의 index-url 을 맞게 바꾼다.

## 전체 실행
```bash
./run_pipeline.sh                  # 아래 ①~③ 전부 (data → cdr → embed → train)
./run_pipeline.sh embed train      # 단계 선택 (data | cdr | embed | train)
DATASETS="AVIDa-SARS-CoV-2" ./run_pipeline.sh embed train   # 데이터셋 선택 (기본: AVIDa-SARS-CoV-2 HIV)
```
환경변수: `PY`(python 경로), `CDR_CONFIG`(기본 `cdr_prediction/config.yaml`), `AG_CONFIG_DIR`(기본 `ab_ag_interaction/configs`), `DATASETS`.

## ① CDR 예측 학습
```bash
data/OAS_paired/scripts/run_all.sh   # 다운로드(3.3GB) → CDR 추출 → cd-hit 90% 클러스터링 + 5-fold → dataset/{heavy,light}/fold_k/*.csv.gz
cd cdr_prediction && ./run.sh both 0 # 결과: cdr_prediction/runs/ablang_cdr_7class_both/fold_0/best_model.pt
cp cdr_prediction/runs/ablang_cdr_7class_both/fold_0/best_model.pt checkpoints/cdr_ablang_best_model.pt
```
`run_pipeline.sh cdr` 가 마지막 복사까지 해준다. 학습 없이 쓰려면 사전학습 체크포인트를 `checkpoints/cdr_ablang_best_model.pt` 에 두면 된다
(<여기에 링크 추가>). 각 `ab_ag_interaction/configs/*.yaml` 의 `pretrained.cdr_ckpt` 가 이 파일을 가리킨다.

## ② 임베딩 추출 (학습 전 1회)
CDR 체크포인트의 AbLang 백본 + CDR 어텐션 레이어로 항체(VH/VL) 잔기 임베딩을, ESM2-150M 으로 항원 임베딩을 만들어 저장한다.
학습 단계는 인코더 없이 이 npz 만 읽는다.
```bash
cd ab_ag_interaction
python build_embeddings.py --config configs/AVIDa_SARS_CoV_2.yaml --dataset AVIDa-SARS-CoV-2
python build_embeddings.py --config configs/HIV.yaml               --dataset HIV
```
결과는 `emb/<dataset>/` (git 제외, 데이터셋당 수 GB).

## ③ 항원-항체 상호작용 학습
```bash
cd ab_ag_interaction
python train.py --config configs/AVIDa_SARS_CoV_2.yaml --dataset AVIDa-SARS-CoV-2
python train.py --config configs/HIV.yaml --dataset HIV/unseen      # ag_unseen, ab_unseen 도 동일
```
결과는 `ab_ag_interaction/runs/<dataset>/...`.

## 데이터 / 체크포인트 (저장소에 없음)
- OAS paired: https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/ — `run_all.sh` 로 재생성 (원본 3.3GB, dataset 2.5GB)
- 큰 상호작용 데이터(il6 `il6_aai_dataset.csv` 등 50MB 초과)와 `AVIDa_hIL6` 원본은 제외됨 → `data/ab_ag/il6/README.md`, `build_il6_dataset.py` 참고
- `checkpoints/` (CDR 모델, AbLang 가중치, 학습된 상호작용 모델)는 git 제외 — 별도 호스팅
