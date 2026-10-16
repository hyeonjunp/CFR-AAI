#!/bin/bash
# 전체 파이프라인: OAS 데이터 → CDR 예측 학습 → CDR 모델 기반 임베딩 추출 → 항원-항체 상호작용 학습
#
#   ./run_pipeline.sh                 # 4단계 전부
#   ./run_pipeline.sh cdr embed       # 지정한 단계만 (data | cdr | embed | train)
#
# 환경변수
#   PY            python 실행 파일            (기본: ./.env/bin/python, 없으면 python)
#   CDR_CONFIG    CDR 예측 config             (기본: cdr_prediction/config.yaml)
#   AG_CONFIG_DIR 상호작용 config 폴더         (기본: ab_ag_interaction/configs)
#   DATASETS      임베딩/학습 대상 데이터셋     (기본: "AVIDa-SARS-CoV-2 HIV")
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
# 폴더 안 자급자족: ./.env (conda -p) 가 있으면 그 python, HF 캐시(ESM2)도 ./.hf_cache
[ -x "$ROOT/.env/bin/python" ] && PY=${PY:-$ROOT/.env/bin/python}
PY=${PY:-python}
export PATH="$(dirname "$PY"):$PATH"   # cd-hit 등 환경 안 실행 파일
export HF_HOME=${HF_HOME:-$ROOT/.hf_cache}
CDR_CONFIG=${CDR_CONFIG:-$ROOT/cdr_prediction/config.yaml}
AG_CONFIG_DIR=${AG_CONFIG_DIR:-$ROOT/ab_ag_interaction/configs}
DATASETS=${DATASETS:-"AVIDa-SARS-CoV-2 HIV"}
STAGES=${*:-data cdr embed train}

# 데이터셋 이름 → config 파일 이름
cfg_of() { case "$1" in HIV*) echo HIV;; AVIDa*) echo AVIDa_SARS_CoV_2;; il6) echo il6;; *) echo "$1";; esac; }
run_stage() { [[ " $STAGES " == *" $1 "* ]]; }
log() { echo; echo "[$(date '+%F %T')] ===== $* ====="; }

$PY "$ROOT/scripts/setup_ablang_weights.py" >/dev/null   # AbLang 가중치를 환경에 설치 (멱등)

if run_stage data; then
  log "1/4 OAS 데이터 (다운로드 → CDR 추출 → 클러스터링/5-fold split)"
  OAS="$ROOT/data/OAS_paired"
  if [ -f "$OAS/dataset/heavy/fold_0/train.csv.gz" ] && [ -f "$OAS/dataset/light/fold_0/train.csv.gz" ]; then
    echo "이미 있음: $OAS/dataset — 건너뜀"
  else
    PY=$PY "$OAS/scripts/run_all.sh"
  fi
fi

if run_stage cdr; then
  log "2/4 CDR 예측 학습 (config: $CDR_CONFIG)"
  ( cd "$ROOT/cdr_prediction" && $PY train.py --config "$CDR_CONFIG" --chain both --fold 0 )
  RUN_DIR=$($PY - "$CDR_CONFIG" "$ROOT/cdr_prediction" <<'PYEOF'
import os, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
save = cfg["save_dir"] if os.path.isabs(cfg["save_dir"]) else os.path.join(sys.argv[2], cfg["save_dir"])
print(os.path.join(save, f"{cfg['run_name']}_both", "fold_0"))
PYEOF
)
  mkdir -p "$ROOT/checkpoints"
  cp "$RUN_DIR/best_model.pt" "$ROOT/checkpoints/cdr_ablang_best_model.pt"
  echo "CDR 체크포인트 → checkpoints/cdr_ablang_best_model.pt  (from $RUN_DIR)"
fi

if run_stage embed; then
  log "3/4 임베딩 추출 (AbLang+CDR 레이어, ESM2) → emb/<dataset>/*.npz"
  for ds in $DATASETS; do
    ( cd "$ROOT/ab_ag_interaction" && $PY build_embeddings.py --config "$AG_CONFIG_DIR/$(cfg_of "$ds").yaml" --dataset "$ds" )
  done
fi

if run_stage train; then
  log "4/4 항원-항체 상호작용 학습"
  for ds in $DATASETS; do
    cfg="$AG_CONFIG_DIR/$(cfg_of "$ds").yaml"
    if [[ "$ds" == HIV* && "$ds" != */* ]]; then
      for sp in unseen ag_unseen ab_unseen; do
        ( cd "$ROOT/ab_ag_interaction" && $PY train.py --config "$cfg" --dataset "HIV/$sp" )
      done
    else
      ( cd "$ROOT/ab_ag_interaction" && $PY train.py --config "$cfg" --dataset "$ds" )
    fi
  done
fi
log "완료: $STAGES"
