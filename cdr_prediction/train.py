"""
ABlang 기반 CDR 예측 — OAS_paired 학습 스크립트, 7-class region + VH/VL.

`model_OAS/train.py`와 같은 흐름이고, 다른 점은
  * frozen 백본이 ESM2 하나가 아니라 ABlang heavy/light 두 개일 수 있다 (chain="both").
    그래서 loader를 만들기 전에 모델을 먼저 만들어 `model.tokenizers`(heavy/light
    ABtokenizer)를 얻고, 그걸 Dataset/collate_fn에 넘긴다 — ESM 버전은 토크나이저가
    전역 싱글턴이라 모델보다 먼저 loader를 만들 수 있었지만 여기서는 순서가 뒤집힌다.
  * `max_len`이 ABlang 체크포인트의 위치 임베딩 테이블 크기(`hparams.max_position_embeddings`)
    를 넘지 않는지 fold 진입 시 확인한다 (넘으면 임베딩 조회에서 바로 에러가 나므로 미리 확인).
  * 테스트 예측 CSV의 `chain_type` 컬럼은 배치에 이미 있는 `chain_type`을 그대로 쓴다
    (모델 라우팅에 쓰는 값과 같은 값이라 따로 test_df에서 다시 뽑지 않는다).

Usage
-----
    python train.py --config config.yaml
    python train.py --config config.yaml --fold 0            # 단일 fold
    python train.py --config config.yaml --fold 0 --resume   # 이어서 학습
    python train.py --config config.yaml --fold 0 --chain light
"""

import argparse
import csv
import functools
import gzip
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# 이 스크립트가 있는 디렉터리에서 import한다 — 복사본이 다른 폴더의 model.py를
# 끌어다 쓰지 않도록 절대 패키지 경로 대신 로컬 경로를 쓴다.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataprocessing import (ABlangCDRDataset, load_oas_fold, collate_dynamic,
                            REGION_NAMES, NUM_REGIONS, CDR_REGION_IDS,
                            mask_to_regions)
from model import ABlangCDRModel
from trainer import ABlangCDRTrainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def check_dataframe(df: pd.DataFrame, name: str, cfg: dict):
    seq_col, mask_col = cfg.get("seq_col", "sequence"), cfg.get("mask_col", "mask")
    for col in [seq_col, mask_col]:
        if col not in df.columns:
            raise ValueError(f"[{name}] missing column: {col}")

    seq_len  = df[seq_col].astype(str).str.len()
    mask_len = df[mask_col].astype(str).str.len()

    bad = mask_len < seq_len
    if bad.any():
        raise ValueError(f"[{name}] mask shorter than sequence: {bad.sum()} rows")

    inv = ~df[mask_col].astype(str).str.fullmatch(r"[01]+")
    if inv.any():
        raise ValueError(f"[{name}] mask contains non-0/1 values: {inv.sum()} rows")

    max_len = cfg.get("max_len", 160)
    trunc   = (seq_len > max_len - 2).sum()
    if trunc:
        print(f"[{name}] WARNING: {trunc:,} rows longer than max_len-2 ({max_len - 2}) "
              f"→ 잘립니다")

    # 7-class 라벨이 성립하려면 마스크의 '1' 구간이 정확히 3개여야 한다
    n_runs = df[mask_col].astype(str).str.count(r"01") + \
             df[mask_col].astype(str).str.startswith("1").astype(int)
    bad_runs = (n_runs != 3).sum()
    if bad_runs:
        raise ValueError(
            f"[{name}] CDR 구간이 3개가 아닌 행 {bad_runs:,}개 — 7-class 라벨을 만들 수 없습니다")

    chain_col = cfg.get("chain_col", "chain_type")
    if chain_col not in df.columns:
        raise ValueError(f"[{name}] missing column: {chain_col} (VH/VL head 및 백본 라우팅에 필요)")
    vc = df[chain_col].value_counts().to_dict()

    cdr_frac = df[mask_col].astype(str).str.count("1").sum() / max(seq_len.sum(), 1)
    print(f"[{name}] OK | n={len(df):,} | "
          f"len range [{seq_len.min()}, {seq_len.max()}] | mean={seq_len.mean():.1f} | "
          f"CDR 비율={cdr_frac:.3f} | chain={vc}")


def make_loader(df, cfg, tokenizers, shuffle: bool) -> DataLoader:
    bs = cfg["batch_size"] if shuffle else cfg["valid_batch_size"]
    dynamic = cfg.get("dynamic_padding", True)
    ds = ABlangCDRDataset(
        df, tokenizers,
        max_len    = cfg.get("max_len", 160),
        pad_to_max = not dynamic,
        seq_col    = cfg.get("seq_col", "sequence"),
        mask_col   = cfg.get("mask_col", "mask"),
        chain_col  = cfg.get("chain_col", "chain_type"),
        id_col     = cfg.get("id_col", "sample_id"),
    )
    nw = cfg.get("num_workers", 8)
    collate_fn = functools.partial(collate_dynamic, tokenizers=tokenizers) if dynamic else None
    return DataLoader(
        ds, batch_size=bs, shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=nw > 0,
        prefetch_factor=4 if nw > 0 else None,
    )


def build_model(cfg: dict) -> ABlangCDRModel:
    mcfg = cfg.get("model", {})
    model = ABlangCDRModel(
        chain            = cfg.get("chain", "heavy"),
        ablang_model_dir = mcfg.get("ablang_model_dir", "download"),
        num_labels       = NUM_REGIONS,
        dropout          = mcfg.get("dropout", 0.1),
        freeze_ablang    = mcfg.get("freeze_ablang", True),
        unfreeze_last_n  = mcfg.get("unfreeze_last_n", 0),
        num_attn_heads   = mcfg.get("num_attn_heads"),
        use_chain_head   = mcfg.get("use_chain_head", True),
        device           = "cpu",
    )
    max_len = cfg.get("max_len", 160)
    if max_len > model.max_position_embeddings:
        raise ValueError(
            f"config.max_len={max_len} > ABlang max_position_embeddings="
            f"{model.max_position_embeddings}. max_len을 줄이세요."
        )
    return model


@torch.no_grad()
def predict_and_save(trainer, loader, save_path):
    """테스트 예측을 gz CSV 로 스트리밍 저장한다 (fold 당 수십만 행이라 메모리에 안 쌓는다).

    저장 컬럼
      true_region / pred_region : 7-class 라벨 문자열 (사용자 표기 1~7)
      true_mask  / pred_mask    : CDR=1 로 접은 binary 마스크 (기존 포맷과 호환)
      true_chain / pred_chain   : VH/VL (heavy/light)
    """
    model, device = trainer.model, trainer.device
    model.eval()
    ds = loader.dataset

    g_tp = g_fp = g_fn = g_tn = 0
    n_rows = n_exact = n_region_exact = 0
    chain_ok = chain_n = 0
    acc_sum = 0.0
    offset  = 0
    cdr_set = set(CDR_REGION_IDS)

    fh = gzip.open(save_path, "wt", newline="")
    writer = csv.writer(fh)
    writer.writerow(["seq_id", "chain_type", "sequence",
                     "true_region", "pred_region", "region_exact_match",
                     "true_mask", "pred_mask", "mask_exact_match",
                     "true_cdr_seq", "pred_cdr_seq",
                     "true_chain", "pred_chain", "chain_correct",
                     "token_accuracy", "cdr_precision", "cdr_recall", "cdr_f1"])

    for batch in tqdm(loader, desc="Test predict"):
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels         = batch["labels"].to(device, non_blocking=True)
        aa_mask        = batch["aa_mask"].to(device, non_blocking=True)
        chain_label    = batch["chain_label"]
        chain_types    = batch["chain_type"]

        with trainer._autocast():
            region_logits, chain_logits = model(input_ids=input_ids,
                                                attention_mask=attention_mask,
                                                chain_types=chain_types,
                                                aa_mask=aa_mask)
        preds = region_logits.float().argmax(dim=-1)              # (B, T) 0..6
        valid = labels != -100

        # CDR(1,3,5) 로 접어 binary 지표 계산 — binary 학습 결과와 비교 가능
        p_bin = torch.isin(preds, torch.tensor(CDR_REGION_IDS, device=preds.device))
        g_bin = torch.isin(labels, torch.tensor(CDR_REGION_IDS, device=labels.device))
        tp = (p_bin & g_bin & valid).sum(1)
        fp = (p_bin & ~g_bin & valid).sum(1)
        fn = (~p_bin & g_bin & valid).sum(1)
        tn = (~p_bin & ~g_bin & valid).sum(1)
        g_tp += int(tp.sum()); g_fp += int(fp.sum())
        g_fn += int(fn.sum()); g_tn += int(tn.sum())

        lens = valid.sum(1).cpu().numpy()
        tp_n, fp_n, fn_n, tn_n = (x.cpu().numpy() for x in (tp, fp, fn, tn))
        preds_n = preds.cpu().numpy()

        if chain_logits is not None:
            chain_pred = (torch.sigmoid(chain_logits.float()) > 0.5).long().cpu().numpy()
        else:
            chain_pred = None
        chain_gold = chain_label.numpy()

        for b in range(input_ids.size(0)):
            L    = int(lens[b])
            seq  = ds.seqs[offset + b].decode()[:L]
            true_mask = ds.masks[offset + b].decode()[:L]
            true_region_ids = mask_to_regions(true_mask)
            # 라벨은 pos 1..L 에 정렬되어 있다 (pos 0 = '<')
            pred_region_ids = list(preds_n[b, 1:1 + L])

            # 사용자 표기 1~7 로 저장
            true_region = "".join(str(x + 1) for x in true_region_ids)
            pred_region = "".join(str(x + 1) for x in pred_region_ids)
            pred_mask   = "".join("1" if x in cdr_set else "0" for x in pred_region_ids)

            prec = tp_n[b] / (tp_n[b] + fp_n[b]) if (tp_n[b] + fp_n[b]) else 0.0
            rec  = tp_n[b] / (tp_n[b] + fn_n[b]) if (tp_n[b] + fn_n[b]) else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            tok_acc = (tp_n[b] + tn_n[b]) / max(L, 1)

            exact = true_mask == pred_mask
            r_exact = true_region == pred_region
            n_exact += int(exact); n_region_exact += int(r_exact)
            acc_sum += tok_acc; n_rows += 1

            gold_c = int(chain_gold[b])
            if chain_pred is not None and gold_c != -100:
                ok = int(chain_pred[b]) == gold_c
                chain_ok += int(ok); chain_n += 1
                t_c = "heavy" if gold_c == 1 else "light"
                p_c = "heavy" if int(chain_pred[b]) == 1 else "light"
            else:
                ok, t_c, p_c = "", "", ""

            writer.writerow([
                ds.ids[offset + b].decode() if ds.ids is not None else str(offset + b),
                chain_types[b],
                seq, true_region, pred_region, r_exact,
                true_mask, pred_mask, exact,
                "".join(a for a, m in zip(seq, true_mask) if m == "1"),
                "".join(a for a, m in zip(seq, pred_mask) if m == "1"),
                t_c, p_c, ok,
                round(float(tok_acc), 4),
                round(float(prec), 4), round(float(rec), 4), round(float(f1), 4),
            ])
        offset += input_ids.size(0)

    fh.close()

    op = g_tp / (g_tp + g_fp) if (g_tp + g_fp) else 0.0
    orr = g_tp / (g_tp + g_fn) if (g_tp + g_fn) else 0.0
    of1 = 2 * op * orr / (op + orr) if (op + orr) else 0.0
    print(f"\n예측 저장: {save_path}")
    print(f"  Region(7-class) 완전 일치 : {n_region_exact}/{n_rows} "
          f"({n_region_exact / max(n_rows,1) * 100:.2f}%)")
    print(f"  Mask(binary) 완전 일치    : {n_exact}/{n_rows} "
          f"({n_exact / max(n_rows,1) * 100:.2f}%)")
    print(f"  Token Acc                 : {acc_sum / max(n_rows, 1):.4f}")
    print(f"  CDR Prec/Rec/F1           : {op:.4f} / {orr:.4f} / {of1:.4f}")
    if chain_n:
        print(f"  VH/VL 분류 정확도          : {chain_ok}/{chain_n} "
              f"({chain_ok / chain_n * 100:.2f}%)")


def run_fold(cfg, fold_run_dir, train_df, valid_df, test_df, resume: bool):
    os.makedirs(fold_run_dir, exist_ok=True)

    model      = build_model(cfg)
    tokenizers = model.tokenizers

    dl_train = make_loader(train_df, cfg, tokenizers, shuffle=True)
    dl_valid = make_loader(valid_df, cfg, tokenizers, shuffle=False)

    trainer = ABlangCDRTrainer(
        model             = model,
        lr                = cfg.get("lr", 1e-4),
        weight_decay      = cfg.get("weight_decay", 1e-3),
        max_grad_norm     = cfg.get("max_grad_norm", 1.0),
        pos_weight        = cfg.get("pos_weight", 3.0),
        train_loader      = dl_train,
        val_loader        = dl_valid,
        num_epochs        = cfg.get("num_epochs", 5),
        amp               = cfg.get("amp", True),
        amp_dtype         = cfg.get("amp_dtype", "bf16"),
        chain_loss_weight = cfg.get("chain_loss_weight", 1.0),
    )

    if resume and trainer.load_latest_checkpoint(fold_run_dir):
        print("  (resume) 마지막 checkpoint 에서 이어서 학습합니다")

    print(f"  Train={len(train_df):,}  Valid={len(valid_df):,}  "
          f"Test={len(test_df):,}  run_dir={fold_run_dir}")

    trainer.fit(
        run_dir        = fold_run_dir,
        epochs         = cfg.get("num_epochs", 5),
        validate_every = cfg.get("validate_every", 1),
        save_every     = cfg.get("save_every", 1),
        patience       = cfg.get("patience", 0),
    )

    trainer.load_best_checkpoint(fold_run_dir)

    # ── test ─────────────────────────────────────────────────────────────────
    dl_test = make_loader(test_df, cfg, tokenizers, shuffle=False)
    test_loss, test_acc, test_prec, test_rec, test_f1 = trainer.validate(
        dl_test, name="test", verbose=True
    )
    L = getattr(trainer, "last_losses", {})
    trainer.log_to_csv(fold_run_dir, cfg.get("num_epochs"), "test",
                       test_loss, test_acc, test_prec, test_rec, test_f1,
                       cdr_loss=L.get("cdr"), chain_loss=L.get("chain"),
                       chain_acc=getattr(trainer, "last_chain_acc", None))

    if cfg.get("save_test_predictions", True):
        predict_and_save(
            trainer     = trainer,
            loader      = dl_test,
            save_path   = os.path.join(fold_run_dir, "test_predictions.csv.gz"),
        )

    return test_loss, test_acc, test_prec, test_rec, test_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold",   type=int, default=None, help="단일 fold 실행")
    parser.add_argument("--chain",  default=None,
                        help="heavy | light | both (config 값을 덮어씀)")
    parser.add_argument("--resume", action="store_true",
                        help="fold run_dir 의 마지막 checkpoint 에서 이어서 학습")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.chain:
        cfg["chain"] = args.chain
    chain = cfg.get("chain", "heavy")

    run_dir   = os.path.join(cfg["save_dir"], f"{cfg['run_name']}_{chain}")
    data_dir  = cfg["dataset_dir"]
    n_folds   = cfg.get("kfold", 5)
    qfilter   = cfg.get("quality_filter", "ok")
    dedup     = cfg.get("dedup", False)

    fold_range = [args.fold] if args.fold is not None else range(n_folds)
    results = []

    for fold in fold_range:
        print(f"\n{'='*60}\n  Fold {fold + 1}/{n_folds}  |  chain={chain}\n{'='*60}")

        splits = {}
        for split, cap in [("train", cfg.get("max_train_samples")),
                           ("valid", cfg.get("max_valid_samples")),
                           ("test",  cfg.get("max_test_samples"))]:
            splits[split] = load_oas_fold(
                dataset_dir    = data_dir,
                chain          = chain,
                fold           = fold,
                split          = split,
                quality_filter = qfilter,
                dedup          = dedup,
                max_samples    = cap,
            )
            check_dataframe(splits[split], f"fold{fold}_{split}", cfg)

        fold_run_dir = os.path.join(run_dir, f"fold_{fold}")
        tl, ta, tp, tr, tf = run_fold(cfg, fold_run_dir,
                                      splits["train"], splits["valid"], splits["test"],
                                      resume=args.resume)
        results.append({"fold": fold, "loss": tl, "acc": ta,
                        "prec": tp, "rec": tr, "f1": tf})

    # ── Summary ───────────────────────────────────────────────────────────────
    if len(results) > 1:
        print(f"\n{'='*60}\n  {n_folds}-Fold CV Summary\n{'='*60}")
        for r in results:
            print(f"  Fold {r['fold']+1}: "
                  f"loss={r['loss']:.4f} acc={r['acc']:.4f} "
                  f"prec={r['prec']:.4f} rec={r['rec']:.4f} f1={r['f1']:.4f}")
        for k in ["loss", "acc", "prec", "rec", "f1"]:
            vals = [r[k] for r in results if r[k] is not None]
            print(f"  {k}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

        os.makedirs(run_dir, exist_ok=True)
        pd.DataFrame(results).to_csv(os.path.join(run_dir, "kfold_summary.csv"), index=False)
        print(f"\n  Summary → {os.path.join(run_dir, 'kfold_summary.csv')}")


if __name__ == "__main__":
    main()
