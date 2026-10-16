"""
AbLang-based Ab-Ag Interaction Prediction — Training Script
(frozen-CDR variant: AbLang 백본도 attn_block 도 fine-tuning 하지 않는다, model.py 참고)

임베딩은 build_embeddings.py 가 만들어 둔 npz 에서만 읽는다 — 여기서는 AbLang/ESM2 를
아예 올리지 않는다 (CDR_ESM150M 의 --prewarm 옵션이 여기엔 없는 이유: 체인마다 인코더가
달라 3개를 다 올려야 하는데, 학습에는 하나도 필요 없다).

Usage
-----
    python build_embeddings.py --config configs/HIV.yaml --dataset HIV    # 먼저 1회
    python train.py --config configs/HIV.yaml --dataset HIV --seed 0
    python train.py --config configs/HIV.yaml --dataset HIV/ab_unseen --seed 0
    python train.py --config configs/HIV.yaml --dataset il6 --all
    python train.py --config configs/HIV.yaml --dataset SAbDab --fold 0

Omitting --fold/--seed runs all folds/seeds (0..N_SPLITS-1) by default.
"""

import argparse
import os
import sys
import random

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from dataprocessing import AbAgCachedDataset
from embedding_cache import EmbeddingCache, chain_max_lens, resolve_cache_dir
from model import AbLangAbAgModel
from Trainer import ESMAbAgTrainer
from prepare_data import (
    load_sabdab_fold, load_hiv_seed, load_covabdab_seed, load_avida_seed,
    load_avida_sars_cov2_seed, dataset_has_vl, N_SPLITS, DATA_ROOT,
)
from build_embeddings import split_dataset_arg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_embeddings(cache: EmbeddingCache, dfs, vh_max_len, vl_max_len, ag_max_len,
                    use_vl: bool = True):
    """build_embeddings.py가 만들어 둔 {vh,vl,ag}_emb.npz 에서 임베딩을 읽어온다.

    npz에 없는 서열이 있으면 학습을 시작하지 않고 에러로 알려준다. train/val/test가 하나의
    npz를 공유하므로 겹치는 서열은 한 번만 들어 있다. VH/VL은 CDR attn_block을 통과한
    post-CDR residue embedding이다."""
    dfs = [d for d in dfs if d is not None]
    if not dfs:
        return
    all_df = pd.concat(dfs, ignore_index=True)
    cache.require(all_df["vh_seq"].tolist(), vh_max_len, "vh")
    if use_vl:
        cache.require(all_df["vl_seq"].tolist(), vl_max_len, "vl")
    cache.require(all_df["ag_seq"].tolist(), ag_max_len, "ag")


def make_loader(df: pd.DataFrame, cfg: dict, binary: bool, shuffle: bool, dataset: str,
                embeddings: dict, use_vl: bool = True) -> DataLoader:
    t  = cfg["train"]
    bs = t["batch_size"] if shuffle else t["valid_batch_size"]
    vh_max_len, vl_max_len, ag_max_len = chain_max_lens(cfg, dataset)
    ds = AbAgCachedDataset(
        df,
        embeddings  = embeddings,
        vh_max_len  = vh_max_len,
        vl_max_len  = vl_max_len,
        ag_max_len  = ag_max_len,
        vh_col      = "vh_seq",
        vl_col      = "vl_seq",
        ag_col      = "ag_seq",
        label_col   = "label",
        binary      = binary,
        use_vl      = use_vl,
    )
    num_workers = t.get("num_workers", 4)
    return DataLoader(
        ds, batch_size=bs, shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
        persistent_workers=num_workers > 0,   # skip worker respawn cost every epoch
        prefetch_factor=4 if num_workers > 0 else None,
    )


def build_model(cfg: dict, use_vl: bool = True) -> AbLangAbAgModel:
    m = cfg["model"]
    return AbLangAbAgModel(
        ag_embed_dim      = m.get("ag_embed_dim", 640),
        num_classes       = m.get("num_classes", 1),
        dropout           = m.get("dropout", 0.1),
        co_attn_heads     = m.get("co_attn_heads", 1),
        ban_k             = m.get("ban_k", 3),
        ban_act           = m.get("ban_act", "ReLU"),
        ban_dropout       = m.get("ban_dropout", 0.2),
        use_vl            = use_vl,
        proj_dim          = m.get("proj_dim", 256),
        use_interaction_residual = m.get("use_interaction_residual", True),
        use_ban_pooling       = m.get("use_ban_pooling", True),
        ban_pool_dim          = m.get("ban_h_dim", m.get("ban_pool_dim", 256)),
        ban_softmax           = m.get("ban_softmax", False),
        classification_hidden = m.get("classification_hidden", 256),
        classification_out    = m.get("classification_out", 128),
        use_ban_logit_gate    = m.get("use_ban_logit_gate", False),
    )


def run_one(df_train, df_val, df_test, cfg, run_dir, binary, resume, test_only, split_name,
            dataset, cache: EmbeddingCache):
    print(f"\n{'='*60}\n  {split_name}")
    print(f"  train={len(df_train)}  "
          f"val={len(df_val) if df_val is not None else '-'}  "
          f"test={len(df_test) if df_test is not None else '-'}")
    print(f"  run_dir: {run_dir}\n{'='*60}")

    vh_max_len, vl_max_len, ag_max_len = chain_max_lens(cfg, dataset)

    # 나노바디(VHH) 데이터셋은 light chain이 없다 — VL 임베딩도, VL 스트림도 만들지 않는다.
    use_vl = dataset_has_vl(dataset)
    if not use_vl:
        print(f"[data] {dataset}: 나노바디(VHH) — VL 없이 VH/AG만 사용합니다")

    # npz 로드를 모델 생성보다 먼저 해서, 서열이 빠져 있으면 체크포인트를 읽기 전에 실패한다.
    load_embeddings(cache, [df_train, df_val, df_test],
                    vh_max_len, vl_max_len, ag_max_len, use_vl=use_vl)
    embeddings = cache.state_dict()

    model = build_model(cfg, use_vl=use_vl)

    dl_train = make_loader(df_train, cfg, binary, shuffle=True,  dataset=dataset, embeddings=embeddings, use_vl=use_vl)
    dl_val   = make_loader(df_val,   cfg, binary, shuffle=False, dataset=dataset, embeddings=embeddings, use_vl=use_vl) if df_val  is not None else None
    dl_test  = make_loader(df_test,  cfg, binary, shuffle=False, dataset=dataset, embeddings=embeddings, use_vl=use_vl) if df_test is not None else None

    if dl_val is None and dl_test is not None:
        dl_val = dl_test

    t = cfg["train"]
    trainer = ESMAbAgTrainer(
        model           = model,
        num_classes     = cfg["model"].get("num_classes", 1),
        lr              = t.get("lr", 1e-4),
        weight_decay    = t.get("weight_decay", 1e-3),
        max_grad_norm   = t.get("max_grad_norm", 1.0),
        pos_weight      = t.get("pos_weight", 1.0),
        class_weights   = t.get("class_weights"),
        train_loader    = dl_train,
        val_loader      = dl_val,
        num_epochs      = t.get("num_epochs", 20),
        final_loss_weight = t.get("final_loss_weight", 0.0),
        branch_loss_weight = t.get("branch_loss_weight", 1.0),
    )

    if test_only:
        trainer.load_best_checkpoint(run_dir)
    else:
        trainer.fit(
            run_dir                  = run_dir,
            epochs                   = t.get("num_epochs", 20),
            validate_every           = t.get("validate_every", 1),
            best_metric_key          = t.get("best_metric_key", "auroc" if binary else "macro_f1"),
            resume                   = resume,
            early_stopping_patience  = t.get("early_stopping_patience", 0),
        )
        trainer.load_best_checkpoint(run_dir)

    test_metrics = None
    if dl_test is not None:
        test_metrics = trainer.evaluate(dl_test, run_dir=run_dir, name="test")
    return test_metrics


def save_summary(results: list, out_dir: str, label: str):
    import csv
    if not results:
        return
    keys = [k for k in results[0] if results[0][k] is not None]
    rows = [{"run": i, **{k: r.get(k) for k in keys}} for i, r in enumerate(results)]
    means = {"run": "mean±std"}
    for k in keys:
        vals = [r[k] for r in results if r.get(k) is not None]
        if vals:
            means[k] = f"{np.mean(vals):.4f}±{np.std(vals):.4f}"
    rows.append(means)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run"] + keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[{label}] Summary → {os.path.join(out_dir, 'summary.csv')}\n  {means}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default=os.path.join(_HERE, "configs", "HIV.yaml"))
    parser.add_argument("--dataset",    default=None,
                        help="SAbDab | HIV | CoVAbDab | il6 "
                             "(HIV/ab_unseen 처럼 split_type을 붙여 써도 됨)")
    parser.add_argument("--split_type", default=None,
                        help="HIV 전용: unseen | ag_unseen | ab_unseen")
    parser.add_argument("--seed",       type=int, default=None)
    parser.add_argument("--fold",       type=int, default=None)
    parser.add_argument("--all",        action="store_true")
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--test_only",  action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    dataset, split_from_arg = split_dataset_arg(args.dataset or cfg["data"].get("dataset", "HIV"))
    binary    = cfg["model"].get("num_classes", 1) == 1
    run_all   = args.all or cfg["data"].get("all", False)
    save_dir  = cfg["train"].get("save_dir", "./runs")
    if not os.path.isabs(save_dir):
        save_dir = os.path.normpath(os.path.join(_HERE, save_dir))
    data_root = cfg["data"].get("data_root", DATA_ROOT)
    if not os.path.isabs(data_root):
        data_root = os.path.normpath(os.path.join(_HERE, data_root))
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"data.data_root does not exist: {data_root}")
    print(f"[data] data_root={data_root}")

    # ── 임베딩 npz — 전체 run이 하나의 인스턴스/폴더를 공유한다
    #    (상위 공유 캐시 emb/<dataset>/{vh,vl,ag}_emb.npz — build_embeddings.py가 생성).
    cache_dir = resolve_cache_dir(cfg, dataset, _HERE, create=False)
    if not os.path.isdir(cache_dir):
        raise SystemExit(
            f"임베딩 폴더가 없습니다: {cache_dir}\n"
            f"  → python build_embeddings.py --config {args.config} --dataset {dataset}")
    print(f"[EmbeddingCache] emb_dir={cache_dir}  (npz 로드 전용, 인코더 forward 없음)")
    cache = EmbeddingCache(encoders=None, cache_dir=cache_dir)

    HIV_ALL = ["unseen", "ag_unseen", "ab_unseen"]

    if dataset == "HIV":
        if args.split_type or split_from_arg:
            selected_types = [args.split_type or split_from_arg]
        elif run_all:
            selected_types = HIV_ALL
        else:
            selected_types = [cfg["data"].get("split_type", "unseen")]
        indices = [args.seed] if args.seed is not None else list(range(N_SPLITS))
        jobs = [(st, indices) for st in selected_types]
    elif dataset == "SAbDab":
        indices = [args.fold] if args.fold is not None else list(range(N_SPLITS))
        jobs = [(None, indices)]
    else:
        indices = [args.seed] if args.seed is not None else list(range(N_SPLITS))
        jobs = [(None, indices)]

    hiv_results   = {st: [] for st in HIV_ALL}
    other_results = []

    for split_type, indices in jobs:
        for idx in indices:
            if dataset == "SAbDab":
                df_train, df_val, df_test = load_sabdab_fold(idx, data_root=data_root)
                split_name = f"SAbDab | fold={idx}"
                run_dir    = os.path.join(save_dir, "SAbDab", f"fold_{idx}")
            elif dataset == "HIV":
                df_train, df_val, df_test = load_hiv_seed(idx, split_type=split_type, data_root=data_root)
                split_name = f"HIV | {split_type} | seed={idx}"
                run_dir    = os.path.join(save_dir, "HIV", split_type, f"seed_{idx}")
            elif dataset == "CoVAbDab":
                df_train, df_val, df_test = load_covabdab_seed(idx, data_root=data_root)
                split_name = f"CoVAbDab | seed={idx}"
                run_dir    = os.path.join(save_dir, "CoVAbDab", f"seed_{idx}")
            elif dataset == "AVIDa_SARS_CoV_2":
                df_train, df_val, df_test = load_avida_sars_cov2_seed(idx, data_root=data_root)
                split_name = f"AVIDa_SARS_CoV_2 | seed={idx}"
                run_dir    = os.path.join(save_dir, "AVIDa_SARS_CoV_2", f"seed_{idx}")
            else:
                df_train, df_val, df_test = load_avida_seed(idx, data_root=data_root)
                split_name = f"il6 | seed={idx}"
                run_dir    = os.path.join(save_dir, "il6", f"seed_{idx}")

            metrics = run_one(
                df_train=df_train, df_val=df_val, df_test=df_test,
                cfg=cfg, run_dir=run_dir, binary=binary,
                resume=args.resume, test_only=args.test_only,
                split_name=split_name, dataset=dataset, cache=cache,
            )
            # release this job's model/trainer + freed CUDA allocator blocks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if metrics:
                if dataset == "HIV":
                    hiv_results[split_type].append(metrics)
                else:
                    other_results.append(metrics)

    if dataset == "HIV":
        for st, res in hiv_results.items():
            if len(res) > 1:
                save_summary(res, os.path.join(save_dir, "HIV", st), f"HIV/{st}")
    elif len(other_results) > 1:
        save_summary(other_results, os.path.join(save_dir, dataset), dataset)


if __name__ == "__main__":
    main()
