"""Save one HIV test-pair BAN interaction map as a PNG and CSV.

Example:
  python visualize_ban_map.py --split unseen --seed 2 --sample-index 17
"""

import argparse
import hashlib
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from model import AbLangAbAgModel, exclude_mask
from prepare_data import load_hiv_seed
from embedding_cache import chain_max_lens, resolve_cache_dir


def load_embedding(cache_dir, role, seq, max_len):
    key = hashlib.sha1(f"{seq}|{max_len}".encode()).hexdigest()
    path = os.path.join(cache_dir, f"{role}_emb.npz")
    with np.load(path, allow_pickle=False) as data:
        hidden = torch.from_numpy(data[f"{key}_hidden"]).float()
        mask = torch.from_numpy(data[f"{key}_mask"]).bool()
    return hidden.unsqueeze(0), mask.unsqueeze(0)


def build_model(cfg):
    m = cfg["model"]
    return AbLangAbAgModel(
        ag_embed_dim=m.get("ag_embed_dim", 640),
        num_classes=m.get("num_classes", 1), dropout=m.get("dropout", 0.1),
        co_attn_heads=m.get("co_attn_heads", 1), ban_k=m.get("ban_k", 3),
        ban_act=m.get("ban_act", "ReLU"), ban_dropout=m.get("ban_dropout", 0.2),
        use_vl=True, proj_dim=m.get("proj_dim", 256),
        use_interaction_residual=m.get("use_interaction_residual", True),
        ban_pool_dim=m.get("ban_h_dim", m.get("ban_pool_dim", 256)),
        ban_softmax=m.get("ban_softmax", False),
        classification_hidden=m.get("classification_hidden", 256),
        classification_out=m.get("classification_out", 128),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "HIV.yaml"))
    ap.add_argument("--split", choices=["unseen", "ag_unseen", "ab_unseen"], default="unseen")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--sample-index", type=int, default=None,
                    help="test_predictions.csv/test split row index; default: highest-confidence true positive")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_root = cfg["data"].get("data_root", "../data/ab_ag")
    if not os.path.isabs(data_root):
        data_root = os.path.normpath(os.path.join(HERE, data_root))
    _, _, test_df = load_hiv_seed(args.seed, split_type=args.split, data_root=data_root)

    run_dir = os.path.join(HERE, "runs", "256_head1", "HIV", args.split, f"seed_{args.seed}")
    pred_df = pd.read_csv(os.path.join(run_dir, "test_predictions.csv"))
    if len(pred_df) != len(test_df):
        raise RuntimeError("prediction CSV and test dataframe row counts differ")
    if args.sample_index is None:
        candidates = pred_df[(pred_df.label == 1) & (pred_df.pred == 1)]
        if candidates.empty:
            raise RuntimeError("no true-positive test sample; specify --sample-index")
        sample_index = int(candidates.prob_pos.idxmax())
    else:
        sample_index = args.sample_index
    row, pred = test_df.iloc[sample_index], pred_df.iloc[sample_index]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg)
    ckpt = torch.load(os.path.join(run_dir, "best_model.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device).eval()

    vh_len, vl_len, ag_len = chain_max_lens(cfg, "HIV")
    cache_dir = resolve_cache_dir(cfg, "HIV", HERE)
    vh, vh_mask = load_embedding(cache_dir, "vh", row.vh_seq, vh_len)
    vl, vl_mask = load_embedding(cache_dir, "vl", row.vl_seq, vl_len)
    ag, ag_mask = load_embedding(cache_dir, "ag", row.ag_seq, ag_len)
    vh, vh_mask = vh.to(device), vh_mask.to(device)
    vl, vl_mask = vl.to(device), vl_mask.to(device)
    ag, ag_mask = ag.to(device), ag_mask.to(device)

    with torch.no_grad():
        logits, info = model(vh, vh_mask, vl, vl_mask, ag, ag_mask, return_attn=True)
    interaction = info["interaction_map"][0].float().cpu()
    ag_idx = (~exclude_mask(ag_mask)[0]).nonzero(as_tuple=True)[0].cpu()
    vh_idx = (~exclude_mask(vh_mask)[0]).nonzero(as_tuple=True)[0].cpu()
    vl_idx = (~exclude_mask(vl_mask)[0]).nonzero(as_tuple=True)[0].cpu() + vh_len
    ag_vh = interaction[ag_idx][:, vh_idx].numpy()
    ag_vl = interaction[ag_idx][:, vl_idx].numpy()
    probability = torch.sigmoid(logits.float()).item()

    out_dir = args.out_dir or os.path.join(run_dir, "attention_maps")
    os.makedirs(out_dir, exist_ok=True)
    stem = f"{args.split}_seed{args.seed}_sample{sample_index}"
    scale = max(np.percentile(np.abs(np.concatenate([ag_vh.ravel(), ag_vl.ravel()])), 99), 1e-8)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True, constrained_layout=True)
    for ax, matrix, title in zip(axes, (ag_vh, ag_vl), ("Antigen × VH", "Antigen × VL")):
        image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-scale, vmax=scale)
        ax.set_title(title)
        ax.set_xlabel("Antibody residue index")
    axes[0].set_ylabel("Antigen residue index")
    fig.colorbar(image, ax=axes, label="raw BAN interaction score")
    fig.suptitle(
        f"HIV {args.split}, seed {args.seed}, sample {sample_index} | "
        f"label={int(row.label)}, prediction={int(pred.pred)}, P(bind)={probability:.4f}")
    png_path = os.path.join(out_dir, f"{stem}.png")
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    pairs = []
    for chain, matrix in (("VH", ag_vh), ("VL", ag_vl)):
        n_pairs = min(20, matrix.size)
        for flat_index in np.argpartition(matrix.ravel(), -n_pairs)[-n_pairs:]:
            ag_i, ab_j = np.unravel_index(flat_index, matrix.shape)
            pairs.append((chain, int(ag_i) + 1, int(ab_j) + 1, float(matrix[ag_i, ab_j])))
    pair_df = pd.DataFrame(pairs, columns=["chain", "ag_residue", "ab_residue", "score"])
    pair_df.sort_values("score", ascending=False).head(20).to_csv(
        os.path.join(out_dir, f"{stem}_top_pairs.csv"), index=False)
    print(f"saved: {png_path}")
    print(pair_df.sort_values("score", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
