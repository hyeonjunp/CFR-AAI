"""
Trainer for ESM2-based Ab-Ag interaction prediction.
Binary (num_classes=1) : BCEWithLogitsLoss
Multi-class (>1)        : CrossEntropyLoss
"""

import os
import csv
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    f1_score,
    confusion_matrix,
    classification_report,
)


class ESMAbAgTrainer:

    def __init__(
        self,
        model: nn.Module,
        num_classes:    int   = 1,
        lr:             float = 1e-4,
        weight_decay:   float = 1e-3,
        max_grad_norm:  float = 1.0,
        pos_weight:     float = 1.0,
        class_weights         = None,
        train_loader          = None,
        val_loader            = None,
        num_epochs:     int   = 10,
        final_loss_weight: float = 0.0,
        branch_loss_weight: float = 1.0,
    ):
        self.num_classes   = num_classes
        self.binary        = (num_classes == 1)
        self.max_grad_norm = max_grad_norm
        self.train_loader  = train_loader
        self.val_loader    = val_loader
        self.num_epochs    = num_epochs
        self.start_epoch   = 0
        self.final_loss_weight = final_loss_weight
        self.branch_loss_weight = branch_loss_weight

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # bf16 autocast: frozen ESM2 forward + trainable layers run faster with
        # ~no accuracy cost on Ampere/Ada/Blackwell tensor cores; no GradScaler needed.
        self.amp_dtype = torch.bfloat16 if self.device.type == "cuda" else None

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
            model = nn.DataParallel(model)
        self.model = model.to(self.device)

        # ── Optimizer: ESM_CDR_pred (esm+attn_block+classifier) is entirely frozen —
        #    no fine-tuning — so a single param group covers everything trainable
        #    (co-attention layers, ResMLPBlock, regressor).
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optim = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

        # ── Loss ──────────────────────────────────────────────────────────────
        if self.binary:
            pw = torch.tensor([pos_weight], dtype=torch.float, device=self.device)
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            cw = torch.tensor(class_weights, dtype=torch.float, device=self.device) \
                 if class_weights else None
            self.loss_fn = nn.CrossEntropyLoss(weight=cw)

        total_steps = max(1, len(train_loader) if train_loader else 1) * num_epochs
        self.scheduler = CosineAnnealingLR(self.optim, T_max=total_steps, eta_min=1e-6)

        self.best_val_metric = -1.0
        self.best_epoch      = -1

    def _unwrap(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _forward(self, batch, return_branches=False):
        vh_hidden = batch["vh_hidden"].to(self.device)
        vh_mask   = batch["vh_attn_mask"].to(self.device)
        # VHH(나노바디) 데이터셋은 배치에 VL 자체가 없다 — 모델도 use_vl=False로 만들어진다.
        vl_hidden = batch["vl_hidden"].to(self.device)    if "vl_hidden"    in batch else None
        vl_mask   = batch["vl_attn_mask"].to(self.device) if "vl_attn_mask" in batch else None
        ag_emb    = batch["ag_emb"].to(self.device)
        ag_mask   = batch["ag_attn_mask"].to(self.device)
        labels    = batch["label"].to(self.device)

        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                             enabled=self.amp_dtype is not None):
            logits, embedding_logits, map_logits = self.model(
                vh_hidden, vh_mask, vl_hidden, vl_mask, ag_emb, ag_mask,
                return_branch_logits=True,
            )

            if self.binary:
                if map_logits is not None:
                    embedding_loss = self.loss_fn(
                        embedding_logits.squeeze(-1).float(), labels)
                    ban_loss = self.loss_fn(map_logits.squeeze(-1).float(), labels)
                    branch_loss = embedding_loss + ban_loss
                    final_loss = self.loss_fn(logits.squeeze(-1).float(), labels)
                    loss = (self.branch_loss_weight * branch_loss
                            + self.final_loss_weight * final_loss)
                else:
                    loss = self.loss_fn(logits.squeeze(-1).float(), labels)
            else:
                loss = self.loss_fn(logits.float(), labels)

        if return_branches:
            return loss, logits, labels, embedding_logits, map_logits
        return loss, logits, labels

    # ── Train ─────────────────────────────────────────────────────────────────

    def train_epoch(self):
        self.model.train()
        total_loss, n = 0.0, 0
        pbar = tqdm(self.train_loader, desc="Train", leave=False)
        for batch in pbar:
            loss, _, _ = self._forward(batch)
            self.optim.zero_grad()
            loss.backward()
            if self.max_grad_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optim.step()
            self.scheduler.step()
            total_loss += loss.item()
            n += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        return total_loss / max(n, 1)

    # ── Validate ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self, loader=None, name="val"):
        loader = loader or self.val_loader
        if loader is None:
            return None

        self.model.eval()
        torch.cuda.empty_cache()  # release fragmented blocks from train_epoch before eval-size allocs
        t0 = time.time()
        total_loss, n = 0.0, 0
        all_labels, all_probs, all_preds = [], [], []
        embedding_probs, embedding_preds = [], []

        for batch in tqdm(loader, desc=name, leave=False):
            loss, logits, labels, embedding_logits, _ = self._forward(
                batch, return_branches=True)
            total_loss += loss.item()
            n += 1

            if self.binary:
                probs = torch.sigmoid(logits.squeeze(-1))
                preds = (probs >= 0.5).long()
                emb_probs = torch.sigmoid(embedding_logits.squeeze(-1))
                emb_preds = (emb_probs >= 0.5).long()
            else:
                probs = torch.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1)
                emb_probs = torch.softmax(embedding_logits, dim=-1)
                emb_preds = embedding_logits.argmax(dim=-1)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            embedding_preds.extend(emb_preds.cpu().tolist())
            embedding_probs.extend(emb_probs.cpu().tolist())

        avg_loss = total_loss / max(n, 1)
        metrics  = self._compute_metrics(all_labels, all_preds, all_probs)
        embedding_metrics = self._compute_metrics(
            all_labels, embedding_preds, embedding_probs)
        elapsed  = time.time() - t0
        self._print_metrics(name, avg_loss, metrics, elapsed)
        self._print_metrics(f"{name}/embedding-only", avg_loss,
                            embedding_metrics, elapsed)
        self.model.train()
        return {"loss": avg_loss, **metrics,
                **{f"embedding_only_{k}": v
                   for k, v in embedding_metrics.items()}}

    def _compute_metrics(self, labels, preds, probs):
        if not labels:
            return {}
        acc = accuracy_score(labels, preds)
        if self.binary:
            prec, rec, f1, _ = precision_recall_fscore_support(
                labels, preds, average="binary", pos_label=1, zero_division=0)
            try:
                auroc = roc_auc_score(labels, probs)
            except ValueError:
                auroc = float("nan")
            try:
                auprc = average_precision_score(labels, probs)
            except ValueError:
                auprc = float("nan")
            mcc = matthews_corrcoef(labels, preds)
            return {"acc": acc, "precision": prec, "recall": rec, "f1": f1,
                    "auroc": auroc, "auprc": auprc, "mcc": mcc}
        else:
            return {
                "acc":         acc,
                "macro_f1":    f1_score(labels, preds, average="macro",    zero_division=0),
                "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
            }

    def _print_metrics(self, name, loss, metrics, elapsed):
        parts = [f"[{name}] loss: {loss:.4f}"]
        for k, v in metrics.items():
            if v is not None:
                parts.append(f"{k}: {v:.4f}")
        parts.append(f"{elapsed:.1f}s")
        print(" | ".join(parts))

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def _state(self, epoch):
        return {
            "model_state":     self._unwrap().state_dict(),
            "optimizer_state": self.optim.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "epoch":           epoch,
        }

    def save_checkpoint(self, run_dir, epoch):
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, "checkpoint_last.pt")
        torch.save(self._state(epoch), path)
        print(f"Checkpoint: {path}")

    def save_best(self, run_dir, epoch, metric_val):
        os.makedirs(run_dir, exist_ok=True)
        path  = os.path.join(run_dir, "best_model.pt")
        state = self._state(epoch)
        state["best_metric"] = metric_val
        torch.save(state, path)
        print(f"  >> Best model saved (epoch={epoch}, metric={metric_val:.4f}): {path}")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self._unwrap().load_state_dict(ckpt["model_state"])
        self.optim.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.start_epoch = int(ckpt.get("epoch", 0))
        print(f"Resumed: {path}  next epoch={self.start_epoch + 1}")

    def load_best_checkpoint(self, run_dir):
        path = os.path.join(run_dir, "best_model.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"best_model.pt not found: {path}")
        self.load_checkpoint(path)

    def load_latest_checkpoint(self, run_dir):
        path = os.path.join(run_dir, "checkpoint_last.pt")
        if os.path.isfile(path):
            self.load_checkpoint(path)

    # ── CSV logging ───────────────────────────────────────────────────────────

    def log_to_csv(self, run_dir, epoch, split, loss, metrics: dict):
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, "training_log.csv")
        row  = {"epoch": epoch, "split": split,
                "loss":  f"{loss:.6f}" if loss is not None else ""}
        for k, v in (metrics or {}).items():
            row[k] = f"{v:.6f}" if v is not None else ""

        if os.path.isfile(path):
            # 기존 파일의 컬럼 스키마가 다를 수 있으므로(예: 새 지표 추가) 합쳐서 헤더를 통일
            with open(path, newline="") as f:
                existing_rows = list(csv.DictReader(f))
            fieldnames = list(dict.fromkeys(
                [k for r in existing_rows for k in r] + list(row.keys())
            ))
            existing_rows.append(row)
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(existing_rows)
        else:
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                w.writeheader()
                w.writerow(row)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def fit(
        self,
        run_dir:               str,
        epochs:                int  = None,
        validate_every:        int  = 1,
        best_metric_key:       str  = "auroc",
        resume:                bool = False,
        early_stopping_patience:      int = 0,
    ):
        epochs = epochs or self.num_epochs
        os.makedirs(run_dir, exist_ok=True)
        if resume:
            self.load_latest_checkpoint(run_dir)

        no_improve = 0
        current    = self.start_epoch

        for epoch in range(self.start_epoch, epochs):
            current = epoch + 1
            print(f"\n{'='*60}\nEpoch {current}/{epochs}")

            train_loss = self.train_epoch()
            print(f"[train] loss: {train_loss:.4f}")
            self.log_to_csv(run_dir, current, "train", train_loss, {})

            val_metrics = None
            if self.val_loader and current % validate_every == 0:
                val_metrics = self.validate(name="val")
                if val_metrics:
                    val_loss = val_metrics.pop("loss")
                    self.log_to_csv(run_dir, current, "val", val_loss, val_metrics)

                    metric_val = val_metrics.get(best_metric_key, 0.0) or 0.0
                    if metric_val > self.best_val_metric:
                        self.best_val_metric = metric_val
                        self.best_epoch      = current
                        self.save_best(run_dir, current, metric_val)
                        no_improve = 0
                    else:
                        no_improve += 1

                    if early_stopping_patience > 0 and no_improve >= early_stopping_patience:
                        print(f"\nEarly stopping (no {best_metric_key} improvement, "
                              f"patience={early_stopping_patience})")
                        break

        self.save_checkpoint(run_dir, current)

        print(f"\nDone. Best epoch={self.best_epoch}, "
              f"best {best_metric_key}={self.best_val_metric:.4f}")

    # ── Evaluation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, loader, run_dir=None, name="test", verbose=True):
        import pandas as pd
        self.model.eval()
        torch.cuda.empty_cache()  # release fragmented blocks from training before eval-size allocs
        all_labels, all_probs, all_preds = [], [], []
        embedding_probs, embedding_preds = [], []

        for batch in tqdm(loader, desc=f"Evaluate [{name}]"):
            _, logits, labels, embedding_logits, _ = self._forward(
                batch, return_branches=True)
            if self.binary:
                probs = torch.sigmoid(logits.squeeze(-1))
                preds = (probs >= 0.5).long()
                emb_probs = torch.sigmoid(embedding_logits.squeeze(-1))
                emb_preds = (emb_probs >= 0.5).long()
            else:
                probs = torch.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1)
                emb_probs = torch.softmax(embedding_logits, dim=-1)
                emb_preds = embedding_logits.argmax(dim=-1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            embedding_preds.extend(emb_preds.cpu().tolist())
            embedding_probs.extend(emb_probs.cpu().tolist())

        metrics = self._compute_metrics(all_labels, all_preds, all_probs)
        embedding_metrics = self._compute_metrics(
            all_labels, embedding_preds, embedding_probs)
        print(f"\n{'='*60}\n[{name}] Evaluation")
        for k, v in metrics.items():
            print(f"  {k:<20}: {v:.4f}")
        print(f"\n[{name}/embedding-only ablation]")
        for k, v in embedding_metrics.items():
            print(f"  {k:<20}: {v:.4f}")

        if verbose and self.binary:
            print("\nConfusion Matrix:")
            print(confusion_matrix(all_labels, all_preds))
            print("\nClassification Report:")
            print(classification_report(all_labels, all_preds,
                                        target_names=["Non-binding", "Binding"],
                                        zero_division=0))

        if run_dir:
            os.makedirs(run_dir, exist_ok=True)
            logged_metrics = dict(metrics)
            logged_metrics.update({f"embedding_only_{k}": v
                                   for k, v in embedding_metrics.items()})
            self.log_to_csv(run_dir, self.best_epoch, name, None, logged_metrics)
            out = os.path.join(run_dir, f"{name}_predictions.csv")
            if self.binary:
                pd.DataFrame({
                    "label": all_labels,
                    "pred": all_preds,
                    "prob_pos": all_probs,
                    "embedding_only_pred": embedding_preds,
                    "embedding_only_prob_pos": embedding_probs,
                }).to_csv(out, index=False)
            print(f"Predictions saved: {out}")

        self.model.train()
        metrics.update({f"embedding_only_{k}": v
                        for k, v in embedding_metrics.items()})
        return metrics
