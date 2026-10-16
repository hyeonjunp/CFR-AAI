"""
Trainer for ABlang-based CDR prediction — OAS_paired 버전, 7-class region + VH/VL.

`model_OAS/trainer.py`와 학습 로직은 동일하다. 차이는 `_forward`가 batch의
`chain_type`(list[str])을 모델에 함께 넘겨 heavy/light 백본으로 라우팅한다는 것과,
ESM 파라미터 lr 그룹을 나누던 부분이 `raw.esm` 대신 `raw.backbones`(heavy/light
frozen 백본 ModuleDict)를 기준으로 한다는 것뿐이다.
"""

import os
import csv
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from dataprocessing import REGION_NAMES, NUM_REGIONS, CDR_REGION_IDS


def _prf(tp: int, fp: int, fn: int):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


class ABlangCDRTrainer:

    def __init__(
        self,
        model,
        lr: float               = 1e-4,
        weight_decay: float     = 1e-3,
        max_grad_norm: float    = 1.0,
        pos_weight: float       = 3.0,    # region CE에서 CDR1/2/3 클래스에 주는 가중치
        train_loader             = None,
        val_loader               = None,
        num_epochs: int         = 10,
        amp: bool                = True,
        amp_dtype: str          = "bf16",
        chain_loss_weight: float = 1.0,
    ):
        self.train_loader  = train_loader
        self.val_loader    = val_loader
        self.num_epochs    = num_epochs
        self.max_grad_norm = max_grad_norm
        self.start_epoch   = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── AMP ───────────────────────────────────────────────────────────────
        self.amp_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
        self.amp = bool(amp) and self.device.type == "cuda"
        # fp16 만 스케일러가 필요하다 (bf16 은 fp32 와 지수부가 같아 언더플로 없음)
        self.scaler = torch.amp.GradScaler("cuda",
                                           enabled=self.amp and self.amp_dtype is torch.float16)
        if self.amp:
            print(f"AMP: enabled ({amp_dtype})")

        if torch.cuda.device_count() > 1:
            # 주의: DataParallel은 텐서만 배치 축으로 scatter한다. forward의
            # `chain_types`(list[str])는 그대로 각 replica에 복제되어 전달되므로
            # chain="both"(멀티 GPU)에서는 라우팅 인덱스가 replica의 부분 배치와
            # 맞지 않아 깨진다. 이 저장소는 단일 GPU 학습을 가정한다.
            print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
            model = nn.DataParallel(model)
        self.model = model.to(self.device)

        # ── Optimizer: ABlang backbone params get lr/10 if any are trainable ─
        raw = self._unwrap()
        backbone_ids = set(id(p) for p in raw.backbones.parameters())

        backbone_params = [p for p in self.model.parameters()
                           if p.requires_grad and id(p) in backbone_ids]
        other_params    = [p for p in self.model.parameters()
                           if p.requires_grad and id(p) not in backbone_ids]

        param_groups = [{"params": other_params, "lr": lr}]
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": lr * 0.1})
            print(f"ABlang LR: {lr * 0.1:.2e}  |  Head LR: {lr:.2e}")
        self.optim = AdamW(param_groups, lr=lr, weight_decay=weight_decay)

        # ── Loss ──────────────────────────────────────────────────────────────
        # region: 7-class CrossEntropy (padding/특수토큰은 -100 으로 제외)
        # CDR 클래스(1,3,5)는 FR 보다 토큰 수가 적어 pos_weight 를 클래스 가중치로 옮긴다
        w = torch.ones(NUM_REGIONS, dtype=torch.float, device=self.device)
        for i in CDR_REGION_IDS:
            w[i] = pos_weight
        self.region_loss_fn = nn.CrossEntropyLoss(weight=w, ignore_index=-100)
        self.chain_loss_fn  = nn.BCEWithLogitsLoss()
        self.chain_loss_weight = chain_loss_weight
        print(f"region class weight (CDR): {pos_weight}  |  chain loss weight: {chain_loss_weight}")

        self.best_val_f1  = -1.0
        self.best_epoch   = -1

    def _unwrap(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _autocast(self):
        return torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp)

    def _forward(self, batch):
        input_ids      = batch["input_ids"].to(self.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
        labels         = batch["labels"].to(self.device, non_blocking=True)
        aa_mask        = batch["aa_mask"].to(self.device, non_blocking=True)
        chain_label    = batch["chain_label"].to(self.device, non_blocking=True)
        chain_types    = batch["chain_type"]        # list[str] — CPU에 남겨둔다 (라우팅용)

        with self._autocast():
            region_logits, chain_logits = self.model(
                input_ids=input_ids, attention_mask=attention_mask,
                chain_types=chain_types, aa_mask=aa_mask)

        region_logits = region_logits.float()                      # (B, T, 7)
        # CrossEntropyLoss 는 (N, C) 를 받는다
        loss = self.region_loss_fn(
            region_logits.reshape(-1, region_logits.size(-1)), labels.reshape(-1))

        region_loss = loss
        chain_loss = None
        if chain_logits is not None:
            valid = chain_label != -100
            if valid.any():
                chain_loss = self.chain_loss_fn(
                    chain_logits.float()[valid], chain_label[valid].float())
                loss = loss + self.chain_loss_weight * chain_loss

        # 항목별 손실 (로그용). chain head 가 꺼져 있으면 chain 은 0.0
        parts = {"region": float(region_loss.detach()),
                 "chain":  float(chain_loss.detach()) if chain_loss is not None else 0.0}
        return loss, region_logits, labels, chain_logits, chain_label, parts

    # ── Train ─────────────────────────────────────────────────────────────────

    def train_epoch(self):
        self.model.train()
        total_loss, n = 0.0, 0
        sum_region = sum_chain = 0.0
        pbar = tqdm(self.train_loader, desc="Train", leave=False)
        for batch in pbar:
            loss, _, _, _, _, parts = self._forward(batch)
            sum_region += parts["region"]; sum_chain += parts["chain"]
            self.optim.zero_grad(set_to_none=True)

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                if self.max_grad_norm:
                    self.scaler.unscale_(self.optim)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                loss.backward()
                if self.max_grad_norm:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optim.step()

            total_loss += loss.item()
            n += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             region=f"{parts['region']:.4f}",
                             chain=f"{parts['chain']:.4f}")
        n = max(n, 1)
        return total_loss / n, sum_region / n, sum_chain / n

    # ── Validate ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self, loader=None, name="val", verbose=False):
        """7-class region 지표 + VH/VL chain 정확도.

        반환은 기존과 호환되도록 (loss, acc, prec, rec, f1) 5개.
        f1 은 CDR 을 하나로 묶은 binary F1 (예전 CDR-F1 과 같은 정의) 이라
        binary 학습 결과와 그대로 비교할 수 있다.
        """
        loader = loader or self.val_loader
        if loader is None:
            return None, None, None, None, None

        self.model.eval()
        t0 = time.time()
        total_loss, n = 0.0, 0
        sum_region = sum_chain = 0.0
        # 7x7 혼동행렬을 GPU 에 누적 (행=정답, 열=예측)
        conf = torch.zeros(NUM_REGIONS, NUM_REGIONS, dtype=torch.long, device=self.device)
        chain_ok = torch.zeros(2, dtype=torch.long, device=self.device)   # [맞음, 전체]

        for batch in tqdm(loader, desc=name, leave=False):
            (loss, region_logits, labels,
             chain_logits, chain_label, parts) = self._forward(batch)
            total_loss += loss.item()
            sum_region += parts["region"]; sum_chain += parts["chain"]
            n += 1

            valid = labels != -100
            pred  = region_logits.argmax(dim=-1)[valid]
            gold  = labels[valid]
            conf += torch.bincount(gold * NUM_REGIONS + pred,
                                   minlength=NUM_REGIONS * NUM_REGIONS
                                   ).reshape(NUM_REGIONS, NUM_REGIONS)

            if chain_logits is not None:
                cv = chain_label != -100
                if cv.any():
                    cp = (torch.sigmoid(chain_logits.float()) > 0.5).long()
                    chain_ok[0] += (cp[cv] == chain_label[cv]).sum()
                    chain_ok[1] += cv.sum()

        avg_loss   = total_loss / max(n, 1)
        avg_region = sum_region / max(n, 1)
        avg_chain  = sum_chain  / max(n, 1)
        self.last_losses = {"total": avg_loss, "cdr": avg_region, "chain": avg_chain}
        cm = conf.cpu().numpy()
        total = int(cm.sum())
        if total == 0:
            self.model.train()
            return avg_loss, 0.0, 0.0, 0.0, 0.0

        acc = float(cm.trace()) / total

        # 클래스별 P/R/F1
        per = []
        for i in range(NUM_REGIONS):
            tp = int(cm[i, i]); fp = int(cm[:, i].sum()) - tp; fn = int(cm[i, :].sum()) - tp
            per.append(_prf(tp, fp, fn) + (int(cm[i, :].sum()),))
        macro_f1 = sum(p[2] for p in per) / NUM_REGIONS

        # CDR 을 하나로 묶은 binary 지표 (binary 학습과 비교용)
        cdr = CDR_REGION_IDS
        tp = int(cm[np.ix_(cdr, cdr)].sum())
        fn = int(cm[cdr, :].sum()) - tp
        fp = int(cm[:, cdr].sum()) - tp
        prec, rec, f1 = _prf(tp, fp, fn)

        chain_acc = float(chain_ok[0]) / max(int(chain_ok[1]), 1)
        self.last_chain_acc = chain_acc
        elapsed = time.time() - t0
        print(
            f"[{name}] loss {avg_loss:.4f} (region {avg_region:.4f} + "
            f"{self.chain_loss_weight:g}*VH/VL {avg_chain:.4f}) | "
            f"region acc: {acc:.4f} | macro-F1: {macro_f1:.4f} | "
            f"CDR-F1(binary): {f1:.4f} | chain acc: {chain_acc:.4f} | {elapsed:.1f}s"
        )

        if verbose:
            print(f"{'region':>8}{'prec':>10}{'rec':>10}{'f1':>10}{'support':>12}")
            for i, nm in enumerate(REGION_NAMES):
                p_, r_, f_, sup = per[i]
                print(f"{nm:>8}{p_:>10.4f}{r_:>10.4f}{f_:>10.4f}{sup:>12,}")
            print("\n혼동행렬 (행=정답, 열=예측)")
            print(f"{'':>8}" + "".join(f"{x:>10}" for x in REGION_NAMES))
            for i, nm in enumerate(REGION_NAMES):
                print(f"{nm:>8}" + "".join(f"{int(v):>10,}" for v in cm[i]))

        self.model.train()
        return avg_loss, acc, prec, rec, f1

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def _state(self, epoch):
        return {
            "model_state":     self._unwrap().state_dict(),
            "optimizer_state": self.optim.state_dict(),
            "epoch":           epoch,
            "best_val_f1":     self.best_val_f1,
            "best_epoch":      self.best_epoch,
        }

    def save_checkpoint(self, run_dir: str, epoch: int):
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, f"checkpoint_epoch{epoch}.pt")
        torch.save(self._state(epoch), path)
        print(f"Checkpoint: {path}")

    def save_best(self, run_dir: str, epoch: int, f1: float):
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, "best_model.pt")
        state = self._state(epoch)
        state["best_val_f1"] = f1
        torch.save(state, path)
        print(f"  >> Best model saved (epoch={epoch}, CDR-F1={f1:.4f}): {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self._unwrap().load_state_dict(ckpt["model_state"])
        self.optim.load_state_dict(ckpt["optimizer_state"])
        self.start_epoch  = int(ckpt.get("epoch", 0))
        self.best_val_f1  = float(ckpt.get("best_val_f1", -1.0))
        self.best_epoch   = int(ckpt.get("best_epoch", -1))
        print(f"Resumed: {path}  next epoch={self.start_epoch + 1}")

    def load_latest_checkpoint(self, run_dir: str) -> bool:
        """run_dir 의 checkpoint_epoch*.pt 중 가장 최근 것으로 재개. 없으면 False."""
        if not os.path.isdir(run_dir):
            return False
        ckpts = [f for f in os.listdir(run_dir)
                 if f.startswith("checkpoint_epoch") and f.endswith(".pt")]
        if not ckpts:
            return False
        latest = max(ckpts, key=lambda f: int(f[len("checkpoint_epoch"):-len(".pt")]))
        self.load_checkpoint(os.path.join(run_dir, latest))
        return True

    def load_best_checkpoint(self, run_dir: str):
        path = os.path.join(run_dir, "best_model.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"best_model.pt not found: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self._unwrap().load_state_dict(ckpt["model_state"])
        print(f"Loaded best model: epoch={ckpt.get('epoch')}, "
              f"CDR-F1={ckpt.get('best_val_f1', 'N/A')}")

    # ── CSV logging ───────────────────────────────────────────────────────────

    def log_to_csv(self, run_dir, epoch, split, loss, acc=None, prec=None, rec=None,
                   f1=None, cdr_loss=None, chain_loss=None, chain_acc=None):
        os.makedirs(run_dir, exist_ok=True)
        path   = os.path.join(run_dir, "training_log.csv")
        is_new = not os.path.isfile(path)
        row = {
            "epoch": epoch, "split": split,
            "loss":       f"{loss:.6f}"       if loss       is not None else "",
            "cdr_loss":   f"{cdr_loss:.6f}"   if cdr_loss   is not None else "",
            "chain_loss": f"{chain_loss:.6f}" if chain_loss is not None else "",
            "chain_acc":  f"{chain_acc:.6f}"  if chain_acc  is not None else "",
            "acc":   f"{acc:.6f}"  if acc  is not None else "",
            "prec":  f"{prec:.6f}" if prec is not None else "",
            "rec":   f"{rec:.6f}"  if rec  is not None else "",
            "f1":    f"{f1:.6f}"   if f1   is not None else "",
        }
        with open(path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if is_new:
                w.writeheader()
            w.writerow(row)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def fit(
        self,
        run_dir: str,
        epochs: int          = None,
        validate_every: int  = 1,
        save_every: int      = 1,
        patience: int        = 0,   # early stopping; 0 = disabled
    ):
        epochs = epochs or self.num_epochs
        os.makedirs(run_dir, exist_ok=True)
        no_improve = 0

        for epoch in range(self.start_epoch, epochs):
            current = epoch + 1
            print(f"\n{'='*60}\nEpoch {current}/{epochs}")

            t0 = time.time()
            train_loss, tr_region, tr_chain = self.train_epoch()
            print(f"[train] loss {train_loss:.4f} (region {tr_region:.4f} + "
                  f"{self.chain_loss_weight:g}*VH/VL {tr_chain:.4f}) | "
                  f"{time.time() - t0:.1f}s")
            self.log_to_csv(run_dir, current, "train", train_loss,
                            cdr_loss=tr_region, chain_loss=tr_chain)

            if self.val_loader and current % validate_every == 0:
                val_loss, acc, prec, rec, f1 = self.validate(name="val")
                L = getattr(self, "last_losses", {})
                self.log_to_csv(run_dir, current, "val", val_loss, acc, prec, rec, f1,
                                cdr_loss=L.get("cdr"), chain_loss=L.get("chain"),
                                chain_acc=getattr(self, "last_chain_acc", None))

                if f1 is not None and f1 > self.best_val_f1:
                    self.best_val_f1 = f1
                    self.best_epoch  = current
                    self.save_best(run_dir, current, f1)
                    no_improve = 0
                else:
                    no_improve += 1
                    if patience > 0 and no_improve >= patience:
                        print(f"\nEarly stopping (patience={patience})")
                        break

            if current % save_every == 0:
                self.save_checkpoint(run_dir, current)

        print(f"\nDone. Best epoch={self.best_epoch}, CDR-F1={self.best_val_f1:.4f}")
