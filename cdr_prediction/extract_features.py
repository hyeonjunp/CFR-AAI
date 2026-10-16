"""
학습된 ABlang CDR 모델로 DeepInterAware용 항체 feature를 뽑는다.

DeepInterAware는 항체 feature를 `{cfg.set.ab_model}_token_ft.pt` 파일명으로 읽는다
(`DeepInterAware/dataloader.py`의 PairDataset). 그래서 여기서 `<name>_token_ft.pt` 를
같은 폴더에 저장하고 yml 의 `ab_model` 을 `<name>` 으로 바꾸면 DeepInterAware 코드를
고치지 않고 임베딩만 교체할 수 있다.

무엇이 바뀌나
--------------
원본 (DeepInterAware/utools/feature_encoder.py):

    heavy_ablang = ablang.pretrained("heavy")        # 사전학습 ABlang 그대로
    h_token_ft = heavy_ablang(seq, mode='rescoding') # → .cpu().detach().numpy()

여기:

    frozen ABlang → 학습된 AbLangStyleLayer 출력 (model.embed)

즉 인코더 본체(ABlang 12층)는 동일하고, 그 위에 OAS 7-class CDR region 태스크로 학습한
레이어 1개를 통과시킨 표현을 쓴다. `--raw` 를 주면 그 레이어를 건너뛰어 원본 ABlang
rescoding 과 같은 값을 뽑으므로, 같은 파이프라인에서 "학습된 레이어의 효과"만 비교할 수 있다.

DeepInterAware 와 맞춘 규약 (utools/feature_encoder.py의 process_ablang_token_ft)
--------------------------------------------------------------------------------
* 서열에서 `[XUZOBJ?*]` 제거 (ABlang vocab 에 없는 문자이기도 하다)
* heavy 임베딩 뒤에 light 임베딩을 이어 붙인다 (chain 하나가 비면 있는 쪽만)
* 각 체인은 157 잔기까지만 (ABlang max_position_embeddings=160, '<'+AA+'>' 기준)
* max_ab_len 까지 padding_tensor 로 채운다
  (`torch.manual_seed(42); (torch.rand(768) * 2 - 1) * 1e-2` — 원본과 같은 값)
* 서열 단위 feature(`<name>_ft.pt`)는 원본 `get_ablang_Seq_feature` 와 같이
  체인별 잔기 평균을 [heavy, light] 순으로 concat 한 (N, 1536). 체인이 없으면 0 벡터.

Usage
-----
    # SAbDab (CDR 문자열 기준 — DeepInterAware 의 SAbDab dataloader 가 ab_cdr 길이로 마스킹한다)
    python extract_features.py \
        --checkpoint runs/ablang_cdr_7class_both/fold_0/best_model.pt \
        --antibody-csv ../DeepInterAware/data/SAbDab/antibody.csv \
        --heavy-col H_cdr --light-col L_cdr \
        --max-ab-len 110 \
        --out-dir ../DeepInterAware/data/SAbDab \
        --name cdrablang

    # 그 다음 DeepInterAware/configs/SAbDab.yml 에서
    #   set.ab_model: cdrablang
"""

import argparse
import os
import re
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import ABlangCDRModel

# ABlang 위치 임베딩 한도(160)에서 '<', '>' 를 뺀 값. DeepInterAware 도 157 로 자른다.
MAX_CHAIN_LEN = 157
# ABlang vocab 에 없는 문자 (원본 feature_encoder.py 와 동일한 정규식)
INVALID_AA = re.compile(r"[XUZOBJ?*]")


def clean(seq) -> str:
    if seq is None or (isinstance(seq, float) and pd.isna(seq)):
        return ""
    return INVALID_AA.sub("", str(seq).strip().upper())


def padding_vector(dim: int) -> torch.Tensor:
    """DeepInterAware 가 쓰는 것과 같은 패딩 벡터."""
    torch.manual_seed(42)
    return (torch.rand(dim) * 2 - 1) * 1e-2


@torch.no_grad()
def embed_chain(model, seqs, chain: str, device, batch_size: int, raw: bool):
    """같은 chain 의 서열 리스트 → [(L_i, H)] 임베딩 리스트 (특수토큰 제외, 원래 순서 유지).

    ABlang 은 위치 임베딩이 160 까지라 157 잔기를 넘는 서열은 통째로 넣을 수 없다.
    DeepInterAware 의 `process_ablang_token_ft` 와 같이 **157 잔기씩 잘라 블록별로 임베딩한 뒤
    이어붙인다** — 그래야 전체 서열 길이만큼의 임베딩이 나오고 마스킹 길이와 맞는다
    (HIV 처럼 전체 서열을 쓰는 데이터셋에서 필요. SAbDab 의 CDR 문자열은 157 미만이라 블록 1개).

    빈 문자열은 None 으로 돌려준다 (체인이 없는 항체).
    """
    tokenizer = model.tokenizers[chain]
    pad_id = tokenizer.pad_token

    # (서열 인덱스, 블록 문자열) 목록으로 펼친다
    flat = []
    for i, s in enumerate(seqs):
        for b in range(0, len(s), MAX_CHAIN_LEN):
            flat.append((i, s[b:b + MAX_CHAIN_LEN]))

    parts = {}
    for start in range(0, len(flat), batch_size):
        batch = flat[start:start + batch_size]
        ids = [tokenizer.encode(s) for _, s in batch]          # '<' + AA + '>'
        T = max(x.shape[0] for x in ids)

        input_ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
        for r, x in enumerate(ids):
            input_ids[r, :x.shape[0]] = x
        input_ids = input_ids.to(device)
        attention_mask = (input_ids != pad_id).long()

        hidden = model.embed(
            input_ids=input_ids,
            attention_mask=attention_mask,
            chain_types=[chain] * len(batch),
            after_stacked_layer=not raw,
        ).float().cpu()                                        # (B, T, H)

        for r, (idx, s) in enumerate(batch):
            parts.setdefault(idx, []).append(hidden[r, 1:1 + len(s)])   # '<' 다음부터

    return [torch.cat(parts[i], dim=0) if i in parts else None
            for i in range(len(seqs))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="model_ablang 학습 결과 (best_model.pt)")
    p.add_argument("--antibody-csv", required=True,
                   help="DeepInterAware 의 antibody.csv")
    p.add_argument("--heavy-col", default="H_cdr")
    p.add_argument("--light-col", default="L_cdr")
    p.add_argument("--max-ab-len", type=int, required=True,
                   help="yml 의 protein.max_antibody_len (SAbDab=110)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--name", default="cdrablang",
                   help="저장 파일 접두사. yml 의 set.ab_model 에 이 값을 넣는다")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--raw", action="store_true",
                   help="학습된 레이어를 건너뛰고 frozen ABlang 출력만 사용 (대조군)")
    args = p.parse_args()

    df = pd.read_csv(args.antibody_csv, keep_default_na=False, na_values=[])
    for col in (args.heavy_col, args.light_col):
        if col not in df.columns:
            raise ValueError(f"antibody.csv 에 '{col}' 컬럼이 없습니다. 있는 컬럼: {list(df.columns)}")
    heavy = [clean(s) for s in df[args.heavy_col]]
    light = [clean(s) for s in df[args.light_col]]
    print(f"[data] {args.antibody_csv} → {len(df):,} rows "
          f"(heavy 없음 {sum(1 for s in heavy if not s)}, light 없음 {sum(1 for s in light if not s)})")

    model = ABlangCDRModel(chain="both", num_labels=7, use_chain_head=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(args.device)
    print(f"[model] {args.checkpoint} (epoch={ckpt.get('epoch')}, "
          f"CDR-F1={ckpt.get('best_val_f1')}) | 임베딩: "
          f"{'frozen ABlang 출력 (--raw)' if args.raw else 'frozen ABlang → 학습된 레이어 출력'}")

    h_emb = embed_chain(model, heavy, "heavy", args.device, args.batch_size, args.raw)
    l_emb = embed_chain(model, light, "light", args.device, args.batch_size, args.raw)

    dim = model.hidden_size
    pad_vec = padding_vector(dim)

    token_ft, seq_ft, truncated = [], [], 0
    for h, l in zip(h_emb, l_emb):
        parts = [x for x in (h, l) if x is not None]
        ab = torch.cat(parts, dim=0) if parts else torch.zeros(0, dim)

        if ab.shape[0] > args.max_ab_len:
            truncated += 1
            ab = ab[:args.max_ab_len]
        elif ab.shape[0] < args.max_ab_len:
            pad = pad_vec.unsqueeze(0).expand(args.max_ab_len - ab.shape[0], dim)
            ab = torch.cat([ab, pad], dim=0)
        token_ft.append(ab)

        # 서열 단위: 체인별 잔기 평균을 [heavy, light] 로 concat (원본 get_ablang_Seq_feature 규약)
        seq_ft.append(torch.cat([
            h.mean(0) if h is not None else torch.zeros(dim),
            l.mean(0) if l is not None else torch.zeros(dim),
        ], dim=-1))

    token_ft = torch.stack(token_ft)      # (N, max_ab_len, H)
    seq_ft   = torch.stack(seq_ft)        # (N, 2H)

    os.makedirs(args.out_dir, exist_ok=True)
    token_path = os.path.join(args.out_dir, f"{args.name}_token_ft.pt")
    seq_path   = os.path.join(args.out_dir, f"{args.name}_ft.pt")
    torch.save(token_ft, token_path)
    torch.save(seq_ft, seq_path)

    if truncated:
        print(f"[warn] max_ab_len({args.max_ab_len}) 초과로 잘린 항체 {truncated:,}개")
    print(f"[save] {token_path}  {tuple(token_ft.shape)}")
    print(f"[save] {seq_path}  {tuple(seq_ft.shape)}")
    print(f"\nDeepInterAware yml 에서 set.ab_model 을 '{args.name}' 으로 바꾸면 적용됩니다.")


if __name__ == "__main__":
    main()
