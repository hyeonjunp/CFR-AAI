"""
Frozen 인코더 임베딩을 npz로 미리 만들어 두는 스크립트 (학습과 분리).

체인마다 인코더가 다르다:

    VH  →  AbLang **heavy** (AbRep, frozen)            768-dim, max 160 토큰
    VL  →  AbLang **light** (AbRep, frozen)            768-dim, max 160 토큰
    AG  →  ESM2 (facebook/esm2_t30_150M_UR50D)         640-dim

항원은 항체 언어모델로 인코딩할 수 없어서 ESM2가 그대로 남는다. 그래서 `ag_emb.npz`는
CDR_ESM150M이 만든 것과 완전히 같다 — 이미 있으면 symlink 하라고 안내한다.

    # 1) 한 번만 (GPU 필요)
    python build_embeddings.py --config configs/HIV.yaml --dataset HIV
    # 2) 학습 (인코더 forward 없음)
    python train.py --config configs/HIV.yaml --dataset HIV

저장 위치 / 파일 형식:

    ../emb/<dataset>/vh_emb.npz      VH (항체 heavy / VHH)
    ../emb/<dataset>/vl_emb.npz      VL
    ../emb/<dataset>/ag_emb.npz      AG (항원)

        "__max_len__"                -> 이 파일을 만들 때 쓴 max_len
        "<sha1(seq|max_len)>_hidden" -> (max_len, D) float16
        "<sha1(seq|max_len)>_mask"   -> (max_len,)  bool
        "<sha1(seq|max_len)>_seq"    -> 서열 문자열

한 데이터셋의 모든 fold/seed(+HIV의 모든 split_type)를 한꺼번에 처리한다. train/val/test가
같은 서열을 많이 공유하므로 unique 서열 기준으로 한 번씩만 인코딩된다.

AbLang 백본은 CDR 체크포인트 안에서도 완전히 frozen(freeze_ablang=true, unfreeze_last_n=0)
이므로 사전학습 가중치와 동일하다. 그래도 어긋남이 없도록 --cdr_ckpt(기본: config의
pretrained.cdr_ckpt) 안의 backbones.<chain>.* 를 그대로 싣는다. ESM2도 마찬가지 이유로
CDR_ESM150M과 같은 사전학습 가중치를 쓴다 (AbLang 체크포인트에는 esm.* 가 없다).

여기서 저장되는 VH/VL 표현은 **AbLang backbone 뒤에 CDR checkpoint의 frozen
AbLangStyleLayer(attn_block)를 적용한 출력**이다. CDR logits는 저장하지 않는다.
"""

import argparse
import os
import sys

import pandas as pd
import numpy as np
import torch
import yaml
from transformers import EsmModel, EsmTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

from embedding_cache import (
    ABLANG_MAX_AA, ABLANG_MAX_LEN, AbLangEncoder, EmbeddingCache, EsmEncoder,
    chain_max_lens, resolve_cache_dir,
)
from prepare_data import (
    load_sabdab_fold, load_hiv_seed, load_covabdab_seed, load_avida_seed,
    load_avida_sars_cov2_seed, dataset_has_vl, N_SPLITS, DATA_ROOT,
)


def _load_ablang_style_layer():
    """cdr_prediction/model.py 의 AbLangStyleLayer. 이 폴더의 model.py 와 이름이 같아 경로로 직접 로드한다."""
    import importlib.util
    path = os.path.join(_REPO_ROOT, "cdr_prediction", "model.py")
    spec = importlib.util.spec_from_file_location("cdr_prediction_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.AbLangStyleLayer


# 데이터셋 키 = data/ 아래 실제 폴더명 (prepare_data.DATASET_LOADERS와 동일)
DATASETS   = ["SAbDab", "HIV", "CoVAbDab", "il6", "AVIDa_SARS_CoV_2"]
ALIASES    = {"avida_hil6": "il6", "avida": "il6", "hil6": "il6",
              "sabdab": "SAbDab", "hiv": "HIV", "covabdab": "CoVAbDab", "il6": "il6",
              "avida_sars_cov_2": "AVIDa_SARS_CoV_2", "avida-sars-cov-2": "AVIDa_SARS_CoV_2",
              "avidasarscov2": "AVIDa_SARS_CoV_2", "sars_cov_2": "AVIDa_SARS_CoV_2"}
HIV_SPLITS = ["unseen", "ag_unseen", "ab_unseen"]

# 항체 체인 role → AbLang 체크포인트 chain
ABLANG_CHAIN = {"vh": "heavy", "vl": "light"}


def normalize_dataset(name: str) -> str:
    """--dataset il6 처럼 짧게 써도 받도록 정규화한다."""
    if name in DATASETS:
        return name
    key = ALIASES.get(str(name).lower())
    if key is None:
        raise SystemExit(f"알 수 없는 dataset: {name!r} (가능: {', '.join(DATASETS)})")
    return key


def split_dataset_arg(name: str):
    """--dataset 값을 (dataset, split_type)으로 나눈다.

    "HIV/ab_unseen" 처럼 슬래시로 붙여 써도 받는다 (= --dataset HIV --split_type ab_unseen).
    split_type이 없으면 두 번째 값은 None.
    """
    ds, _, st = str(name).partition("/")
    dataset   = normalize_dataset(ds.strip())
    st        = st.strip().strip("/")
    if not st:
        return dataset, None
    if dataset != "HIV":
        raise SystemExit(f"split_type은 HIV 전용입니다: {name!r} (dataset={dataset})")
    if st not in HIV_SPLITS:
        raise SystemExit(f"알 수 없는 HIV split_type: {st!r} (가능: {', '.join(HIV_SPLITS)})")
    return dataset, st


def collect_dfs(dataset: str, data_root: str, indices=None, split_types=None) -> list:
    """해당 데이터셋의 모든 split을 모아서 dataframe 리스트로 돌려준다."""
    indices = indices if indices is not None else list(range(N_SPLITS))
    dfs = []
    for idx in indices:
        if dataset == "SAbDab":
            dfs += list(load_sabdab_fold(idx, data_root=data_root))
        elif dataset == "HIV":
            for st in (split_types or HIV_SPLITS):
                dfs += list(load_hiv_seed(idx, split_type=st, data_root=data_root))
        elif dataset == "CoVAbDab":
            dfs += list(load_covabdab_seed(idx, data_root=data_root))
        elif dataset == "AVIDa_SARS_CoV_2":
            dfs += list(load_avida_sars_cov2_seed(idx, data_root=data_root))
        else:
            dfs += list(load_avida_seed(idx, data_root=data_root))
    return [d for d in dfs if d is not None]


def build_ablang_encoder(chain: str, ablang_model_dir: str, cdr_ckpt_path: str,
                         device, cdr_layer):
    """AbLang(chain 전용)을 올리고, CDR 체크포인트의 backbones.<chain>.* 로 덮어쓴다."""
    import ablang

    pt = ablang.pretrained(chain=chain, model_folder=ablang_model_dir, device="cpu")
    abrep = pt.AbRep

    if cdr_ckpt_path and os.path.isfile(cdr_ckpt_path):
        ckpt   = torch.load(cdr_ckpt_path, map_location="cpu")
        state  = ckpt.get("model_state", ckpt)
        prefix = f"backbones.{chain}."
        bb_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if bb_state:
            missing, unexpected = abrep.load_state_dict(bb_state, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"체크포인트의 {prefix}* 가 AbLang 구조와 맞지 않습니다: {cdr_ckpt_path}\n"
                    f"  missing={list(missing)[:8]}  unexpected={list(unexpected)[:8]}")
            print(f"[build_embeddings] {prefix}* loaded from {cdr_ckpt_path}")
        else:
            print(f"[build_embeddings] 체크포인트에 {prefix}* 없음 — "
                  f"AbLang 사전학습 가중치 그대로 사용")
    else:
        print(f"[build_embeddings] cdr_ckpt 없음 — AbLang({chain}) 사전학습 가중치 그대로 사용")

    abrep = abrep.to(device).eval()
    for p in abrep.parameters():
        p.requires_grad = False
    return AbLangEncoder(abrep, pt.tokenizer, device, chain, cdr_layer=cdr_layer)


def build_cdr_layer(cdr_ckpt_path: str, hidden_size: int, num_heads: int, dropout: float):
    if not cdr_ckpt_path or not os.path.isfile(cdr_ckpt_path):
        raise FileNotFoundError(
            "post-CDR embedding을 만들려면 유효한 CDR checkpoint가 필요합니다: "
            f"{cdr_ckpt_path!r}")
    ckpt = torch.load(cdr_ckpt_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    attn_state = {k[len("attn_block."):]: v for k, v in state.items()
                  if k.startswith("attn_block.")}
    if not attn_state:
        raise RuntimeError(f"checkpoint에 attn_block.*가 없습니다: {cdr_ckpt_path}")
    actual_hidden = attn_state["attn_layer_norm.weight"].shape[0]
    intermediate = attn_state["expand_dense.weight"].shape[0]
    if actual_hidden != hidden_size:
        raise RuntimeError(f"CDR attn_block hidden={actual_hidden}, AbLang hidden={hidden_size}")
    AbLangStyleLayer = _load_ablang_style_layer()
    layer = AbLangStyleLayer(hidden_size, num_heads, intermediate, dropout=dropout)
    layer.load_state_dict(attn_state, strict=True)
    layer.eval()
    for p in layer.parameters():
        p.requires_grad = False
    print(f"[build_embeddings] CDR attn_block loaded: {cdr_ckpt_path}")
    return layer


def reset_stale_ablang_cache(cache_dir: str):
    for role in ("vh", "vl"):
        path = os.path.join(cache_dir, f"{role}_emb.npz")
        if not os.path.isfile(path):
            continue
        with np.load(path, allow_pickle=False) as npz:
            representation = str(npz["__representation__"]) if "__representation__" in npz else "raw"
        if representation != "ablang_post_cdr_attn_block":
            os.remove(path)
            print(f"[build_embeddings] stale {role.upper()} cache removed: {path}")


def build_esm_encoder(esm_model_name: str, device):
    """항원용 ESM2 (사전학습 가중치 그대로 — AbLang 체크포인트에는 esm.* 가 없다)."""
    esm = EsmModel.from_pretrained(esm_model_name, add_pooling_layer=False)
    esm = esm.to(device).eval()
    for p in esm.parameters():
        p.requires_grad = False
    print(f"[build_embeddings] AG 인코더: {esm_model_name} (hidden={esm.config.hidden_size})")
    return EsmEncoder(esm, EsmTokenizer.from_pretrained(esm_model_name), device)


def chain_specs(dataset: str, vh_max_len: int, vl_max_len: int, ag_max_len: int):
    """이 데이터셋에서 실제로 만들 (컬럼, max_len, role) 목록.

    나노바디(VHH) 데이터셋은 light chain이 없으므로 VL은 아예 빠진다. VHH는 heavy 계열이라
    VH role 그대로 AbLang heavy 백본을 쓴다."""
    specs = [("vh_seq", vh_max_len, "vh")]
    if dataset_has_vl(dataset):
        specs.append(("vl_seq", vl_max_len, "vl"))
    specs.append(("ag_seq", ag_max_len, "ag"))
    return specs


def hint_shared_ag(cache_dir: str, dataset: str, ag_max_len: int):
    """항원 npz는 CDR_ESM150M 과 같은 파일이다 — 이미 있으면 symlink를 안내한다."""
    sibling = os.path.normpath(os.path.join(
        _HERE, "..", "CDR_ESM150M", "emb", dataset, "ag_emb.npz"))
    target  = os.path.join(cache_dir, "ag_emb.npz")
    if os.path.isfile(target) or not os.path.isfile(sibling):
        return
    size = os.path.getsize(sibling) / 1024**3
    print(f"\n[build_embeddings] 참고: 같은 ESM2/max_len 으로 만든 항원 임베딩이 이미 있습니다 "
          f"({size:.2f} GB)")
    print(f"  이 스크립트는 그대로 다시 계산합니다. 디스크/시간을 아끼려면 중단하고 symlink 하세요:")
    print(f"    ln -s {sibling} {target}")
    print(f"  (max_len 이 {ag_max_len} 로 같을 때만 유효 — 다르면 캐시가 에러를 냅니다)\n")


def main():
    ap = argparse.ArgumentParser(description="AbLang/ESM2 임베딩 npz 생성 (학습 전 1회)")
    ap.add_argument("--config",     default=os.path.join(_HERE, "configs", "HIV.yaml"))
    ap.add_argument("--dataset",    default=None,
                    help=f"{' | '.join(DATASETS)} (HIV/ab_unseen 처럼 split_type을 붙여 써도 됨)")
    ap.add_argument("--split_type", default=None, help="HIV 전용. 생략하면 3종 모두")
    ap.add_argument("--seed",       type=int, default=None, help="특정 seed만. 생략하면 0..N-1 전부")
    ap.add_argument("--fold",       type=int, default=None, help="SAbDab 전용. 생략하면 전체 fold")
    ap.add_argument("--cdr_ckpt",   default=None, help="기본값: config의 pretrained.cdr_ckpt")
    ap.add_argument("--batch_size", type=int, default=None, help="기본값: config의 embedding_cache_batch_size")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dataset, split_from_arg = split_dataset_arg(args.dataset or cfg["data"].get("dataset", "HIV"))
    data_root = cfg["data"].get("data_root", DATA_ROOT)
    if not os.path.isabs(data_root):
        data_root = os.path.normpath(os.path.join(_HERE, data_root))
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"data.data_root does not exist: {data_root}")
    cache_dir = resolve_cache_dir(cfg, dataset, _HERE, create=True)
    vh_max_len, vl_max_len, ag_max_len = chain_max_lens(cfg, dataset)

    idx = args.fold if dataset == "SAbDab" else args.seed
    indices     = [idx] if idx is not None else None
    split_types = [args.split_type or split_from_arg] if (args.split_type or split_from_arg) else None

    print(f"[build_embeddings] dataset={dataset}"
          + (f"  split_type={split_types[0]}" if split_types else "")
          + f"  out={cache_dir}")
    if dataset == "HIV" and split_types:
        print("[build_embeddings] 참고: HIV의 npz는 split_type 3종이 공유합니다 — "
              "--split_type 없이 한 번 돌려두면 나머지도 재사용됩니다.")
    print(f"[build_embeddings] max_len  vh={vh_max_len} vl={vl_max_len} ag={ag_max_len} "
          f"(AbLang 한도: 폭 {ABLANG_MAX_LEN} 토큰 / 잔기 {ABLANG_MAX_AA}개)")

    dfs = collect_dfs(dataset, data_root, indices=indices, split_types=split_types)
    all_df = pd.concat(dfs, ignore_index=True)
    print(f"[build_embeddings] rows={len(all_df)}  "
          f"unique vh={all_df['vh_seq'].nunique()} vl={all_df['vl_seq'].nunique()} "
          f"ag={all_df['ag_seq'].nunique()}")

    hint_shared_ag(cache_dir, dataset, ag_max_len)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cdr_ckpt  = args.cdr_ckpt if args.cdr_ckpt is not None else cfg.get("pretrained", {}).get("cdr_ckpt")
    specs     = chain_specs(dataset, vh_max_len, vl_max_len, ag_max_len)
    roles     = {role for _, _, role in specs}

    # 인코더는 실제로 쓸 role 만 만든다 (VHH 데이터셋에서 light 백본을 올릴 이유가 없다).
    reset_stale_ablang_cache(cache_dir)
    cdr_layer = build_cdr_layer(
        cdr_ckpt, 768, cfg["model"].get("ablang_attn_heads", 12),
        cfg["model"].get("dropout", 0.1),
    ).to(device)
    encoders = {}
    for role in ("vh", "vl"):
        if role in roles:
            encoders[role] = build_ablang_encoder(
                ABLANG_CHAIN[role],
                cfg["model"].get("ablang_model_dir", "download"),
                cdr_ckpt, device, cdr_layer,
            )
    if "ag" in roles:
        encoders["ag"] = build_esm_encoder(
            cfg["model"].get("esm_model_name", "facebook/esm2_t30_150M_UR50D"), device)

    cache = EmbeddingCache(
        encoders   = encoders,
        cache_dir  = cache_dir,
        batch_size = args.batch_size or cfg["train"].get("embedding_cache_batch_size", 32),
    )

    for col, max_len, role in specs:
        cache.prewarm(all_df[col].tolist(), max_len, role)
        path = cache.shard_path(role)
        size = os.path.getsize(path) / 1024**3 if os.path.isfile(path) else 0.0
        print(f"[build_embeddings] {role.upper()}: max_len={max_len}  "
              f"{cache.load(role, max_len)} seqs → {path}  ({size:.2f} GB)")
        enc = encoders[role]
        if isinstance(enc, AbLangEncoder) and (enc.n_truncated or enc.n_substituted):
            print(f"[build_embeddings]   ⚠ {role.upper()}: "
                  f"{min(max_len - 2, ABLANG_MAX_AA)}잔기 초과로 잘린 서열 {enc.n_truncated}개, "
                  f"AbLang vocab 밖 문자(X 등)를 '*'로 바꾼 서열 {enc.n_substituted}개")

    print("\n[build_embeddings] 완료. 이제 학습은 인코더 없이 이 npz만 읽습니다:")
    print(f"  python train.py --config {args.config} --dataset {dataset}")


if __name__ == "__main__":
    main()
