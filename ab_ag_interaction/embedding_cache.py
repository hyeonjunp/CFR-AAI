"""
Frozen 인코더 임베딩 캐시 — 체인(role)마다 다른 인코더를 쓴다.

`CDR_ESM150M/embedding_cache.py`와 npz 포맷은 **완전히 같다**. 달라진 것은 하나뿐:
role마다 인코더가 다르다.

    role   저장 표현                                  hidden dim
    ────   ──────────────────────────────────────   ──────────
    vh     AbLang heavy (AbRep, frozen)                  768
    vl     AbLang light (AbRep, frozen)                  768
    ag     ESM2 (facebook/esm2_t30_150M_UR50D)           640

항원은 항체 언어모델로 인코딩할 수 없으므로(AbLang의 vocab/위치임베딩은 항체 V 도메인
전용) ESM2를 그대로 쓴다. 그래서 `ag_emb.npz`는 CDR_ESM150M이 만든 것과 **바이트 단위로
같다** — 이미 있으면 새로 만들지 말고 symlink 하면 된다 (build_embeddings.py가 안내한다).

인코더는 전부 frozen + eval 이므로 (서열, max_len)에 대해 출력이 결정적이다. 따라서 한 번
계산해서 npz에 넣어두고 학습 때는 읽기만 한다.

Disk layout (CDR_ESM150M과 동일):

    cache_dir/vh_emb.npz      VH (항체 heavy / VHH)
    cache_dir/vl_emb.npz      VL
    cache_dir/ag_emb.npz      AG (항원)

        "__max_len__"                -> 이 파일을 만들 때 쓴 max_len (int32)
        "<sha1(seq|max_len)>_hidden" -> (max_len, D) float16
        "<sha1(seq|max_len)>_mask"   -> (max_len,)  bool
        "<sha1(seq|max_len)>_seq"    -> 0-d str array (원본 서열 — 해시가 아니라 서열로
                                         다시 키를 잡을 수 있게)

hidden state는 패딩 길이에 따라 달라지므로 파일에 max_len을 적어두고, config의 max_len과
다르면 조용히 섞이는 대신 에러를 낸다.

주의: vh/vl npz에는 AbLang 백본 출력에 CDR 체크포인트의 frozen AbLangStyleLayer
(attn_block)를 적용한 **post-CDR residue embedding**이 저장된다. ag npz에는 기존 ESM2
출력이 저장된다. raw AbLang 캐시는 표현 단계 표식이 없어 더 이상 읽지 않는다.
"""

import os
import hashlib
import re

import numpy as np
import torch
from tqdm import tqdm

ROLES = ("vh", "vl", "ag")

# AbLang을 쓰는 체인. 이 role의 max_len은 AbLang 위치 임베딩 한도를 넘을 수 없다.
ABLANG_ROLES = {"vh": "heavy", "vl": "light"}

# AbLang 위치 임베딩 테이블 크기 (hparams.max_position_embeddings, heavy/light 모두 160).
# 텐서 폭(max_len)의 상한이다 — 패딩은 position id 0 을 받으므로 폭 자체는 160까지 괜찮다.
ABLANG_MAX_LEN = 160

# 실제로 넣을 수 있는 잔기 수. AbLang 의 position id 는 pad 가 아닌 토큰의 1-based cumsum
# 이라서 (ablang/embedding.py: create_position_ids_from_input_ids) 마지막 실토큰이 받는
# id 가 곧 실토큰 개수다. 테이블 인덱스는 0..159 이므로 실토큰 ≤ 159,
# 즉 '<' + AA×157 + '>' 가 한계다. 158잔기를 넣으면 position id 160 으로 인덱스를 넘겨
# CUDA device-side assert 가 난다. model_ablang/extract_features.py 의 MAX_CHAIN_LEN 과 같은 값.
ABLANG_MAX_AA = ABLANG_MAX_LEN - 3

# AbLang vocab에 없는 문자 → 마스크 토큰 '*' 로 치환한다. 지워버리면(원본
# feature_encoder.py 방식) 서열 길이가 바뀌어 토큰 정렬이 어긋나므로, 길이를 보존하는
# 치환을 쓴다. '*'는 AbLang 사전학습(25% 마스킹)에서 대량으로 본 토큰이라 안전하다.
_NON_ABLANG_AA = re.compile(r"[^MRHKDESTNQCGPAVIFYWL]")

# 디스크 저장 dtype. 메모리에는 bfloat16으로 들고 있으므로 float32로 쓰면 용량만 2배가 된다
# (bf16은 가수 8비트라 float16(가수 11비트)으로 담아도 가수 비트는 하나도 안 잃는다).
# 다만 float16은 지수 범위가 좁아 |x| > 65504 는 inf가 되므로, 저장 직전에 실제로 그런 값이
# 있는지 확인하고 있으면 실패시킨다 (조용히 inf가 섞여 들어가는 것보다 낫다).
STORAGE_DTYPE = np.float16
_FP16_MAX     = 65504.0
REPRESENTATION = {
    "vh": "ablang_post_cdr_attn_block",
    "vl": "ablang_post_cdr_attn_block",
    "ag": "esm2",
}


def _hash_key(seq: str, max_len: int) -> str:
    return hashlib.sha1(f"{seq}|{max_len}".encode()).hexdigest()


# ============================================================================
# 인코더 — role 하나를 (서열 리스트, max_len) → (hidden, mask) 로 바꾼다
# ============================================================================

class EsmEncoder:
    """ESM2 (항원용). HuggingFace EsmTokenizer + EsmModel."""

    dim_name = "ESM2"

    def __init__(self, esm_model, tokenizer, device):
        self.model     = esm_model
        self.tokenizer = tokenizer
        self.device    = device

    @torch.no_grad()
    def __call__(self, seqs, max_len: int):
        enc = self.tokenizer(
            [str(s).strip().upper() for s in seqs],
            add_special_tokens=True, max_length=max_len,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        ids  = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        out  = self.model(input_ids=ids, attention_mask=mask).last_hidden_state
        return out, mask.bool()


class AbLangEncoder:
    """AbLang (항체 체인용). chain 전용 AbRep + ABtokenizer.

    토큰 배치는 ESM과 같은 모양이다 — pos 0 = 시작('<'), pos 1..L = AA, pos L+1 = 끝('>'),
    그 뒤는 pad('-'). 그래서 model.py의 `special_token_mask`/`exclude_mask`가 ESM과 똑같이
    동작한다 (첫 토큰과 마지막 유효 토큰을 제외).

    한도(min(max_len - 2, 157)잔기)를 넘는 서열은 **앞에서부터 잘라** 쓴다. V 도메인
    (FR1~FR4, CDR 3개)은 N-말단 ~120잔기 안에 전부 들어 있고, 그 뒤는 불변부위(CH1/CL)라
    AbLang도 CDR 모델도 학습한 적 없는 분포다. 잘린 서열 수는 prewarm이 집계해서 알려준다.
    """

    dim_name = "AbLang"

    def __init__(self, abrep, tokenizer, device, chain: str, cdr_layer=None):
        self.abrep     = abrep
        self.tokenizer = tokenizer
        self.device    = device
        self.chain     = chain
        self.cdr_layer = cdr_layer
        self.pad_id    = tokenizer.pad_token
        self.n_truncated = 0
        self.n_substituted = 0

    def _clean(self, seq: str, max_aa: int) -> str:
        s = str(seq).strip().upper()
        s, n_sub = _NON_ABLANG_AA.subn("*", s)
        if n_sub:
            self.n_substituted += 1
        if len(s) > max_aa:
            self.n_truncated += 1
            s = s[:max_aa]
        return s

    @torch.no_grad()
    def __call__(self, seqs, max_len: int):
        if max_len > ABLANG_MAX_LEN:
            raise ValueError(
                f"AbLang의 위치 임베딩은 {ABLANG_MAX_LEN} 토큰까지입니다 (max_len={max_len}). "
                f"config의 data.max_len에서 vh/vl을 {ABLANG_MAX_LEN} 이하로 낮추세요.")

        cleaned = [self._clean(s, min(max_len - 2, ABLANG_MAX_AA)) for s in seqs]
        ids     = [self.tokenizer.encode(s) for s in cleaned]   # '<' + AA... + '>'

        input_ids = torch.full((len(ids), max_len), self.pad_id, dtype=torch.long)
        for r, x in enumerate(ids):
            input_ids[r, :x.shape[0]] = x
        input_ids = input_ids.to(self.device)
        mask      = (input_ids != self.pad_id)

        # AbRep은 넘겨준 attention_mask를 무시하고 pad_token_id로 자기가 다시 만든다
        # (ablang/model.py: AbRep.forward). 패딩만 pad_token으로 해두면 된다.
        out = self.abrep(input_ids).last_hidden_states
        if self.cdr_layer is not None:
            out = self.cdr_layer(out, mask == 0)
        return out, mask


# ============================================================================
# 캐시
# ============================================================================

class EmbeddingCache:
    """role별 인코더로 임베딩을 만들고 npz로 보관한다.

    encoders : {"vh": AbLangEncoder, "vl": AbLangEncoder, "ag": EsmEncoder}
               학습 때(train.py)는 읽기만 하므로 None 이어도 된다 — 그때는 `require()`만
               쓰이고 인코더는 호출되지 않는다.
    """

    def __init__(self, encoders: dict = None, cache_dir: str = None,
                 dtype=torch.bfloat16, batch_size: int = 32):
        self.encoders   = encoders or {}
        self.cache_dir  = cache_dir
        self.dtype      = dtype
        self.batch_size = batch_size
        os.makedirs(cache_dir, exist_ok=True)
        self._mem: dict = {}              # (seq, max_len) -> (hidden cpu dtype, mask cpu bool)
        self._role_keys: dict = {r: set() for r in ROLES}   # role -> {(seq, max_len)}
        self._loaded_shards: set = set()  # 이 프로세스에서 이미 디스크에서 읽은 role

    # ── paths ────────────────────────────────────────────────────────────────

    def shard_path(self, role: str) -> str:
        return os.path.join(self.cache_dir, f"{role}_emb.npz")

    # ── disk I/O ─────────────────────────────────────────────────────────────

    def _read_npz(self, path: str, role: str, max_len: int) -> int:
        with np.load(path, allow_pickle=False) as npz:
            files = set(npz.files)
            if "__max_len__" not in files:
                raise RuntimeError(
                    f"{path}: __max_len__ 이 없습니다 (손상됐거나 다른 도구가 만든 파일). "
                    f"build_embeddings.py 로 다시 만드세요.")
            stored = int(npz["__max_len__"])
            if stored != max_len:
                raise RuntimeError(
                    f"{os.path.basename(path)} 의 max_len={stored} 인데 config는 {max_len} 입니다.\n"
                    f"  hidden state는 패딩 길이에 따라 달라지므로 그대로 쓸 수 없습니다 — "
                    f"config를 되돌리거나 build_embeddings.py 로 다시 만드세요.")
            expected = REPRESENTATION[role]
            if "__representation__" not in files:
                if role != "ag":
                    raise RuntimeError(
                        f"{path}: raw AbLang cache입니다. post-CDR embedding으로 다시 만드세요.")
            elif str(npz["__representation__"]) != expected:
                raise RuntimeError(
                    f"{path}: representation={str(npz['__representation__'])!r}, "
                    f"expected={expected!r}. build_embeddings.py 로 다시 만드세요.")
            hashes = sorted({k.rsplit("_", 1)[0] for k in files if not k.startswith("__")})
            for h in hashes:
                seq    = str(npz[f"{h}_seq"])
                hidden = torch.from_numpy(npz[f"{h}_hidden"]).to(self.dtype)
                mask   = torch.from_numpy(npz[f"{h}_mask"]).bool()
                self._mem[(seq, max_len)] = (hidden, mask)
                self._role_keys[role].add((seq, max_len))
        return len(hashes)

    def _load_shard(self, role: str, max_len: int):
        """Lazy, once per process: 이 role의 npz를 _mem으로 읽어들인다."""
        if role in self._loaded_shards:
            return
        self._loaded_shards.add(role)

        path = self.shard_path(role)
        if os.path.isfile(path):
            self._read_npz(path, role, max_len)

    def _save_shard(self, role: str, max_len: int):
        """이 role이 들고 있는 전부를 npz로 다시 쓴다."""
        arrays  = {
            "__max_len__": np.array(max_len, dtype=np.int32),
            "__representation__": np.array(REPRESENTATION[role]),
        }
        max_abs = 0.0
        for key in sorted(self._role_keys[role]):
            seq, ml = key
            hidden, mask = self._mem[key]
            h = _hash_key(seq, ml)
            max_abs = max(max_abs, float(hidden.abs().max()))
            arrays[f"{h}_hidden"] = hidden.float().numpy().astype(STORAGE_DTYPE)
            arrays[f"{h}_mask"]   = mask.numpy()
            arrays[f"{h}_seq"]    = np.array(seq)
        if STORAGE_DTYPE == np.float16 and max_abs > _FP16_MAX:
            raise RuntimeError(
                f"hidden state에 float16 범위를 넘는 값이 있습니다 (max|x|={max_abs:.1f} > {_FP16_MAX}). "
                f"embedding_cache.STORAGE_DTYPE 를 np.float32 로 바꾸고 다시 만드세요.")
        np.savez(self.shard_path(role), **arrays)

    # ── public API ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def prewarm(self, seqs, max_len: int, role: str, desc: str = None):
        """seqs: iterable of (not necessarily unique) sequence strings."""
        desc = desc or f"{role.upper()} embeddings"
        encoder = self.encoders.get(role)
        if encoder is None:
            raise RuntimeError(f"role={role!r} 의 인코더가 없습니다 (읽기 전용 캐시입니다)")
        self._load_shard(role, max_len)

        unique = sorted(set(str(s) for s in seqs))
        todo = [s for s in unique if (s, max_len) not in self._mem]
        # 이미 _mem에 있지만 이 role 파일에는 없는 것(= 다른 role이 올려둔 것)도 기록해 둔다
        for s in unique:
            self._role_keys[role].add((s, max_len))
        if not todo:
            return

        for i in tqdm(range(0, len(todo), self.batch_size), desc=desc, leave=False):
            chunk = todo[i:i + self.batch_size]
            out, mask = encoder(chunk, max_len)
            out      = out.to(self.dtype).cpu()
            mask_cpu = mask.cpu().bool()
            for j, s in enumerate(chunk):
                self._mem[(s, max_len)] = (out[j].clone(), mask_cpu[j].clone())

        self._save_shard(role, max_len)   # one rewrite for the whole batch of new sequences

    def load(self, role: str, max_len: int) -> int:
        """디스크의 npz를 메모리로 읽어들인다 (인코더 불필요). 보유 중인 서열 수를 돌려준다."""
        self._load_shard(role, max_len)
        return len(self._role_keys[role])

    def require(self, seqs, max_len: int, role: str, desc: str = None):
        """npz에 이미 있는 것만 쓰는 모드. 빠진 서열이 하나라도 있으면 즉시 실패시킨다 —
        학습 도중 KeyError로 죽는 대신, 무엇을 다시 만들어야 하는지 알려준다."""
        desc = desc or f"{role.upper()} embeddings"
        self._load_shard(role, max_len)
        missing = sorted({str(s) for s in seqs if (str(s), max_len) not in self._mem})
        if missing:
            raise RuntimeError(
                f"[{desc}] max_len={max_len}: npz에 없는 서열 {len(missing)}개.\n"
                f"  shard: {self.shard_path(role)}\n"
                f"  예시 : {missing[0][:40]!r}\n"
                f"  → build_embeddings.py 를 먼저 실행하세요.")

    def state_dict(self) -> dict:
        """Plain-tensor dict — safe to hand to Dataset objects that cross DataLoader
        worker processes (no GPU model reference to pickle)."""
        return self._mem


# ── config helpers — train.py와 build_embeddings.py가 같은 경로/길이를 쓰도록 공유 ──

def chain_max_lens(cfg: dict, dataset: str) -> tuple:
    """config의 data.max_len[dataset] (없으면 기본값) → (vh, vl, ag) max_len.

    vh/vl은 AbLang 한도(160)를 넘을 수 없으므로 여기서 바로 확인한다 — 임베딩을 다 만든
    뒤에 터지는 것보다 낫다."""
    d = cfg["data"]
    per_dataset = d.get("max_len", {}).get(dataset, {})
    vh = per_dataset.get("vh", d.get("vh_max_len", ABLANG_MAX_LEN))
    vl = per_dataset.get("vl", d.get("vl_max_len", ABLANG_MAX_LEN))
    ag = per_dataset.get("ag", d.get("ag_max_len", 512))
    for name, v in (("vh", vh), ("vl", vl)):
        if v > ABLANG_MAX_LEN:
            raise ValueError(
                f"data.max_len[{dataset}].{name}={v} 는 AbLang 한도({ABLANG_MAX_LEN})를 넘습니다. "
                f"{ABLANG_MAX_LEN} 이하로 낮추세요.")
    return vh, vl, ag


def resolve_cache_dir(cfg: dict, dataset: str, project_dir: str, create: bool = False) -> str:
    """임베딩 npz 저장 위치: <base>/<dataset>/.

    ``train.embedding_cache_dir``가 있으면 그 경로를 쓰며, 상대 경로는 config 파일이
    있는 프로젝트 기준이다. 이 프로젝트는 ``../emb``로 상위 공유 캐시를 사용한다.
    """
    base = cfg.get("train", {}).get("embedding_cache_dir") or os.path.join(project_dir, "emb")
    if not os.path.isabs(base):
        base = os.path.normpath(os.path.join(project_dir, base))
    path = os.path.join(base, dataset)
    if create:
        os.makedirs(path, exist_ok=True)
    return path
