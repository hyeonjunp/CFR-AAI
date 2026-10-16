"""체크포인트 슬림화 + Hugging Face 업로드/다운로드.

    python scripts/ckpt_tool.py slim     [--src checkpoints] [--dst checkpoints_slim]
    python scripts/ckpt_tool.py upload   --repo USER/NAME [--dir checkpoints_slim]
    python scripts/ckpt_tool.py download --repo USER/NAME [--dir checkpoints_slim]

slim   : optimizer/scheduler 를 버리고 model_state + 메타(epoch, best_*)만 남긴다.
         ablang/ (외부 사전학습 가중치)는 setup_ablang_weights.py 로 받으므로 건너뜀.
         MANIFEST.json(sha256, 크기)도 같이 생성. 슬림본은 추론/feature 추출용 (학습 재개 불가).
upload : 비공개 repo 생성(없으면) 후 폴더 통째 업로드. 사전에 `huggingface-cli login`.
download: repo 전체를 --dir 로 받는다.
"""
import argparse
import hashlib
import json
import os

KEEP_META = ("epoch", "best_metric", "best_val_f1", "best_epoch")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def slim(src, dst):
    import torch
    manifest = {}
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d != "ablang"]
        for fn in files:
            if not fn.endswith(".pt"):
                continue
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, src)
            ck = torch.load(p, map_location="cpu", weights_only=False)
            if not (isinstance(ck, dict) and "model_state" in ck):
                print(f"skip (model_state 없음): {rel}")
                continue
            out = os.path.join(dst, rel)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            torch.save({"model_state": ck["model_state"],
                        **{k: ck[k] for k in KEEP_META if k in ck}}, out)
            manifest[rel] = {"sha256": sha256(out), "bytes": os.path.getsize(out)}
            print(f"{rel}: {os.path.getsize(p)/2**20:.1f}MB -> {manifest[rel]['bytes']/2**20:.1f}MB")
    with open(os.path.join(dst, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("slim")
    s.add_argument("--src", default="checkpoints")
    s.add_argument("--dst", default="checkpoints_slim")
    for name in ("upload", "download"):
        p = sub.add_parser(name)
        p.add_argument("--repo", required=True)
        p.add_argument("--dir", default="checkpoints_slim")
    a = ap.parse_args()

    if a.cmd == "slim":
        slim(a.src, a.dst)
    else:
        from huggingface_hub import HfApi, snapshot_download
        if a.cmd == "upload":
            api = HfApi()
            api.create_repo(a.repo, private=True, exist_ok=True)
            api.upload_folder(repo_id=a.repo, folder_path=a.dir, allow_patterns=["*.pt", "MANIFEST.json"])
        else:
            snapshot_download(a.repo, local_dir=a.dir)


if __name__ == "__main__":
    main()
