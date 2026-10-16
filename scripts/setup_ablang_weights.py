"""checkpoints/ablang/{heavy,light}/ 의 AbLang 가중치를 `ablang` 패키지의 캐시 위치에 설치한다.

ablang.pretrained(model_folder="download") 는 <site-packages>/ablang/model-weights-<chain>/amodel.pt 가
없으면 opig.stats.ox.ac.uk 에서 내려받는데, 이 서버가 403 을 돌려줄 때가 있다.
이 스크립트로 미리 넣어두면 다운로드를 건너뛴다.   사용: python scripts/setup_ablang_weights.py
"""
import os
import shutil

import ablang

src_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints", "ablang")
for chain in ("heavy", "light"):
    src = os.path.join(src_root, chain)
    dst = os.path.join(os.path.dirname(ablang.__file__), f"model-weights-{chain}")
    os.makedirs(dst, exist_ok=True)
    for f in ("amodel.pt", "hparams.json", "vocab.json"):
        shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    print(f"[ablang] {chain}: {dst}")
