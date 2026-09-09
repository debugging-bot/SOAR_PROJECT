"""
download_models.py — MediaPipe 모델 파일 2개를 내려받는다.

실행 (프로젝트 루트 teammate/ 에서):
    python setup/download_models.py

한 번만 실행하면 models/ 폴더에 파일이 생기고, 이후에는 다시 받지 않는다.
회사/학교 방화벽 때문에 실패하면 아래 URL을 브라우저 주소창에 직접 붙여 넣어
파일을 받은 뒤 models/ 폴더에 넣어도 똑같다.
"""

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import HAND_TASK, MODEL_DIR, POSE_TASK  # noqa: E402

FILES = [
    (HAND_TASK,
     "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
     "hand_landmarker/float16/1/hand_landmarker.task"),
    (POSE_TASK,
     "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
     "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"),
]


def _progress(block_num, block_size, total_size):
    if total_size <= 0:
        return
    done = min(block_num * block_size, total_size)
    pct = done * 100 / total_size
    print(f"\r    {pct:5.1f}%  ({done/1e6:.1f}MB / {total_size/1e6:.1f}MB)",
          end="", flush=True)


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    failed = []

    for path, url in FILES:
        if path.exists() and path.stat().st_size > 100_000:
            print(f"[skip] 이미 있음: {path.name}")
            continue
        print(f"[down] {path.name}")
        print(f"       {url}")
        try:
            urllib.request.urlretrieve(url, path, _progress)
            print(f"\n[ ok ] 저장: {path}")
        except Exception as e:                      # noqa: BLE001
            print(f"\n[fail] {e}")
            failed.append((path, url))

    if failed:
        print("\n실패한 파일이 있습니다. 브라우저로 아래 주소를 열어 직접 받은 뒤")
        print(f"{MODEL_DIR} 폴더에 같은 이름으로 넣어 주세요.")
        for path, url in failed:
            print(f"  - {path.name}: {url}")
        return 1

    print("\n완료. 이제 python setup/check_env.py 를 실행하세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
