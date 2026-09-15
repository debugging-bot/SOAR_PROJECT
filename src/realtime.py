"""
realtime.py — 웹캠 앞에서 실시간으로 단어를 인식한다. (조원 4)

서버도 브라우저도 필요 없는 가장 단순한 버전이다.
9주차 완료 기준("웹캠 앞에서 수어를 하면 단어가 한 번만 출력된다")을
이 파일 하나로 확인한다. 10주차에 이 로직을 server.py가 그대로 가져다 쓴다.

실행
    python src/realtime.py
    python src/realtime.py --threshold 0.75   # 잘 안 잡히면 낮춰 본다

조작
    q : 종료
    r : 버퍼 초기화
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (CONF_THRESHOLD, CONSECUTIVE_N, FEATURE_DIM, FRAME_STRIDE,
                    KERAS_MODEL, MIRROR, PREDICT_EVERY, SEQ_LEN)
from confirm import Confirmer
from features import build_feature_vector, hand_detected_count
from labels import id_to_display, load_labels
from landmarker import Landmarker, open_camera
from sampler import FrameSampler

# ---------------------------------------------------------------- 한글 그리기
_FONT_CANDIDATES = [
    "C:/Windows/Fonts/malgun.ttf",                      # Windows
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",       # macOS
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Linux
]


def _load_font(size=64):
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:      # noqa: BLE001
                pass
    return None


_FONT = _load_font()


def put_korean(img, text, xy, color=(255, 255, 255)):
    """OpenCV는 한글을 못 그리므로 PIL을 빌려 쓴다. 폰트가 없으면 그냥 건너뛴다."""
    if _FONT is None or not text:
        return img
    from PIL import Image, ImageDraw
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text(xy, text, font=_FONT, fill=color[::-1])
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=CONF_THRESHOLD)
    ap.add_argument("--every", type=int, default=PREDICT_EVERY,
                    help="표본 몇 개마다 예측할 것인가 (프레임이 아니라 표본 기준)")
    args = ap.parse_args()

    if not KERAS_MODEL.exists():
        print(f"모델 파일이 없습니다: {KERAS_MODEL}")
        print("먼저 데이터를 모으고(python src/collect.py) 학습하세요(python src/train.py).")
        return 1

    from tensorflow import keras
    print("모델 불러오는 중...")
    model = keras.models.load_model(KERAS_MODEL)

    words = load_labels()
    disp = id_to_display(words)

    cap = open_camera()
    lm = Landmarker()
    confirmer = Confirmer(threshold=args.threshold)

    sampler = FrameSampler()
    frame_no = 0
    fps_t, fps_n, fps = time.time(), 0, 0.0
    top_id, top_conf = -1, 0.0
    history: list[str] = []
    banner, banner_until = "", 0.0

    print(f"인식 창 {FrameSampler.window_seconds():.1f}초 "
          f"(표본 {SEQ_LEN}개, 카메라 {FRAME_STRIDE}프레임마다 1개)")
    print("준비 완료. 카메라 앞에서 수어를 해 보세요. (q=종료)")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if MIRROR:
                frame = cv2.flip(frame, 1)
            frame_no += 1

            hands, pose = lm.process(frame)
            vec = build_feature_vector(hands, pose)
            taken = sampler.offer(vec)

            # 손이 오래 안 잡히면 이전 기록을 버린다 (엉뚱한 확정 방지)
            if hand_detected_count(vec) == 0:
                if sampler.full() and all(
                        hand_detected_count(v) == 0 for v in sampler.recent(10)):
                    confirmer.reset()

            if taken and sampler.full() and sampler.taken % args.every == 0:
                x = sampler.array()[np.newaxis, ...]               # (1,30,146)
                prob = model.predict(x, verbose=0)[0]
                top_id, top_conf = int(np.argmax(prob)), float(np.max(prob))

                confirmed = confirmer.update(top_id, top_conf)
                if confirmed is not None:
                    text = disp.get(confirmed, str(confirmed))
                    history.append(text)
                    banner, banner_until = text, time.time() + 2.0
                    print(f"  >>> 확정: {text}  (확신도 {top_conf:.2f})")

            # ---------------- 화면 ----------------
            for hand in hands:
                for (x_, y_, _z) in hand:
                    cv2.circle(frame, (int(x_ * frame.shape[1]),
                                       int(y_ * frame.shape[0])),
                               3, (0, 220, 120), -1)

            fps_n += 1
            if time.time() - fps_t >= 1.0:
                fps, fps_n, fps_t = fps_n / (time.time() - fps_t), 0, time.time()

            state = ("WAIT" if hand_detected_count(vec) == 0
                     else ("READING" if confirmer.progress > 0 else "READY"))
            color = {"WAIT": (120, 120, 120), "READY": (0, 200, 255),
                     "READING": (0, 255, 0)}[state]
            cv2.putText(frame, f"{state}  fps={fps:4.1f}  buf={len(sampler)}/{SEQ_LEN}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if top_id >= 0:
                cv2.putText(frame, f"top=id{top_id} conf={top_conf:.2f}",
                            (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            # 확정까지 얼마나 남았는지 게이지
            g = int(frame.shape[1] * confirmer.progress)
            cv2.rectangle(frame, (0, frame.shape[0] - 8), (g, frame.shape[0]),
                          (0, 255, 0), -1)

            if time.time() < banner_until:
                frame = put_korean(frame, banner, (20, frame.shape[0] // 2 - 40),
                                   (0, 255, 255))

            cv2.imshow("realtime - TeamMate", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                sampler.reset()
                confirmer.reset()
    finally:
        cap.release()
        lm.close()
        cv2.destroyAllWindows()

    print("\n이번 세션에서 확정된 단어:", " / ".join(history) if history else "(없음)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
