"""
collect.py — 학습용 수어 데이터를 촬영해 저장한다. (조원 1, 조원 2)

영상 파일은 저장하지 않는다. 좌표만 뽑아서 (30, 146) 배열 하나를 .npy로 남긴다.
용량이 작고(약 18KB), 개인 얼굴이 저장되지 않는다.

실행 예시 (프로젝트 루트 teammate/ 에서)
    python src/collect.py --word pain --person kim --speed normal --count 30
    python src/collect.py --word pain --person kim --speed slow   --count 15

  --word    labels.json의 slug (영문). 예: pain
  --person  촬영자 영문 이름. 예: kim, lee, oh, shin, jung
  --speed   normal(표준 속도) 또는 slow(고령 사용자 재현: 천천히, 작게)
  --count   이번에 몇 번 찍을 것인가

조작
    SPACE : 한 번 촬영 시작 (3-2-1 카운트다운 후 30프레임 녹화)
    a     : 자동 반복 모드 켜기/끄기 (쉬는 시간 2초 후 자동으로 다음 촬영)
    d     : 방금 저장한 파일 삭제 (동작을 실수했을 때)
    q     : 종료

화면 글씨는 OpenCV가 한글을 못 그리기 때문에 영어로 나온다. 정상이다.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_DIR, MIRROR, SEQ_LEN
from features import build_feature_vector, hand_detected_count
from labels import load_labels, slug_to_id
from landmarker import Landmarker, open_camera

# 손이 하나도 안 잡힌 프레임이 이 비율을 넘으면 저장하지 않고 다시 찍게 한다
MAX_EMPTY_RATIO = 0.30


def draw_guide(frame, text_lines, color=(255, 255, 255)):
    h, w = frame.shape[:2]
    # 손을 두어야 할 영역 가이드
    cv2.rectangle(frame, (int(w * 0.15), int(h * 0.10)),
                  (int(w * 0.85), int(h * 0.95)), (80, 80, 80), 1)
    y = 26
    for line in text_lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 1, cv2.LINE_AA)
        y += 26


def draw_hands(frame, hands):
    h, w = frame.shape[:2]
    for hand in hands:
        for (x, y, _z) in hand:
            cv2.circle(frame, (int(x * w), int(y * h)), 3, (0, 220, 120), -1)


def next_index(folder: Path, person: str, speed: str) -> int:
    existing = list(folder.glob(f"{person}_{speed}_*.npy"))
    nums = []
    for p in existing:
        try:
            nums.append(int(p.stem.split("_")[-1]))
        except ValueError:
            pass
    return max(nums) + 1 if nums else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", required=True, help="labels.json의 slug")
    ap.add_argument("--person", required=True, help="촬영자 영문 이름")
    ap.add_argument("--speed", default="normal", choices=["normal", "slow"])
    ap.add_argument("--count", type=int, default=30)
    args = ap.parse_args()

    words = load_labels()
    s2i = slug_to_id(words)
    if args.word not in s2i:
        print(f"'{args.word}' 는 labels.json에 없습니다. 가능한 값:")
        print("  " + ", ".join(s2i.keys()))
        return 1

    folder = DATA_DIR / args.word
    folder.mkdir(parents=True, exist_ok=True)
    idx = next_index(folder, args.person, args.speed)

    ko = next(w["ko"] for w in words if w["slug"] == args.word)
    print(f"단어: {args.word} ({ko}) / 촬영자: {args.person} / 속도: {args.speed}")
    print(f"목표: {args.count}회, 저장 위치: {folder}")
    print("창을 클릭해 활성화한 뒤 SPACE를 누르세요. (q=종료, a=자동반복, d=직전삭제)")

    cap = open_camera()
    lm = Landmarker()

    saved = 0
    auto = False
    last_saved_path = None
    state = "idle"          # idle -> countdown -> recording
    count_from = 0.0
    buf: list[np.ndarray] = []
    flash = ""
    flash_until = 0.0

    try:
        while saved < args.count:
            ok, frame = cap.read()
            if not ok:
                print("카메라에서 프레임을 읽지 못했습니다.")
                break
            if MIRROR:
                frame = cv2.flip(frame, 1)

            hands, pose = lm.process(frame)
            vec = build_feature_vector(hands, pose)

            view = frame.copy()
            draw_hands(view, hands)

            now = time.time()
            if state == "idle":
                draw_guide(view, [
                    f"word={args.word}  person={args.person}  speed={args.speed}",
                    f"saved {saved}/{args.count}   next #{idx:03d}",
                    "SPACE=record   a=auto:%s   d=delete last   q=quit" % ("ON" if auto else "off"),
                    f"hands detected: {hand_detected_count(vec)}",
                ], (0, 255, 255))
                if auto and now - count_from > 2.0:
                    state, count_from = "countdown", now

            elif state == "countdown":
                left = 3.0 - (now - count_from)
                if left <= 0:
                    state, buf = "recording", []
                else:
                    draw_guide(view, ["GET READY"], (0, 200, 255))
                    cv2.putText(view, str(int(left) + 1),
                                (view.shape[1] // 2 - 30, view.shape[0] // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 200, 255), 6)

            elif state == "recording":
                buf.append(vec)
                p = len(buf) / SEQ_LEN
                w = view.shape[1]
                cv2.rectangle(view, (0, 0), (int(w * p), 10), (0, 0, 255), -1)
                draw_guide(view, [f"RECORDING {len(buf)}/{SEQ_LEN}"], (0, 0, 255))

                if len(buf) == SEQ_LEN:
                    seq = np.array(buf, dtype=np.float32)      # (30, 146)
                    empty = sum(1 for v in seq if hand_detected_count(v) == 0)
                    if empty / SEQ_LEN > MAX_EMPTY_RATIO:
                        flash = f"REJECTED (no hand in {empty}/{SEQ_LEN} frames)"
                        print(f"  버림: 손이 안 잡힌 프레임 {empty}개. 조명/거리를 조정하고 다시 찍으세요.")
                    else:
                        path = folder / f"{args.person}_{args.speed}_{idx:03d}.npy"
                        np.save(path, seq)
                        last_saved_path = path
                        saved += 1
                        idx += 1
                        flash = f"SAVED {path.name}"
                        print(f"  저장 {saved}/{args.count}: {path.name}  shape={seq.shape}")
                    flash_until = now + 1.2
                    state, count_from = "idle", now

            if now < flash_until:
                cv2.putText(view, flash, (12, view.shape[0] - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                cv2.putText(view, flash, (12, view.shape[0] - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            cv2.imshow("collect - TeamMate", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and state == "idle":
                state, count_from = "countdown", time.time()
            if key == ord("a"):
                auto = not auto
                count_from = time.time()
            if key == ord("d") and last_saved_path and last_saved_path.exists():
                last_saved_path.unlink()
                saved = max(0, saved - 1)
                idx = max(1, idx - 1)
                print(f"  삭제: {last_saved_path.name}")
                last_saved_path = None
    finally:
        cap.release()
        lm.close()
        cv2.destroyAllWindows()

    print(f"\n종료. 이번 세션 저장 {saved}개. 폴더 전체 개수: {len(list(folder.glob('*.npy')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
