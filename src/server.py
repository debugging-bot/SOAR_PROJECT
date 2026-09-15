r"""
server.py — 브라우저 한 화면으로 합친 최종 실행 파일. (조원 4 + 조원 5)

구조
    [백그라운드 스레드]  카메라 -> 랜드마크 -> 특징 -> 모델 -> 확정
            |                    |
            | 확정된 단어         | 화면에 보여 줄 JPEG
            v                    v
    [WebSocket /ws]        [MJPEG /video]
            \                   /
             ------ 브라우저 ------   web/index.html

영상은 네트워크로 나가지 않고 같은 PC 안에서만 오간다. 그래서 지연이 거의 없다.

실행
    python src/server.py
    브라우저에서 http://localhost:8000 접속

    (모델이 아직 없으면 --demo 로 화면만 먼저 확인할 수 있다)
    python src/server.py --demo
"""

import argparse
import asyncio
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import (CONF_THRESHOLD, FRAME_STRIDE, KERAS_MODEL, MIRROR,
                    PREDICT_EVERY, SEQ_LEN)
from confirm import Confirmer
from features import build_feature_vector, hand_detected_count
from labels import id_to_display, load_labels
from landmarker import Landmarker, open_camera
from sampler import FrameSampler

STATE = {
    "jpeg": None,          # 최신 프레임(JPEG 바이트)
    "events": [],          # 확정된 단어 목록 [{"seq":1,"word":"아파요","t":...}]
    "status": "WAIT",      # WAIT / READY / READING
    "progress": 0.0,
    "fps": 0.0,
    "error": None,
    "running": True,
}
LOCK = threading.Lock()


def worker(demo: bool, threshold: float):
    """카메라를 읽고 인식하는 배경 작업. 웹 요청과 별개로 계속 돈다."""
    model = None
    disp = id_to_display(load_labels())

    if not demo:
        try:
            from tensorflow import keras
            print("[worker] 모델 불러오는 중...")
            model = keras.models.load_model(KERAS_MODEL)   # 서버 시작 시 한 번만
            print("[worker] 모델 준비 완료")
        except Exception as e:                             # noqa: BLE001
            with LOCK:
                STATE["error"] = f"모델을 불러오지 못했습니다: {e}"
            print(f"[worker] {STATE['error']}")
            return

    try:
        cap = open_camera()
        lm = Landmarker()
    except Exception as e:                                 # noqa: BLE001
        with LOCK:
            STATE["error"] = str(e)
        print(f"[worker] {e}")
        return

    sampler = FrameSampler()
    confirmer = Confirmer(threshold=threshold)
    frame_no, seq_id = 0, 0
    print(f"[worker] 인식 창 {FrameSampler.window_seconds():.1f}초 "
          f"(표본 {SEQ_LEN}개, 카메라 {FRAME_STRIDE}프레임마다 1개)")
    fps_t, fps_n = time.time(), 0

    try:
        while STATE["running"]:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            if MIRROR:
                frame = cv2.flip(frame, 1)
            frame_no += 1

            hands, pose = lm.process(frame)
            vec = build_feature_vector(hands, pose)
            taken = sampler.offer(vec)

            n_hand = hand_detected_count(vec)
            if n_hand == 0 and len(sampler) >= 10 and all(
                    hand_detected_count(v) == 0 for v in sampler.recent(10)):
                confirmer.reset()

            if (model is not None and taken and sampler.full()
                    and sampler.taken % PREDICT_EVERY == 0):
                x = sampler.array()[np.newaxis, ...]
                prob = model.predict(x, verbose=0)[0]
                cid, conf = int(np.argmax(prob)), float(np.max(prob))
                done = confirmer.update(cid, conf)
                if done is not None:
                    seq_id += 1
                    word = disp.get(done, str(done))
                    with LOCK:
                        STATE["events"].append(
                            {"seq": seq_id, "word": word, "conf": round(conf, 3)})
                        STATE["events"] = STATE["events"][-50:]
                    print(f"[worker] 확정 #{seq_id}: {word} ({conf:.2f})")

            # 손 위치 점만 가볍게 그린다
            h, w = frame.shape[:2]
            for hand in hands:
                for (x_, y_, _z) in hand:
                    cv2.circle(frame, (int(x_ * w), int(y_ * h)), 3, (0, 220, 120), -1)

            fps_n += 1
            if time.time() - fps_t >= 1.0:
                cur = fps_n / (time.time() - fps_t)
                fps_n, fps_t = 0, time.time()
            else:
                cur = None

            ok_enc, jpg = cv2.imencode(".jpg", frame,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            with LOCK:
                if ok_enc:
                    STATE["jpeg"] = jpg.tobytes()
                STATE["status"] = ("WAIT" if n_hand == 0
                                   else ("READING" if confirmer.progress > 0 else "READY"))
                STATE["progress"] = confirmer.progress
                if cur is not None:
                    STATE["fps"] = round(cur, 1)
    finally:
        cap.release()
        lm.close()
        print("[worker] 종료")


def build_app(demo: bool, threshold: float):
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="TeamMate 수어 번역")
    web_dir = ROOT / "web"

    th = threading.Thread(target=worker, args=(demo, threshold), daemon=True)
    th.start()

    @app.get("/")
    def index():
        return FileResponse(web_dir / "index.html")

    app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

    def gen_frames():
        blank = np.zeros((480, 640, 3), np.uint8)
        _, bjpg = cv2.imencode(".jpg", blank)
        blank_bytes = bjpg.tobytes()
        while True:
            with LOCK:
                data = STATE["jpeg"] or blank_bytes
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
            time.sleep(1 / 30)

    @app.get("/video")
    def video():
        return StreamingResponse(
            gen_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/health")
    def health():
        with LOCK:
            return {"status": STATE["status"], "fps": STATE["fps"],
                    "error": STATE["error"], "demo": demo}

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        seen = 0
        try:
            while True:
                with LOCK:
                    new = [e for e in STATE["events"] if e["seq"] > seen]
                    payload = {"status": STATE["status"],
                               "progress": round(STATE["progress"], 2),
                               "fps": STATE["fps"],
                               "error": STATE["error"],
                               "words": new}
                if new:
                    seen = new[-1]["seq"]
                await sock.send_json(payload)
                await asyncio.sleep(0.08)
        except WebSocketDisconnect:
            pass
        except Exception:            # noqa: BLE001
            pass

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true",
                    help="모델 없이 화면과 카메라만 확인")
    ap.add_argument("--threshold", type=float, default=CONF_THRESHOLD)
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    if not args.demo and not KERAS_MODEL.exists():
        print(f"모델이 없습니다: {KERAS_MODEL}")
        print("화면만 먼저 보려면: python src/server.py --demo")
        return 1

    import uvicorn
    app = build_app(args.demo, args.threshold)
    print(f"\n브라우저에서 http://localhost:{args.port} 를 여세요. (종료: Ctrl+C)\n")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    STATE["running"] = False
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
