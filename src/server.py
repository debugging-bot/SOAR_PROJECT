r"""
server.py — 브라우저 한 화면으로 합친 최종 실행 파일. (조원 4 + 조원 5)

구조
    [워커 스레드]  카메라 -> RecognitionEngine(engine.py) -> 확정
            |   카메라가 끊기면 2초마다 다시 연결, 내부 오류가 나도 스스로 다시 시작
            | 확정된 단어 / 상태          | 화면에 보여 줄 JPEG
            v                            v
    [WebSocket /ws]               [MJPEG /video]
            \                           /
             --------- 브라우저 ---------   web/index.html

인터페이스 (산출물정의서 SL-IF-1001~1006)
    GET  /             메인 화면
    GET  /video        카메라 미리보기 (MJPEG)
    WS   /ws           상태 + 확정 단어 (서버 -> 브라우저 단방향)
    GET  /api/labels   인식 어휘 목록 (labels.json)
    GET  /api/status   서버·카메라·모델 상태
    POST /api/session  인식 시작·종료  {"action": "start"} / {"action": "stop"}
                       시작하기(S00)를 누르기 전과 종료(S07) 후에는 카메라를 끄고 인식하지 않는다
    GET  /health       /api/status와 같음 (시연 스크립트가 서버 준비를 확인할 때)

영상은 네트워크로 나가지 않고 같은 PC 안에서만 오간다. 그래서 지연이 거의 없다.

실행
    python src/server.py
    브라우저에서 http://localhost:8000 접속

    (모델이 아직 없으면 --demo 로 화면만 먼저 확인할 수 있다)
    python src/server.py --demo
    (예외 주입 시험: POST /api/debug/fault 로 워커에 일부러 오류를 낸다)
    python src/server.py --debug
"""

import argparse
import asyncio
import sys
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import CONF_THRESHOLD, FRAME_STRIDE, KERAS_MODEL, MIRROR, SEQ_LEN
from engine import RecognitionEngine, label_mismatch, load_keras_predict
from labels import id_to_display, load_labels
from landmarker import Landmarker, open_camera
from overlay import draw_overlay
from person import GUIDE_BOX, MAX_HANDS_SCAN, MAX_PEOPLE, PrimaryUserSelector
from sampler import FrameSampler

BOOT_ID = uuid.uuid4().hex[:8]   # 서버를 켤 때마다 바뀐다. 브라우저가 재시작을 알아챈다
MAX_EVENTS = 50
CAM_FAIL_LIMIT = 30              # read()가 연속 이만큼(약 1초) 실패하면 끊긴 것으로 본다
CAM_RETRY_SEC = 2.0              # 끊긴 카메라를 다시 열어 보는 간격
STALE_SEC = 2.0                  # 이 시간 동안 새 프레임이 없으면 영상이 멈춘 것

try:    # 카메라를 2초마다 다시 열 때 OpenCV 경고가 콘솔을 덮지 않게 한다
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:                # noqa: BLE001
    pass


def _initial_state():
    return {
        "jpeg": None,          # 최신 프레임(JPEG 바이트). 카메라가 없으면 None -> 검은 화면
        "events": [],          # 확정된 단어 [{"seq":1,"id":4,"word":"아파요","conf":..,"t_confirm":..}]
        "seq": 0,              # 마지막으로 발급한 확정 번호. 이력을 비워도 줄어들지 않는다
        "status": "IDLE",      # IDLE / CAMERA_ERROR / WAIT / READY / READING  (UI 설계서 상태와의 대응은 build_app 참고)
        "progress": 0.0,
        "fps": 0.0,
        "infer_ms": 0.0,
        "errors": {},          # 출처별 오류 문구. 원인이 해결되면 그 항목만 지운다
        "camera_ok": False,
        "model": "loading",    # loading / ready / missing / error / demo
        "last_frame_t": 0.0,
        "session": False,      # 시작하기(S00)를 눌러야 True. False면 카메라를 끄고 인식하지 않는다
        "reset_req": False,
        "fault_req": False,
        "running": True,
    }


STATE = _initial_state()
LOCK = threading.RLock()
_ERROR_ORDER = ("camera", "model", "landmarker", "worker")


def set_error(key, msg):
    """출처별로 오류를 기록한다. msg=None이면 그 출처의 오류만 지운다."""
    with LOCK:
        if msg is None:
            STATE["errors"].pop(key, None)
        else:
            STATE["errors"][key] = msg


def snapshot():
    """/api/status와 WebSocket이 함께 쓰는 현재 상태."""
    with LOCK:
        errs = [STATE["errors"][k] for k in _ERROR_ORDER if k in STATE["errors"]]
        return {
            "boot_id": BOOT_ID,
            "status": STATE["status"],
            "progress": round(STATE["progress"], 2),
            "fps": STATE["fps"],
            "infer_ms": round(STATE["infer_ms"], 1),
            "error": " / ".join(errs) or None,
            "camera_connected": STATE["camera_ok"],
            "stale": (STATE["camera_ok"]
                      and time.time() - STATE["last_frame_t"] > STALE_SEC),
            "model": STATE["model"],
            "session_active": STATE["session"],
            "last_seq": STATE["seq"],
            "guide_box": list(GUIDE_BOX),   # 가이드 프레임 (x0, y0, x1, y1), 화면 비율
        }


class Worker:
    """
    카메라를 읽고 인식하는 배경 작업. 웹 요청과 별개로 계속 돈다.

    무슨 일이 있어도 이 루프는 끝나지 않는다 (NFR-05).
      - 카메라가 없거나 끊기면 CAMERA_ERROR를 알리고 retry_sec마다 다시 연다
      - 모델 파일이 없으면 영상만 송출하고 안내 문구를 띄운다
      - 루프 안에서 예외가 나면 기록하고 잠시 쉰 뒤 다시 시작한다

    카메라·랜드마커·모델을 바깥에서 바꿔 넣을 수 있다.
    tests/smoke_test.py는 가짜 카메라로 끊김·복구·오류 재시작을 확인한다.
    """

    def __init__(self, demo, threshold, open_cam=open_camera,
                 make_landmarker=lambda: Landmarker(max_hands=MAX_HANDS_SCAN,
                                                    max_poses=MAX_PEOPLE),
                 load_predict=load_keras_predict,
                 retry_sec=CAM_RETRY_SEC, fail_limit=CAM_FAIL_LIMIT, crash_backoff=1.0):
        self.demo = demo
        self.threshold = threshold
        self.open_cam = open_cam
        self.make_landmarker = make_landmarker
        self.load_predict = load_predict
        self.retry_sec = retry_sec
        self.fail_limit = fail_limit
        self.crash_backoff = crash_backoff
        self.disp = id_to_display(load_labels())
        self.lm = None
        self.cap = None
        self.was_connected = False

    # ------------------------------------------------------------ 준비
    def _prepare(self):
        predict = None
        if self.demo:
            with LOCK:
                STATE["model"] = "demo"
        else:
            try:
                print("[worker] 모델 불러오는 중...")
                predict = self.load_predict()                # 서버 시작 시 한 번만
                with LOCK:
                    STATE["model"] = "ready"
                print("[worker] 모델 준비 완료")
                msg = label_mismatch(predict, len(self.disp))
                if msg:                                     # 동작은 계속하되 화면에 알린다
                    set_error("model", msg)
                    print(f"[worker] {msg}")
            except FileNotFoundError as e:
                with LOCK:
                    STATE["model"] = "missing"
                set_error("model", "모델 파일이 없습니다. 영상만 보여 드립니다")
                print(f"[worker] {e}")
            except Exception as e:                           # noqa: BLE001
                with LOCK:
                    STATE["model"] = "error"
                set_error("model", f"모델을 불러오지 못했습니다: {e}")
                print(f"[worker] 모델을 불러오지 못했습니다: {e}")

        try:
            self.lm = self.make_landmarker()
        except Exception as e:                               # noqa: BLE001
            set_error("landmarker", f"손 인식 모듈을 열지 못했습니다: {e}")
            print(f"[worker] 손 인식 모듈을 열지 못했습니다: {e}")

        print(f"[worker] 인식 창 {FrameSampler.window_seconds():.1f}초 "
              f"(표본 {SEQ_LEN}개, 카메라 {FRAME_STRIDE}프레임마다 1개)")
        return RecognitionEngine(predict, threshold=self.threshold, landmarker=self.lm,
                                 selector=PrimaryUserSelector())

    # ------------------------------------------------------------ 카메라
    def _open_camera(self, engine):
        try:
            cap = self.open_cam()
        except Exception as e:                               # noqa: BLE001
            msg = ("카메라 연결이 끊겼습니다. 케이블을 확인해 주세요 (자동으로 다시 연결합니다)"
                   if self.was_connected else
                   "카메라를 찾을 수 없습니다. 연결을 확인해 주세요")
            with LOCK:
                first = "camera" not in STATE["errors"]
                STATE.update(camera_ok=False, status="CAMERA_ERROR", jpeg=None, progress=0.0)
            set_error("camera", msg)
            if first:
                print(f"[worker] {e}\n[worker] {self.retry_sec:g}초마다 다시 시도합니다")
            return None

        self.was_connected = True
        engine.reset()
        with LOCK:
            STATE["camera_ok"] = True
            STATE["last_frame_t"] = time.time()
        set_error("camera", None)
        print("[worker] 카메라 연결됨")
        return cap

    def _camera_lost(self, cap):
        print("[worker] 카메라 연결이 끊겼습니다")
        try:
            cap.release()
        except Exception:                                    # noqa: BLE001
            pass
        with LOCK:
            STATE.update(camera_ok=False, status="CAMERA_ERROR", jpeg=None, progress=0.0)
        set_error("camera", "카메라 연결이 끊겼습니다. 케이블을 확인해 주세요 (자동으로 다시 연결합니다)")

    # ------------------------------------------------------------ 한 프레임
    def _take_requests(self, engine):
        with LOCK:
            reset, fault = STATE["reset_req"], STATE["fault_req"]
            STATE["reset_req"] = STATE["fault_req"] = False
            session = STATE["session"]
        if reset:
            engine.reset()
        if fault:
            raise RuntimeError("테스트용 강제 예외 (--debug)")
        return session

    def _publish(self, cid, conf):
        word = self.disp.get(cid, str(cid))
        t_ms = int(time.time() * 1000)       # 지연시간 측정용 (브라우저 speak 시각과 비교)
        with LOCK:
            STATE["seq"] += 1
            seq = STATE["seq"]
            STATE["events"].append({"seq": seq, "id": cid, "word": word,
                                    "conf": round(conf, 3), "t_confirm": t_ms})
            STATE["events"] = STATE["events"][-MAX_EVENTS:]
        print(f"[worker] 확정 #{seq}: {word} ({conf:.2f}) t_confirm={t_ms}")

    def _process(self, engine, frame):
        if MIRROR:
            frame = cv2.flip(frame, 1)
        if self.lm is None:
            return frame, "WAIT", 0.0

        res = engine.step(frame)
        if res.confirmed is not None:
            self._publish(res.confirmed, res.conf)

        # 인식에 쓰는 손만 초록 점으로 가볍게 그린다 (점 전부 보기는 realtime.py)
        draw_overlay(frame, res.hands, full=False)
        return frame, res.status, res.progress

    # ------------------------------------------------------------ 감독 루프
    def _sleep(self, sec):
        end = time.time() + sec
        while STATE["running"] and time.time() < end:
            time.sleep(0.05)

    def _tick(self, engine):
        """감독 루프의 한 바퀴. 예외는 run()이 받아서 처리한다."""
        # 세션 밖(S00 시작 전, S07 종료 후)에는 카메라를 끄고 인식하지 않는다
        if not self._take_requests(engine):
            if self.cap is not None:
                self.cap.release()
                self.cap = None
                print("[worker] 세션 종료 — 카메라를 끕니다")
            self.was_connected = False
            set_error("camera", None)
            with LOCK:
                STATE.update(camera_ok=False, jpeg=None, status="IDLE", progress=0.0)
            self._sleep(0.05)
            return

        if self.cap is None:
            self.cap = self._open_camera(engine)
            if self.cap is None:
                self._sleep(self.retry_sec)
                return
            self.fails = 0

        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.fails += 1
            if self.fails >= self.fail_limit:
                self._camera_lost(self.cap)
                self.cap = None
            else:
                time.sleep(0.03)
            return
        self.fails = 0

        frame, status, progress = self._process(engine, frame)

        self.fps_n += 1
        now = time.time()
        cur = None
        if now - self.fps_t >= 1.0:
            cur = self.fps_n / (now - self.fps_t)
            self.fps_n, self.fps_t = 0, now

        ok_enc, jpg = cv2.imencode(".jpg", frame,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        with LOCK:
            if ok_enc:
                STATE["jpeg"] = jpg.tobytes()
            STATE["status"] = status
            STATE["progress"] = progress
            STATE["last_frame_t"] = now
            STATE["infer_ms"] = engine.infer_ms
            if cur is not None:
                STATE["fps"] = round(cur, 1)

    def run(self):
        engine = self._prepare()
        self.cap = None
        self.fails = 0                         # 카메라 read() 연속 실패 횟수
        self.fps_t, self.fps_n = time.time(), 0
        crashes = 0                            # 루프 내부 오류 연속 횟수

        while STATE["running"]:
            try:
                self._tick(engine)
            except Exception:                                # noqa: BLE001
                crashes += 1
                traceback.print_exc()
                set_error("worker", "일시적인 오류가 발생해 다시 시작합니다")
                engine.reset()
                self._sleep(min(self.crash_backoff * 2 ** (crashes - 1), 10.0))
                continue
            if crashes:              # 한 바퀴를 예외 없이 돌았으면 다시 정상이다
                crashes = 0
                set_error("worker", None)
                print("[worker] 다시 정상 동작합니다")

        if self.cap is not None:
            self.cap.release()
        if self.lm is not None:
            self.lm.close()
        print("[worker] 종료")


def build_app(demo: bool, threshold: float, start_worker: bool = True,
              debug: bool = False, auto_start: bool = False):
    """
    서버 status와 UI 설계서(SL-CM-02) 화면의 대응
        IDLE          S00 시작 / S07 종료   (세션 밖, 카메라 꺼짐)
        WAIT          S01 대기              (손 미검출)
        READY         S02 준비됨            (손 검출, 확정 진행 0)
        READING       S03 읽는 중           (확정 진행 중, progress 0~1)
        words 도착    S04 단어 확정         (브라우저가 표시·TTS 후 S01로)
        CAMERA_ERROR  E01 카메라 오류       (복구되면 WAIT)
    """
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    from typing import Literal

    with LOCK:
        STATE.clear()
        STATE.update(_initial_state())
        STATE["session"] = auto_start

    @asynccontextmanager
    async def lifespan(_app):
        th = None
        if start_worker:
            th = threading.Thread(target=Worker(demo, threshold).run,
                                  name="worker", daemon=True)
            th.start()
        yield
        STATE["running"] = False
        if th is not None:
            th.join(timeout=2.0)

    app = FastAPI(title="SOAR 수어 번역", lifespan=lifespan)
    web_dir = ROOT / "web"

    # ---------------------------------------------------- SL-IF-1001 메인 화면
    @app.get("/")
    def index():
        # 캐시 헤더가 없으면 브라우저가 예전 화면을 그대로 보여 줄 수 있어 매번 확인하게 한다
        return FileResponse(web_dir / "index.html",
                            headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

    # ---------------------------------------------------- SL-IF-1002 영상
    def gen_frames():
        _, bjpg = cv2.imencode(".jpg", np.zeros((480, 640, 3), np.uint8))
        blank_bytes = bjpg.tobytes()
        while STATE["running"]:
            with LOCK:
                data = STATE["jpeg"] or blank_bytes
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
            time.sleep(1 / 30)

    @app.get("/video")
    def video():
        return StreamingResponse(
            gen_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame")

    # ---------------------------------------------------- SL-IF-1004 어휘
    @app.get("/api/labels")
    def api_labels():
        return {"words": load_labels()}

    # ---------------------------------------------------- SL-IF-1005 상태
    @app.get("/api/status")
    def api_status():
        return {**snapshot(), "demo": demo}

    @app.get("/health")
    def health():
        return {**snapshot(), "demo": demo}

    # ---------------------------------------------------- SL-IF-1006 세션
    class SessionReq(BaseModel):
        action: Literal["start", "stop"]

    @app.post("/api/session")
    def api_session(req: SessionReq):
        with LOCK:
            STATE["session"] = req.action == "start"
            STATE["events"].clear()      # 새 대화 시작(S00)·종료(S07) 때 이력을 비운다
            STATE["reset_req"] = True    # 워커가 다음 프레임에서 버퍼를 비운다
            # 시작하면 워커가 카메라를 연다. 실패하면 곧 CAMERA_ERROR(E01)로 바뀐다
            STATE["status"] = "WAIT" if STATE["session"] else "IDLE"
            STATE["progress"] = 0.0
        print(f"[server] 세션 {req.action}")
        return snapshot()

    if debug:
        @app.post("/api/debug/fault")
        def api_fault():
            with LOCK:
                STATE["fault_req"] = True
            return {"ok": True}

    # ---------------------------------------------------- SL-IF-1003 확정 이벤트
    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        # 다시 연결할 때 지난 단어를 처음부터 보내면 브라우저가 전부 다시 읽는다.
        # 같은 서버(boot 일치)에 since=마지막 seq를 알려 주면 그 뒤부터, 아니면 지금부터 보낸다.
        q = sock.query_params
        with LOCK:
            seen = STATE["seq"]
            if q.get("boot") == BOOT_ID and q.get("since", "").isdigit():
                seen = min(int(q["since"]), STATE["seq"])
        try:
            while True:
                payload = snapshot()
                with LOCK:
                    new = [e for e in STATE["events"] if e["seq"] > seen]
                if new:
                    seen = new[-1]["seq"]
                payload["words"] = new
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
    ap.add_argument("--debug", action="store_true",
                    help="예외 주입 시험용 POST /api/debug/fault 를 연다")
    ap.add_argument("--auto-start", action="store_true",
                    help="시작하기 버튼 없이 바로 인식 (세션 API를 아직 안 쓰는 화면에서 시험할 때)")
    args = ap.parse_args()

    # 모델을 불러오기 전에 포트부터 확인한다 (이전 서버가 켜져 있으면 바로 알려 주고 끝낸다)
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", args.port)) == 0:
            print(f"[오류] 포트 {args.port}을(를) 이미 다른 프로그램이 쓰고 있습니다.")
            print(f"       이전에 켠 서버가 남아 있을 가능성이 큽니다: http://localhost:{args.port}")
            print("       그 서버 창에서 Ctrl+C로 끄고 다시 실행하거나, --port 8001 처럼 다른 포트를 쓰세요.")
            return 1

    if not args.demo and not KERAS_MODEL.exists():
        print(f"[주의] 모델 파일이 없습니다: {KERAS_MODEL}")
        print("       서버는 켜지지만 영상만 보입니다. 화면만 보려면: python src/server.py --demo")

    import uvicorn
    app = build_app(args.demo, args.threshold, debug=args.debug,
                    auto_start=args.auto_start)
    print(f"\n브라우저에서 http://localhost:{args.port} 를 여세요. (종료: Ctrl+C)\n")
    # 영상·인식 결과가 PC 밖으로 나가지 않도록 이 PC 안에서만 접속을 받는다 (설계 원칙)
    # /video 영상 스트림은 끝나지 않으므로, 브라우저가 열려 있으면 Ctrl+C를 눌러도
    # 연결이 끝나기를 영영 기다린다. 2초만 기다리고 끊는다.
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning",
                timeout_graceful_shutdown=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
