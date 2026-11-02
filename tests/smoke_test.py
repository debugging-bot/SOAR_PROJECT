"""
smoke_test.py — 카메라도 모델도 없이 코드가 제대로 도는지 확인한다.

실행 (프로젝트 루트 teammate/ 에서)
    python tests/smoke_test.py

전부 통과하면 특징 추출 -> 증강 -> 데이터 로드 -> 확정 로직이
서로 맞물려 돌아간다는 뜻이다. 1주차에 한 번, 그리고 코드를 고칠 때마다 돌린다.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from augment import add_noise, augment_dataset, time_resample, time_shift  # noqa: E402
from config import FEATURE_DIM, FRAME_STRIDE, SEQ_LEN  # noqa: E402
from confirm import Confirmer  # noqa: E402
from features import (build_feature_vector, empty_feature_vector,  # noqa: E402
                      hand_detected_count, normalize_hand_shape)

passed, failed = 0, 0


def ok(name, cond, extra=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"[ OK ] {name} {extra}")
    else:
        failed += 1
        print(f"[FAIL] {name} {extra}")


def fake_hand(center=(0.5, 0.5), scale=0.10, seed=0):
    """가짜 손 좌표 (21,3). center 위치에 scale 크기로 만든다."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, (21, 3)).astype(np.float32)
    base[0] = 0.0                       # 손목을 원점에
    base[9] = np.array([0, 1, 0])       # 중지 밑동
    out = base * scale
    out[:, 0] += center[0]
    out[:, 1] += center[1]
    return out.astype(np.float32)


def fake_pose(shoulder_y=0.45, width=0.30):
    p = np.zeros((33, 3), dtype=np.float32)
    p[11] = (0.5 - width / 2, shoulder_y, 0.0)   # 왼 어깨
    p[12] = (0.5 + width / 2, shoulder_y, 0.0)   # 오른 어깨
    p[13] = (0.5 - width / 2, shoulder_y + 0.2, 0.0)
    p[14] = (0.5 + width / 2, shoulder_y + 0.2, 0.0)
    return p


print("=" * 60)
print("1. 특징 추출")
print("=" * 60)

v = build_feature_vector([fake_hand()], fake_pose())
ok("한 손 -> 146차원", v.shape == (FEATURE_DIM,), f"shape={v.shape}")
ok("손 개수 세기", hand_detected_count(v) == 1)

v2 = build_feature_vector(
    [fake_hand(center=(0.35, 0.5), seed=1), fake_hand(center=(0.65, 0.5), seed=2)],
    fake_pose())
ok("두 손 -> 슬롯 2개 채움", hand_detected_count(v2) == 2)

v0 = build_feature_vector([], None)
ok("아무것도 없을 때 0벡터", np.allclose(v0, 0) and v0.shape == (FEATURE_DIM,))
ok("empty_feature_vector 일치", np.allclose(v0, empty_feature_vector()))

# 정규화가 실제로 거리·위치를 지우는지 (가장 중요한 검증)
near = fake_hand(center=(0.5, 0.5), scale=0.20, seed=7)   # 카메라에 가까이
far = fake_hand(center=(0.5, 0.5), scale=0.08, seed=7)    # 멀리
d = np.abs(normalize_hand_shape(near) - normalize_hand_shape(far)).max()
ok("거리가 달라도 손 모양 특징은 같다", d < 1e-4, f"최대 차이 {d:.2e}")

left = fake_hand(center=(0.2, 0.4), scale=0.12, seed=7)
right = fake_hand(center=(0.8, 0.4), scale=0.12, seed=7)
d2 = np.abs(normalize_hand_shape(left) - normalize_hand_shape(right)).max()
ok("위치가 달라도 손 모양 특징은 같다", d2 < 1e-4, f"최대 차이 {d2:.2e}")

# 반대로, 손 '위치' 정보는 남아 있어야 한다
a = build_feature_vector([fake_hand(center=(0.5, 0.20), seed=7)], fake_pose())
b = build_feature_vector([fake_hand(center=(0.5, 0.70), seed=7)], fake_pose())
ok("얼굴 앞 / 가슴 앞은 다른 값이어야 한다",
   not np.allclose(a, b), f"차이 {np.abs(a - b).max():.3f}")

print()
print("=" * 60)
print("2. 데이터 증강")
print("=" * 60)

seq = np.stack([build_feature_vector([fake_hand(center=(0.5, 0.5 - i * 0.01), seed=3)],
                                     fake_pose()) for i in range(SEQ_LEN)])
ok("시퀀스 만들기", seq.shape == (SEQ_LEN, FEATURE_DIM), f"shape={seq.shape}")

for f in (0.8, 1.0, 1.25):
    r = time_resample(seq, f)
    ok(f"시간 리샘플 {f}배 -> 길이 30 유지", r.shape == (SEQ_LEN, FEATURE_DIM))

flags = set(np.unique(time_resample(seq, 0.8)[:, [66, 133]]).tolist())
ok("리샘플 후 검출 플래그가 0/1 유지", flags <= {0.0, 1.0}, f"{flags}")

n = add_noise(seq)
ok("노이즈 후 shape 유지", n.shape == seq.shape)
ok("노이즈는 아주 작아야 한다", np.abs(n - seq).max() < 0.2,
   f"최대 {np.abs(n - seq).max():.3f}")
ok("빈 손 슬롯에는 노이즈가 들어가지 않는다",
   np.allclose(n[:, 67:133][seq[:, 67:133] == 0], 0))
ok("시간 이동", time_shift(seq, 3).shape == seq.shape)

X = np.stack([seq] * 4)
y = np.array([0, 1, 0, 1])
Xa, ya = augment_dataset(X, y, n_aug=2)
ok("증강 배수", len(Xa) == 12 and len(ya) == 12, f"{len(Xa)}개")

print()
print("=" * 60)
print("3. 데이터셋 저장/불러오기")
print("=" * 60)

tmp = Path(tempfile.mkdtemp())
try:
    import dataset as ds
    from labels import load_labels
    words = load_labels()
    use = [w["slug"] for w in words[:3]]
    for k, slug in enumerate(use):
        (tmp / slug).mkdir(parents=True)
        for i in range(4):
            np.save(tmp / slug / f"kim_normal_{i:03d}.npy",
                    (seq + k * 0.01).astype(np.float32))
    Xd, yd, meta = ds.load_dataset(tmp, verbose=False)
    ok("불러온 개수", len(yd) == 12, f"{len(yd)}개")
    ok("배열 shape", Xd.shape == (12, SEQ_LEN, FEATURE_DIM), f"{Xd.shape}")
    ok("촬영자 파싱", meta[0]["person"] == "kim" and meta[0]["speed"] == "normal")
    try:
        import sklearn  # noqa: F401  split_random이 안에서 쓴다
    except Exception as e:                  # noqa: BLE001  깨진 설치는 ImportError가 아닌 ValueError 등
        print(f"[skip] 학습/검증 분할 생략 (scikit-learn을 불러오지 못함: {e})")
    else:
        Xtr, Xva, ytr, yva = ds.split_random(Xd, yd, val_ratio=0.25)
        ok("학습/검증 분할", len(ytr) == 9 and len(yva) == 3, f"{len(ytr)}/{len(yva)}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print()
print("=" * 60)
print("4. 확정 로직")
print("=" * 60)

t = [1000.0]
c = Confirmer(threshold=0.85, n=5, cooldown=1.5, clock=lambda: t[0])

ok("확신도 낮으면 확정 안 됨",
   all(c.update(3, 0.60) is None for _ in range(10)))

res = [c.update(3, 0.95) for _ in range(5)]
ok("연속 5번이면 확정", res[:4] == [None] * 4 and res[4] == 3, f"{res}")

res2 = [c.update(3, 0.95) for _ in range(5)]
ok("쿨다운 중에는 다시 확정 안 됨", all(r is None for r in res2))

t[0] += 2.0
res3 = [c.update(3, 0.95) for _ in range(20)]
ok("쿨다운이 지나도 같은 단어는 잠겨서 재확정 안 됨",
   all(r is None for r in res3) and c.progress == 0, f"progress={c.progress}")

c.reset()                                           # 손을 내림
res4 = [c.update(3, 0.95) for _ in range(5)]
ok("손을 내렸다 다시 하면 같은 단어 재확정", res4[-1] == 3, f"{res4}")

t[0] += 3.0
for _ in range(4):
    c.update(3, 0.40)                               # 확률이 잠깐 떨어짐 (4번)
blip = [c.update(3, 0.95) for _ in range(5)]
ok("확률이 잠깐 떨어진 정도로는 잠금 유지", all(r is None for r in blip), f"{blip}")
for _ in range(5):
    c.update(7, 0.40)                               # 다른 동작이 5번 연속
res5 = [c.update(3, 0.95) for _ in range(5)]
ok("다른 동작이 이어진 뒤에는 같은 단어 재확정", res5[-1] == 3, f"{res5}")

t[0] += 3.0
res6 = [c.update(5, 0.95) for _ in range(5)]
ok("다른 단어는 잠금과 무관하게 확정", res6[-1] == 5, f"{res6}")

c.reset()
mixed = [c.update(i % 2, 0.95) for i in range(10)]
ok("다른 단어가 섞이면 확정 안 됨", all(m is None for m in mixed))

print()
print("=" * 60)
print("5. 프레임 샘플러 (촬영/실시간 공통)")
print("=" * 60)

from sampler import FrameSampler  # noqa: E402

sp = FrameSampler()
ok("스트라이드 설정", sp.stride == FRAME_STRIDE, f"stride={sp.stride}")
ok("담기는 시간", abs(FrameSampler.window_seconds() - SEQ_LEN * FRAME_STRIDE / 30) < 1e-9,
   f"{FrameSampler.window_seconds():.2f}초")

frames = 0
while not sp.full():
    sp.offer(np.zeros(FEATURE_DIM, dtype=np.float32))
    frames += 1
    if frames > 10000:
        break
ok("버퍼가 채워진다", sp.full(), f"카메라 {frames}프레임 소비")
ok("소비 프레임 수가 스트라이드에 비례",
   frames == (SEQ_LEN - 1) * FRAME_STRIDE + 1, f"{frames}장")
ok("array() shape", sp.array().shape == (SEQ_LEN, FEATURE_DIM), f"{sp.array().shape}")
ok("표본 수 집계", sp.taken == SEQ_LEN, f"taken={sp.taken}")

sp2 = FrameSampler()
taken_flags = [sp2.offer(np.zeros(FEATURE_DIM, dtype=np.float32)) for _ in range(6)]
expect = [i % FRAME_STRIDE == 0 for i in range(6)]
ok("담는 프레임 간격이 일정", taken_flags == expect, f"{taken_flags}")

sp2.reset()
ok("reset 후 비워짐", len(sp2) == 0 and sp2.frames == 0 and sp2.taken == 0)

sp3 = FrameSampler()
for i in range(500):
    sp3.offer(np.full(FEATURE_DIM, float(i), dtype=np.float32))
last = sp3.array()[-1][0]
ok("가장 최근 표본이 버퍼 끝에 있다", last > 490, f"마지막 값 {last:.0f}")

print()
print("=" * 60)
print("6. 모델 (TensorFlow가 있을 때만)")
print("=" * 60)
tf_imported = False
try:
    import tensorflow  # noqa: F401
    from labels import load_labels
    from train import build_model
    tf_imported = True
    n_cls = len(load_labels())
    m = build_model(n_cls)
    out = m.predict(np.zeros((2, SEQ_LEN, FEATURE_DIM), dtype=np.float32), verbose=0)
    ok("labels.json 클래스 수", n_cls > 0, f"{n_cls}개")
    ok("모델 출력 shape", out.shape == (2, n_cls), f"{out.shape}")
    ok("확률 합 = 1", np.allclose(out.sum(axis=1), 1.0, atol=1e-4))
except Exception as e:                      # noqa: BLE001  깨진 설치는 ImportError가 아닌 ValueError 등
    if tf_imported:     # 점검 도중에 난 오류는 생략하지 않고 그대로 실패로 드러낸다
        raise
    print(f"[skip] TensorFlow를 불러오지 못함 ({type(e).__name__}) — 학습 담당(조원 3)만 확인하면 됩니다")

print()
print("=" * 60)
print("7. 실시간 엔진 (카메라·모델 없이)")
print("=" * 60)

from engine import RecognitionEngine  # noqa: E402

hand_vec = build_feature_vector([fake_hand()], fake_pose())
empty_vec = build_feature_vector([], fake_pose())
calls = [0]


def fake_predict(x, cid=4):
    calls[0] += 1
    p = np.full(10, 0.01, dtype=np.float32)
    p[cid] = 0.91
    return p


eng = RecognitionEngine(fake_predict, threshold=0.85, clock=lambda: 1000.0)
for _ in range(200):
    eng.feed(empty_vec)
ok("손이 없으면 예측 함수를 부르지 않는다", calls[0] == 0, f"호출 {calls[0]}회")

eng.reset()
got = [r.confirmed for r in (eng.feed(hand_vec) for _ in range(300)) if r.confirmed is not None]
ok("같은 단어가 계속 나오면 정확히 1번 확정", got == [4], f"{got}")
ok("예측 1회 시간 기록", eng.infer_ms >= 0.0, f"{eng.infer_ms:.3f}ms")

eng2 = RecognitionEngine(fake_predict, threshold=0.85, clock=lambda: 1000.0)
for _ in range(SEQ_LEN * FRAME_STRIDE + 2):
    eng2.feed(hand_vec)
before = eng2.confirmer.progress
for _ in range(10 * FRAME_STRIDE):
    r = eng2.feed(empty_vec)
ok("손이 사라지면 진행률이 0으로", before > 0 and r.progress == 0 and r.status == "WAIT",
   f"{before:.1f} -> {r.progress:.1f}")

# 실제 시간 흐름(30fps)으로 같은 동작을 계속 유지했을 때 (10주차 피드백 4)
tt = [5000.0]
eng4 = RecognitionEngine(fake_predict, threshold=0.85, clock=lambda: tt[0])


def run_frames(vec, sec):
    out = []
    for _ in range(int(sec * 30)):
        tt[0] += 1 / 30
        r = eng4.feed(vec)
        if r.confirmed is not None:
            out.append(r.confirmed)
    return out


hold = run_frames(hand_vec, 5.0)
ok("동작 유지 5초 = 확정 1회", hold == [4], f"{hold}")
hold2 = run_frames(hand_vec, 5.0)
ok("10초까지 계속 유지해도 추가 확정 없음", hold2 == [], f"{hold2}")
run_frames(empty_vec, 1.0)
again = run_frames(hand_vec, 5.0)
ok("손을 내렸다 다시 하면 다시 확정", again == [4], f"{again}")

# 손이 안 잡힌 표본이 60%를 넘는 창은 학습 데이터에 없으므로 예측하지 않는다
eng3 = RecognitionEngine(fake_predict, threshold=0.85, clock=lambda: 1000.0)
calls[0] = 0
for i in range(300):
    eng3.feed(hand_vec if i % 3 == 0 else empty_vec)    # 손 1/3만 보임
ok("손이 거의 안 보이는 창은 예측 생략", calls[0] == 0, f"호출 {calls[0]}회")

print()
print("=" * 60)
print("8. 서버 API (FastAPI가 있을 때만, 카메라 없이)")
print("=" * 60)
server_imported = False
try:
    import threading  # noqa: E402
    import time  # noqa: E402

    # httpx2(또는 httpx)가 없으면 starlette가 ImportError가 아니라 RuntimeError를 낸다
    from fastapi.testclient import TestClient  # noqa: E402

    import server  # noqa: E402
    from labels import load_labels as _ll  # noqa: E402
    server_imported = True

    def wait_until(cond, timeout=5.0):
        end = time.time() + timeout
        while time.time() < end:
            if cond():
                return True
            time.sleep(0.02)
        return False

    app = server.build_app(demo=True, threshold=0.85, start_worker=False)
    with TestClient(app) as cli:
        lab = cli.get("/api/labels").json()
        ok("/api/labels 단어 수 = labels.json", len(lab["words"]) == len(_ll()),
           f"{len(lab['words'])}개")
        st = cli.get("/api/status").json()
        need = {"status", "fps", "camera_connected", "model", "session_active", "error", "boot_id"}
        ok("/api/status 필드", need <= set(st), f"{sorted(need - set(st)) or 'OK'}")
        ok("시작하기 전에는 세션 꺼짐 (S00)",
           st["session_active"] is False and st["status"] == "IDLE")

        with server.LOCK:
            server.STATE["seq"] = 2
            server.STATE["events"] = [{"seq": 1, "word": "네"}, {"seq": 2, "word": "아니요"}]
        with cli.websocket_connect("/ws") as w:
            first = w.receive_json()
        ok("WS 재연결 시 지난 단어를 다시 보내지 않는다", first["words"] == [], f"{first['words']}")
        with cli.websocket_connect(f"/ws?boot={server.BOOT_ID}&since=1") as w:
            first = w.receive_json()
        ok("WS since 이후 단어만 받는다", [e["seq"] for e in first["words"]] == [2])

        r = cli.post("/api/session", json={"action": "stop"}).json()
        ok("세션 종료 -> 인식 중지 + 이력 비움",
           r["session_active"] is False and server.STATE["events"] == [])
        r = cli.post("/api/session", json={"action": "start"}).json()
        ok("세션 시작", r["session_active"] is True)
        ok("잘못된 세션 요청은 422", cli.post("/api/session", json={"action": "x"}).status_code == 422)

    # 가짜 카메라로 워커의 끊김·복구·오류 재시작을 확인한다
    class FakeCam:
        def __init__(self, plug):
            self.plug = plug

        def read(self):
            time.sleep(0.005)
            if not self.plug[0]:
                return False, None
            return True, np.zeros((120, 160, 3), np.uint8)

        def release(self):
            pass

    class FakeLm:
        def process(self, frame):
            return [fake_hand()], fake_pose()

        def close(self):
            pass

    plug = [False]
    opened = [0]

    def fake_open():
        opened[0] += 1
        if not plug[0]:
            raise RuntimeError("가짜 카메라 없음")
        return FakeCam(plug)

    server.build_app(demo=False, threshold=0.85, start_worker=False)   # STATE 초기화
    wk = server.Worker(False, 0.85, open_cam=fake_open, make_landmarker=FakeLm,
                       load_predict=lambda: fake_predict, retry_sec=0.05,
                       fail_limit=5, crash_backoff=0.05)
    th = threading.Thread(target=wk.run, daemon=True)
    th.start()
    try:
        time.sleep(0.3)
        ok("시작하기 전에는 카메라를 열지 않는다", opened[0] == 0, f"열기 {opened[0]}회")
        server.STATE["session"] = True
        ok("카메라 없이 시작 -> CAMERA_ERROR + 안내", wait_until(
            lambda: server.snapshot()["status"] == "CAMERA_ERROR"
            and "카메라" in (server.snapshot()["error"] or "")))
        plug[0] = True
        ok("카메라 연결 -> 오류 해제", wait_until(
            lambda: server.snapshot()["camera_connected"] and server.snapshot()["error"] is None))
        ok("가짜 카메라로 단어 확정", wait_until(lambda: server.STATE["seq"] >= 1),
           f"seq={server.STATE['seq']}")
        plug[0] = False
        ok("실행 중 분리 -> 끊김 안내", wait_until(
            lambda: not server.snapshot()["camera_connected"]
            and "끊겼" in (server.snapshot()["error"] or "")))
        plug[0] = True
        ok("다시 꽂으면 자동 복구", wait_until(lambda: server.snapshot()["camera_connected"]))
        server.STATE["fault_req"] = True
        ok("워커 내부 오류 -> 안내", wait_until(
            lambda: "일시적" in (server.snapshot()["error"] or "")))
        ok("워커 오류 후 자동 재시작", wait_until(
            lambda: server.snapshot()["error"] is None) and th.is_alive())
        server.STATE["session"] = False
        ok("종료하기 -> 카메라 끄고 IDLE (S07)", wait_until(
            lambda: server.snapshot()["status"] == "IDLE"
            and not server.snapshot()["camera_connected"]))
        server.STATE["session"] = True
        ok("다시 시작하기 -> 카메라 다시 연결", wait_until(
            lambda: server.snapshot()["camera_connected"]))
    finally:
        server.STATE["running"] = False
        th.join(timeout=2.0)

    server.build_app(demo=False, threshold=0.85, start_worker=False, auto_start=True)

    def no_model():
        raise FileNotFoundError("없음")
    plug[0] = True
    wk = server.Worker(False, 0.85, open_cam=fake_open, make_landmarker=FakeLm,
                       load_predict=no_model, retry_sec=0.05)
    th = threading.Thread(target=wk.run, daemon=True)
    th.start()
    try:
        ok("모델 파일 없음 -> 서버 유지 + 안내", wait_until(
            lambda: server.snapshot()["model"] == "missing"
            and server.snapshot()["camera_connected"]
            and "모델" in (server.snapshot()["error"] or "")))
    finally:
        server.STATE["running"] = False
        th.join(timeout=2.0)

    # 카메라가 없는 동안 난 워커 오류도 지워져야 한다 (카메라 안내만 남음)
    server.build_app(demo=False, threshold=0.85, start_worker=False, auto_start=True)
    plug[0] = False
    wk = server.Worker(False, 0.85, open_cam=fake_open, make_landmarker=FakeLm,
                       load_predict=lambda: fake_predict, retry_sec=0.05,
                       crash_backoff=0.05)
    th = threading.Thread(target=wk.run, daemon=True)
    th.start()
    try:
        wait_until(lambda: server.snapshot()["status"] == "CAMERA_ERROR")
        server.STATE["fault_req"] = True
        seen_fault = wait_until(lambda: "일시적" in (server.snapshot()["error"] or ""))
        ok("카메라 없을 때 워커 오류 -> 오류만 해제, 카메라 안내 유지", seen_fault and wait_until(
            lambda: "일시적" not in (server.snapshot()["error"] or "")
            and "카메라" in (server.snapshot()["error"] or "")))
    finally:
        server.STATE["running"] = False
        th.join(timeout=2.0)
except (ImportError, RuntimeError) as e:
    if server_imported:     # 점검 도중에 난 오류는 생략하지 않고 그대로 실패로 드러낸다
        raise
    print(f"[skip] 서버 점검 생략 ({e}) — pip install -r requirements.txt")

print()
print("=" * 60)
print("9. 주 사용자 고정 (다인원, 카메라 없이)")
print("=" * 60)

from person import PrimaryUserSelector  # noqa: E402


def person_pose(cx, width, y=0.45):
    """어깨 중심 x=cx, 어깨 폭 width인 사람. 손목(15·16)은 어깨 바로 아래."""
    p = np.zeros((33, 3), dtype=np.float32)
    p[11] = (cx - width / 2, y, 0.0)
    p[12] = (cx + width / 2, y, 0.0)
    p[13] = (cx - width / 2, y + 0.2, 0.0)
    p[14] = (cx + width / 2, y + 0.2, 0.0)
    p[15] = (cx - width / 2, y + 0.1, 0.0)
    p[16] = (cx + width / 2, y + 0.1, 0.0)
    return p


user = person_pose(0.50, 0.30)                  # 정면 사용자
side = person_pose(0.82, 0.12)                  # 옆에 멀리 선 사람
u_hands = [fake_hand((0.35, 0.55), seed=1), fake_hand((0.65, 0.55), seed=2)]
s_hands = [fake_hand((0.76, 0.55), seed=3), fake_hand((0.88, 0.55), seed=4)]


def same(a, b):
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


h, p = PrimaryUserSelector().select(u_hands, [user])
ok("1명일 때는 이전과 같은 손·상체", same(h, u_hands) and p is user)

h, p = PrimaryUserSelector().select(s_hands + u_hands, [side, user])
ok("옆 사람 손이 먼저 잡혀도 사용자 손만 남긴다", same(h, u_hands) and p is user,
   f"손 {len(h)}개")

h, _ = PrimaryUserSelector().select(u_hands + [fake_hand((0.05, 0.5), seed=5)], [user])
ok("가이드 프레임 밖 손은 버린다", same(h, u_hands), f"손 {len(h)}개")

# 뒤에 선 사람: 몸이 가려져 상체는 안 잡히고, 손만 사용자 어깨 너머로 보임 (실측에서 발견)
behind = [fake_hand((0.62, 0.22), seed=6), fake_hand((0.30, 0.25), seed=7)]
sel = PrimaryUserSelector()
h, p = sel.select(behind + u_hands, [user])
ok("뒤 사람 상체가 안 잡혀도 그 손은 버린다", same(h, u_hands) and p is user,
   f"남은 손 {len(h)}개, 버린 손 {len(sel.rejected)}개")
h, _ = PrimaryUserSelector().select(behind, [user])
ok("사용자 손이 내려가 있으면 뒤 사람 손만으로 인식하지 않음", h == [], f"손 {len(h)}개")

sel = PrimaryUserSelector()
sel.select(u_hands, [user])
near = person_pose(0.28, 0.45)                  # 더 가까운 사람이 들어옴
_, p = sel.select(u_hands, [near, user])
ok("더 가까운 사람이 와도 주 사용자 유지", p is user)
_, p = sel.select([], [near])
ok("사용자가 나가면 남은 사람으로 다시 선택", p is near)

h, p = PrimaryUserSelector().select(u_hands, [person_pose(0.96, 0.15)])
ok("가이드 안에 사람이 없으면 손을 모두 버림 (대기)", h == [] and p is None)

h, p = PrimaryUserSelector().select(u_hands, [])
ok("상체를 못 찾으면 손만 그대로 사용", same(h, u_hands) and p is None)


class CrowdLm:
    def process_all(self, frame):
        return s_hands + u_hands, [side, user]


eng5 = RecognitionEngine(None, landmarker=CrowdLm(), selector=PrimaryUserSelector())
r = eng5.step(np.zeros((120, 160, 3), np.uint8))
ok("엔진이 주 사용자 손만 특징에 넣는다", r.n_hand == 2 and same(r.hands, u_hands))

print()
print("=" * 60)
print(f"통과 {passed}개 / 실패 {failed}개")
print("=" * 60)
raise SystemExit(1 if failed else 0)
