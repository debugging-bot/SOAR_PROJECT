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
res3 = [c.update(3, 0.95) for _ in range(5)]
# 쿨다운 동안에도 연속 기록은 쌓여 있으므로, 시간이 지나자마자 확정된다
ok("쿨다운이 지나면 다시 확정", res3.count(3) == 1, f"{res3}")

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
try:
    import tensorflow  # noqa: F401
    from train import build_model
    m = build_model(12)
    out = m.predict(np.zeros((2, SEQ_LEN, FEATURE_DIM), dtype=np.float32), verbose=0)
    ok("모델 출력 shape", out.shape == (2, 12), f"{out.shape}")
    ok("확률 합 = 1", np.allclose(out.sum(axis=1), 1.0, atol=1e-4))
except ImportError:
    print("[skip] TensorFlow 미설치 — 학습 담당(조원 3)만 확인하면 됩니다")

print()
print("=" * 60)
print(f"통과 {passed}개 / 실패 {failed}개")
print("=" * 60)
raise SystemExit(1 if failed else 0)
