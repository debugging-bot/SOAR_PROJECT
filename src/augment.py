"""
augment.py — 적은 데이터를 늘린다. (조원 2)

원본 시퀀스 하나에서 조금씩 다른 시퀀스를 여러 개 만들어 학습에 넣는다.
사람이 매번 똑같이 수어를 하지 않으므로, 이 과정이 실제 정확도를 크게 올린다.

**하지 않는 것**
  - 좌우 반전: 수어는 손 위치가 의미를 바꾸므로 뒤집으면 다른 단어가 된다.
  - 검출 여부 플래그(1/0)에 노이즈 추가: 의미가 깨진다.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import HAND_DIM, SEQ_LEN

# 특징 벡터 안에서 '검출 여부' 플래그가 들어 있는 위치 (노이즈 제외 대상)
FLAG_IDX = [HAND_DIM - 1, HAND_DIM * 2 - 1]      # 66, 133


def time_resample(seq: np.ndarray, factor: float) -> np.ndarray:
    """
    동작을 조금 빠르게/느리게 만든 뒤 다시 30프레임으로 맞춘다.
    factor < 1 이면 느린 동작, > 1 이면 빠른 동작을 흉내 낸다.
    """
    n = seq.shape[0]
    src = np.linspace(0, n - 1, num=max(2, int(round(n / factor))))
    stretched = np.stack([np.interp(src, np.arange(n), seq[:, d])
                          for d in range(seq.shape[1])], axis=1)
    dst = np.linspace(0, stretched.shape[0] - 1, num=SEQ_LEN)
    out = np.stack([np.interp(dst, np.arange(stretched.shape[0]), stretched[:, d])
                    for d in range(seq.shape[1])], axis=1)
    # 플래그는 보간하면 0.5 같은 값이 되므로 0/1로 되돌린다
    for i in FLAG_IDX:
        out[:, i] = (out[:, i] > 0.5).astype(np.float32)
    return out.astype(np.float32)


def add_noise(seq: np.ndarray, sigma: float = 0.01, rng=None) -> np.ndarray:
    """좌표에 아주 작은 흔들림을 준다. 손떨림과 검출 오차를 흉내 낸다."""
    rng = rng or np.random.default_rng()
    out = seq.copy()
    noise = rng.normal(0.0, sigma, size=seq.shape).astype(np.float32)
    for i in FLAG_IDX:
        noise[:, i] = 0.0
    # 손이 검출되지 않아 0으로 채운 블록에는 노이즈를 넣지 않는다
    mask = (np.abs(out) > 1e-9).astype(np.float32)
    return (out + noise * mask).astype(np.float32)


def time_shift(seq: np.ndarray, shift: int) -> np.ndarray:
    """동작 시작 시점이 조금 이르거나 늦은 경우를 흉내 낸다."""
    if shift == 0:
        return seq.copy()
    out = np.roll(seq, shift, axis=0)
    if shift > 0:
        out[:shift] = seq[0]
    else:
        out[shift:] = seq[-1]
    return out.astype(np.float32)


def augment_dataset(X: np.ndarray, y: np.ndarray, n_aug: int = 3, seed: int = 42):
    """
    원본 + 변형본을 합쳐 돌려준다. n_aug=3이면 데이터가 4배가 된다.
    """
    rng = np.random.default_rng(seed)
    outX, outY = [X], [y]

    for _ in range(n_aug):
        batch = np.empty_like(X)
        for i in range(len(X)):
            s = X[i]
            s = time_resample(s, float(rng.uniform(0.8, 1.25)))
            s = time_shift(s, int(rng.integers(-3, 4)))
            s = add_noise(s, sigma=0.01, rng=rng)
            batch[i] = s
        outX.append(batch)
        outY.append(y.copy())

    return np.concatenate(outX), np.concatenate(outY)


if __name__ == "__main__":
    demo = np.random.rand(SEQ_LEN, 146).astype(np.float32)
    demo[:, FLAG_IDX] = 1.0
    for f in (0.8, 1.0, 1.25):
        r = time_resample(demo, f)
        print(f"factor={f}: {r.shape}, flag값 {set(np.unique(r[:, FLAG_IDX]).tolist())}")
    print("noise:", add_noise(demo).shape)
    print("shift:", time_shift(demo, 3).shape)
