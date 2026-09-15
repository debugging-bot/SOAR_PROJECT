"""
sampler.py — 카메라 프레임을 FRAME_STRIDE 간격으로 골라 시퀀스 버퍼에 담는다.

**이 파일이 존재하는 이유**
촬영(collect.py), 실시간 인식(realtime.py), 서버(server.py)가 프레임을 고르는
방식이 조금이라도 다르면 "학습 정확도는 90%인데 실시간에서는 하나도 못 맞추는"
문제가 생긴다. 그래서 세 파일이 전부 이 클래스 하나만 쓴다.
프레임 건너뛰기 로직을 다른 곳에 따로 쓰지 말 것.

현재 설정 (config.py)
    SEQ_LEN = 30, FRAME_STRIDE = 2
    -> 카메라 2장마다 1장을 담아서 30장을 모으면 2.0초 분량
    -> 모델 입력은 (30, 146) 그대로
"""

from collections import deque

import numpy as np

from config import FEATURE_DIM, FRAME_STRIDE, SEQ_LEN


class FrameSampler:
    """
    사용법
        sampler = FrameSampler()
        for 매 프레임:
            vec = build_feature_vector(hands, pose)
            taken = sampler.offer(vec)          # 담겼으면 True
            if sampler.full() and taken:
                x = sampler.array()[np.newaxis, ...]   # (1, 30, 146)
    """

    def __init__(self, maxlen: int = SEQ_LEN, stride: int = FRAME_STRIDE):
        self.maxlen = int(maxlen)
        self.stride = max(1, int(stride))
        self.buf: deque = deque(maxlen=self.maxlen)
        self.frames = 0        # offer()가 불린 총 횟수 (카메라 프레임 수)
        self.taken = 0         # 실제로 담긴 표본 수

    def offer(self, vec: np.ndarray) -> bool:
        """프레임마다 한 번씩 호출한다. 이번 프레임이 담겼으면 True."""
        take = (self.frames % self.stride == 0)
        self.frames += 1
        if take:
            self.buf.append(vec)
            self.taken += 1
        return take

    def full(self) -> bool:
        return len(self.buf) == self.maxlen

    def array(self) -> np.ndarray:
        """(SEQ_LEN, FEATURE_DIM) 배열. 아직 안 찼으면 ValueError."""
        if not self.full():
            raise ValueError(f"버퍼가 아직 안 찼습니다: {len(self.buf)}/{self.maxlen}")
        arr = np.array(self.buf, dtype=np.float32)
        assert arr.shape == (self.maxlen, FEATURE_DIM), f"shape 오류: {arr.shape}"
        return arr

    def recent(self, n: int) -> list:
        """가장 최근 표본 n개. 손이 사라졌는지 판단할 때 쓴다."""
        return list(self.buf)[-n:]

    def reset(self):
        """손이 화면에서 사라졌을 때 등. 프레임 카운터까지 초기화한다."""
        self.buf.clear()
        self.frames = 0
        self.taken = 0

    def __len__(self) -> int:
        return len(self.buf)

    @property
    def progress(self) -> float:
        """0.0~1.0. 촬영 진행 막대에 쓴다."""
        return len(self.buf) / self.maxlen

    @staticmethod
    def window_seconds(fps: float = 30.0) -> float:
        """이 설정으로 담기는 시간(초). 화면에 표시하거나 점검할 때 쓴다."""
        return SEQ_LEN * FRAME_STRIDE / fps


if __name__ == "__main__":
    s = FrameSampler()
    print(f"SEQ_LEN={SEQ_LEN}, FRAME_STRIDE={FRAME_STRIDE}")
    print(f"담기는 시간 = {FrameSampler.window_seconds():.2f}초")
    print(f"표본 간격   = {FRAME_STRIDE / 30:.3f}초")
    need = 0
    while not s.full():
        s.offer(np.zeros(FEATURE_DIM, dtype=np.float32))
        need += 1
    print(f"버퍼를 채우는 데 필요한 카메라 프레임 수 = {need}장")
