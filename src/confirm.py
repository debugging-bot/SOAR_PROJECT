"""
confirm.py — 매 프레임 예측을 그대로 쓰지 않고, 확실할 때만 단어를 확정한다. (조원 4)

이 로직 하나로 오인식이 눈에 띄게 줄어든다.
카메라 없이도 테스트할 수 있도록 클래스로 분리했다(tests/smoke_test.py 참고).

세 가지 조건을 모두 만족해야 확정
  1) 가장 높은 확률이 임계값(기본 0.85) 이상
  2) 같은 단어가 연속 N번(기본 5) 나옴
  3) 직전 확정으로부터 쿨다운(config.COOLDOWN_SEC, 현재 2.5초) 경과

재확정 잠금 (10주차 중간발표 피드백 4)
  '아니요'처럼 반복하는 동작을 계속하면 쿨다운이 지날 때마다 같은 단어가 또 확정됐다.
  그래서 방금 확정한 단어는 잠그고, 아래 중 하나가 확인될 때까지 다시 확정하지 않는다.
    - 손을 내림: 엔진이 reset()을 부른다 (손 미검출이 이어질 때)
    - 다른 동작: 잠긴 단어가 아닌 예측(다른 단어이거나 확신 낮음)이 release_n번 연속
  반복 동작 중에 확률이 한두 번 떨어지는 것으로는 풀리지 않도록 '연속'을 요구한다.
  '네, 네'처럼 일부러 반복할 때는 손을 내렸다가 다시 하면 된다.
"""

import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CONF_THRESHOLD, CONSECUTIVE_N, COOLDOWN_SEC

# 잠금을 풀기 위해 필요한 '잠긴 단어가 아닌 예측'의 연속 횟수.
# 예측 주기 0.13초 기준 약 0.7초 동안 다른 동작이 이어져야 풀린다.
RELEASE_N = 5


class Confirmer:
    def __init__(self, threshold=CONF_THRESHOLD, n=CONSECUTIVE_N,
                 cooldown=COOLDOWN_SEC, release_n=RELEASE_N, clock=time.monotonic):
        # monotonic: PC 시간이 자동 동기화로 바뀌어도 쿨다운 계산이 흔들리지 않는다
        self.threshold = threshold
        self.n = n
        self.cooldown = cooldown
        self.release_n = release_n
        self.clock = clock                  # 테스트에서 가짜 시계를 넣기 위함
        self.recent = deque(maxlen=n)
        self.last_time = float("-inf")      # 첫 확정은 쿨다운 없이
        self.last_word = None
        self.locked = None                  # 재확정이 잠긴 단어 (없으면 None)
        self.other = 0                      # 잠긴 단어가 아닌 예측의 연속 횟수

    def update(self, class_id: int, confidence: float):
        """
        확정되면 class_id를, 아니면 None을 돌려준다.
        매 예측마다 한 번씩 호출한다.
        """
        high = confidence >= self.threshold

        if self.locked is not None:
            if high and class_id == self.locked:
                self.other = 0
                self.recent.clear()         # 잠긴 단어는 진행 게이지도 올리지 않는다
                return None
            self.other += 1
            if self.other >= self.release_n:
                self.locked = None

        if not high:
            self.recent.clear()             # 확신 없으면 연속 카운트 초기화
            return None

        self.recent.append(class_id)
        if len(self.recent) < self.n:
            return None
        if len(set(self.recent)) != 1:      # 연속 N번이 모두 같은 단어여야 한다
            return None

        now = self.clock()
        if now - self.last_time < self.cooldown:
            return None

        self.last_time = now
        self.last_word = class_id
        self.locked = class_id
        self.other = 0
        self.recent.clear()                 # 같은 단어가 연달아 또 나오는 것 방지
        return class_id

    def reset(self):
        """손이 화면에서 사라졌을 때 호출. 이전 예측 기록을 버리고 재확정 잠금도 푼다."""
        self.recent.clear()
        self.locked = None
        self.other = 0

    @property
    def progress(self) -> float:
        """0.0~1.0. 화면에 '인식 중' 게이지를 그릴 때 쓴다."""
        return len(self.recent) / self.n
