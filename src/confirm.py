"""
confirm.py — 매 프레임 예측을 그대로 쓰지 않고, 확실할 때만 단어를 확정한다. (조원 4)

이 로직 하나로 오인식이 눈에 띄게 줄어든다.
카메라 없이도 테스트할 수 있도록 클래스로 분리했다(tests/smoke_test.py 참고).

세 가지 조건을 모두 만족해야 확정
  1) 가장 높은 확률이 임계값(기본 0.85) 이상
  2) 같은 단어가 연속 N번(기본 5) 나옴
  3) 직전 확정으로부터 쿨다운(기본 1.5초) 경과
"""

import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CONF_THRESHOLD, CONSECUTIVE_N, COOLDOWN_SEC


class Confirmer:
    def __init__(self, threshold=CONF_THRESHOLD, n=CONSECUTIVE_N,
                 cooldown=COOLDOWN_SEC, clock=time.time):
        self.threshold = threshold
        self.n = n
        self.cooldown = cooldown
        self.clock = clock                  # 테스트에서 가짜 시계를 넣기 위함
        self.recent = deque(maxlen=n)
        self.last_time = 0.0
        self.last_word = None

    def update(self, class_id: int, confidence: float):
        """
        확정되면 class_id를, 아니면 None을 돌려준다.
        매 예측마다 한 번씩 호출한다.
        """
        if confidence < self.threshold:
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
        self.recent.clear()                 # 같은 단어가 연달아 또 나오는 것 방지
        return class_id

    def reset(self):
        """손이 화면에서 사라졌을 때 호출. 이전 예측 기록을 버린다."""
        self.recent.clear()

    @property
    def progress(self) -> float:
        """0.0~1.0. 화면에 '인식 중' 게이지를 그릴 때 쓴다."""
        return len(self.recent) / self.n
