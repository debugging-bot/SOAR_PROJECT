"""
engine.py — 프레임 한 장을 받아 인식을 한 단계 진행한다. (조원 4)

realtime.py(콘솔)와 server.py(브라우저)는 반드시 같은 추론 규칙을 써야 한다.
예전에는 두 파일에 루프가 복사되어 있었고, 손이 사라졌을 때의 리셋 조건이 이미
서로 달라져 있었다. 이제 두 파일 모두 이 클래스 하나만 쓴다.

한 단계에서 하는 일
    프레임 -> Landmarker -> (주 사용자 손만, person.py) -> build_feature_vector -> FrameSampler
          -> (손이 충분히 보일 때만) 모델 예측 -> Confirmer
          -> 확정되면 버퍼를 비우고 그 단어는 재확정 잠금 (confirm.py)

모델 입력 특징(features.py, sampler.py, config.py)은 건드리지 않는다.
여기서 더한 규칙(손 검출 게이팅)은 예측을 '할지 말지'만 정하므로 데이터 규격과 무관하다.

카메라와 TensorFlow 없이 테스트할 수 있도록 예측 함수를 바깥에서 넣는다
(tests/smoke_test.py 7번 항목 참고).
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CONF_THRESHOLD, FEATURE_DIM, KERAS_MODEL, PREDICT_EVERY, SEQ_LEN
from confirm import Confirmer
from features import build_feature_vector, hand_detected_count
from sampler import FrameSampler

# collect.py와 같은 기준. 학습 데이터에는 손이 안 잡힌 표본이 60%를 넘는 시퀀스가 없으므로
# 그런 창은 모델이 한 번도 본 적 없는 입력이다. 예측하지 않고 건너뛴다.
MAX_EMPTY_RATIO = 0.60
# 최근 표본이 이만큼 연속으로 손 없이 들어오면 이전 예측 기록을 버린다 (엉뚱한 확정 방지)
RESET_AFTER_EMPTY = 10


@dataclass
class StepResult:
    status: str = "WAIT"            # WAIT / READY / READING
    progress: float = 0.0           # 확정까지 진행률 0.0~1.0
    confirmed: int | None = None    # 이번 단계에서 확정된 class id
    conf: float = 0.0               # 확정됐을 때의 확신도
    top_id: int = -1                # 가장 최근 예측 (화면 디버그용)
    top_conf: float = 0.0
    n_hand: int = 0
    predicted: bool = False         # 이번 단계에서 모델을 호출했는가
    hands: list = field(default_factory=list)


class RecognitionEngine:
    """
    사용법
        engine = RecognitionEngine(load_keras_predict(), landmarker=Landmarker())
        for 매 프레임:
            res = engine.step(frame)
            if res.confirmed is not None:
                print(disp[res.confirmed])
    """

    def __init__(self, predict_fn=None, threshold=CONF_THRESHOLD,
                 every=PREDICT_EVERY, landmarker=None, selector=None, clock=time.monotonic):
        self.predict_fn = predict_fn    # (1,30,146) -> (클래스 수,) 확률. None이면 예측 생략
        self.landmarker = landmarker
        self.selector = selector        # person.PrimaryUserSelector. 있으면 주 사용자 손만 쓴다
        self.every = max(1, int(every))
        self.sampler = FrameSampler()
        self.confirmer = Confirmer(threshold=threshold, clock=clock)
        self.top_id, self.top_conf = -1, 0.0
        self.infer_ms = 0.0             # 마지막 예측 1회에 걸린 시간 (NFR-01 측정용)

    def step(self, frame_bgr) -> StepResult:
        """카메라 프레임 한 장. 거울 반전은 호출하는 쪽에서 미리 한다."""
        if self.selector is not None and hasattr(self.landmarker, "process_all"):
            hands, poses = self.landmarker.process_all(frame_bgr)
            hands, pose = self.selector.select(hands, poses)
        else:
            hands, pose = self.landmarker.process(frame_bgr)
        res = self.feed(build_feature_vector(hands, pose))
        res.hands = hands
        return res

    def feed(self, vec: np.ndarray) -> StepResult:
        """특징 벡터 한 개. 카메라 없이 테스트할 때 직접 부른다."""
        taken = self.sampler.offer(vec)
        n_hand = hand_detected_count(vec)
        res = StepResult(n_hand=n_hand)

        recent = self.sampler.recent(RESET_AFTER_EMPTY)
        if (n_hand == 0 and len(recent) >= RESET_AFTER_EMPTY
                and all(hand_detected_count(v) == 0 for v in recent)):
            self.confirmer.reset()

        if (self.predict_fn is not None and taken and self.sampler.full()
                and self.sampler.taken % self.every == 0):
            seq = self.sampler.array()
            empty = sum(1 for v in seq if hand_detected_count(v) == 0)
            if empty / SEQ_LEN > MAX_EMPTY_RATIO:
                self.confirmer.reset()
            else:
                t0 = time.perf_counter()
                prob = np.asarray(self.predict_fn(seq[np.newaxis, ...])).reshape(-1)
                self.infer_ms = (time.perf_counter() - t0) * 1000
                self.top_id, self.top_conf = int(np.argmax(prob)), float(np.max(prob))
                res.predicted = True
                done = self.confirmer.update(self.top_id, self.top_conf)
                if done is not None:
                    res.confirmed, res.conf = done, self.top_conf
                    # 방금 확정한 동작이 창에 남아 다시 예측되지 않도록 버퍼를 비운다.
                    # 다음 예측은 새 표본 30개(약 2초)가 다시 찰 때부터다.
                    self.sampler.reset()

        res.top_id, res.top_conf = self.top_id, self.top_conf
        res.progress = self.confirmer.progress
        res.status = ("WAIT" if n_hand == 0
                      else ("READING" if res.progress > 0 else "READY"))
        return res

    def reset(self):
        """버퍼와 예측 기록을 모두 버린다. 카메라 재연결·세션 시작 때 부른다."""
        self.sampler.reset()
        self.confirmer.reset()
        if self.selector is not None:
            self.selector.reset()
        self.top_id, self.top_conf = -1, 0.0


def keras_predict_fn(model):
    """
    model.predict()는 호출마다 고정 비용이 커서 실시간 루프에 맞지 않는다.
    model(x) 직접 호출은 컴파일되지 않아(eager) LSTM이 더 느리다.
    predict_on_batch는 컴파일된 함수를 재사용해 가장 빠르다.
        2026-10-23 측정 (sign_model.keras, CPU, 1회 평균)
        predict 33.6ms / model(x) 107ms / predict_on_batch 2.0ms
    첫 호출이 느린 것을 막기 위해 한 번 미리 돌린다.
    """
    def predict(x):
        return np.asarray(model.predict_on_batch(x))[0]

    predict(np.zeros((1, SEQ_LEN, FEATURE_DIM), dtype=np.float32))   # warm-up
    predict.n_classes = int(model.output_shape[-1])   # labels.json 단어 수와 비교용
    return predict


def label_mismatch(predict_fn, n_labels):
    """모델 출력 개수와 labels.json 단어 수가 다르면 안내 문구, 같으면 None."""
    n_model = getattr(predict_fn, "n_classes", None)
    if n_model is None or n_model == n_labels:
        return None
    return (f"모델 단어 수({n_model}개)와 labels.json 단어 수({n_labels}개)가 다릅니다. "
            "재학습한 모델로 바꿔 주세요")


def load_keras_predict(path=KERAS_MODEL):
    """모델 파일을 열어 예측 함수를 돌려준다. 파일이 없으면 FileNotFoundError."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {path}")
    from tensorflow import keras
    return keras_predict_fn(keras.models.load_model(path))
