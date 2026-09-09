"""
features.py — 랜드마크 좌표를 146차원 특징 벡터로 바꾼다.

이 파일에는 MediaPipe가 등장하지 않는다. 순수한 숫자 계산만 있어서
카메라 없이도 테스트할 수 있다(tests/smoke_test.py).

**가장 중요한 규칙**
학습할 때와 실시간으로 인식할 때가 반드시 이 파일의 같은 함수를 호출해야 한다.
여기가 어긋나면 "학습 정확도는 90%인데 실시간에서는 하나도 못 맞추는" 문제가 생긴다.
"""

import numpy as np

from config import (
    FEATURE_DIM, HAND_DIM, POSE_USED,
    FALLBACK_CENTER, FALLBACK_WIDTH, EPS,
)

# MediaPipe 손 랜드마크 번호
WRIST = 0        # 손목
MIDDLE_MCP = 9   # 중지 밑동


def normalize_hand_shape(hand: np.ndarray) -> np.ndarray:
    """
    손 '모양'만 남긴다. 손이 화면 어디에 있든, 카메라에서 얼마나 멀든 같은 값이 나온다.

    hand : (21, 3) 손 랜드마크 원본 좌표
    return : (63,)

    방법
      1) 손목을 원점(0,0,0)으로 옮긴다  -> 위치 정보 제거
      2) '손목~중지 밑동' 길이로 전부 나눈다 -> 크기(거리) 정보 제거
    """
    wrist = hand[WRIST]
    rel = hand - wrist
    scale = float(np.linalg.norm(hand[MIDDLE_MCP] - wrist))
    if scale < EPS:
        scale = EPS
    return (rel / scale).astype(np.float32).flatten()


def hand_position(wrist: np.ndarray,
                  center: np.ndarray,
                  width: float) -> np.ndarray:
    """
    손이 '몸의 어디쯤'에 있는지. 얼굴 앞인지 가슴 앞인지가 수어의 뜻을 바꾸므로
    이 정보를 버리면 안 된다.

    wrist  : (3,) 손목 좌표
    center : (3,) 양 어깨의 중점
    width  : 어깨 너비 (스칼라)
    return : (3,)
    """
    if width < EPS:
        width = EPS
    return ((wrist - center) / width).astype(np.float32)


def pose_block(pose: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, float]:
    """
    상체 12차원 + 어깨 중심 + 어깨 너비를 계산한다.

    pose : (33, 3) 또는 None
    return : (12,) 상체 특징, (3,) 어깨 중심, 어깨 너비
    """
    if pose is None:
        center = np.array(FALLBACK_CENTER, dtype=np.float32)
        return np.zeros(12, dtype=np.float32), center, FALLBACK_WIDTH

    l_sh, r_sh = pose[POSE_USED[0]], pose[POSE_USED[1]]
    center = ((l_sh + r_sh) / 2.0).astype(np.float32)
    width = float(np.linalg.norm(l_sh[:2] - r_sh[:2]))
    if width < EPS:
        width = FALLBACK_WIDTH

    pts = pose[POSE_USED]                      # (4, 3)
    block = ((pts - center) / width).astype(np.float32).flatten()   # (12,)
    return block, center, width


def build_feature_vector(hands: list[np.ndarray],
                         pose: np.ndarray | None) -> np.ndarray:
    """
    한 프레임 -> (146,) 특징 벡터.

    hands : 검출된 손 좌표 리스트. 각 원소는 (21, 3). 0개, 1개, 2개 모두 가능.
    pose  : (33, 3) 또는 None

    슬롯 배정 규칙 (매우 중요)
      MediaPipe가 알려 주는 '왼손/오른손' 라벨은 화면을 좌우 반전했는지에 따라
      뒤집혀서 신뢰하기 어렵다. 그래서 우리는 라벨을 쓰지 않고,
      손목의 x좌표가 어깨 중심보다 왼쪽이면 슬롯0, 오른쪽이면 슬롯1에 넣는다.
      학습과 실시간이 같은 규칙을 쓰므로 일관성이 보장된다.

    구성
      [ 슬롯0 손 67 | 슬롯1 손 67 | 상체 12 ] = 146
      손 67 = 모양 63 + 위치 3 + 검출여부 1
      검출되지 않은 손은 67개 값을 전부 0으로 채운다(검출여부도 0).
    """
    pose_feat, center, width = pose_block(pose)

    slots: list[np.ndarray | None] = [None, None]
    for hand in hands:
        if hand is None or len(hand) < 21:
            continue
        wrist = hand[WRIST]
        idx = 0 if wrist[0] < center[0] else 1
        if slots[idx] is None:
            slots[idx] = hand
        else:
            other = 1 - idx
            if slots[other] is None:      # 두 손이 같은 쪽에 몰린 경우
                slots[other] = hand

    parts = []
    for hand in slots:
        if hand is None:
            parts.append(np.zeros(HAND_DIM, dtype=np.float32))
        else:
            shape = normalize_hand_shape(hand)                 # (63,)
            pos = hand_position(hand[WRIST], center, width)    # (3,)
            flag = np.array([1.0], dtype=np.float32)           # (1,)
            parts.append(np.concatenate([shape, pos, flag]))   # (67,)

    parts.append(pose_feat)                                    # (12,)
    vec = np.concatenate(parts).astype(np.float32)

    assert vec.shape == (FEATURE_DIM,), f"차원 오류: {vec.shape}"
    return vec


def empty_feature_vector() -> np.ndarray:
    """손도 몸도 안 잡힌 프레임. 버리지 말고 0으로 채워 넣는다."""
    return np.zeros(FEATURE_DIM, dtype=np.float32)


def hand_detected_count(vec: np.ndarray) -> int:
    """특징 벡터 하나에서 손이 몇 개 잡혔는지. 품질 점검용."""
    flag0 = vec[HAND_DIM - 1]
    flag1 = vec[HAND_DIM * 2 - 1]
    return int(flag0 > 0.5) + int(flag1 > 0.5)
