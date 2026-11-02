"""
overlay.py — 인식에 쓰는 점을 카메라 화면에 그린다. 시험·디버그용. (조원 4)

전체 점은 realtime.py에서만 그린다 (d 키로 끄고 켬).
server.py는 full=False로 인식에 쓰는 손만 초록 점으로 그린다.

색 구분 (OpenCV는 BGR 순서)
    초록 점        인식에 쓰는 주 사용자의 손 관절 21개
    빨간 점        버린 손 (다른 사람 손, 가이드 프레임 밖 손)
    하늘색 큰 점   주 사용자 상체 중 특징에 쓰는 4점 (양 어깨, 양 팔꿈치)
    분홍 점·타원   주 사용자 손목(Pose 15·16)과, 손을 그 사람 것으로 받아 주는 범위
    노란 작은 점   주 사용자 상체의 나머지 점
    회색 점        다른 사람의 상체
    흰 사각형      가이드 프레임
"""

import cv2

from config import POSE_USED
from person import L_WRIST, R_WRIST, WRIST_MATCH, _body

GREEN = (0, 220, 120)
RED = (0, 0, 255)
CYAN = (255, 200, 0)
MAGENTA = (255, 0, 255)
YELLOW = (0, 220, 255)
GRAY = (150, 150, 150)
WHITE = (255, 255, 255)

LEGEND = "green=used hand  red=ignored  cyan=pose feature  pink=wrist zone  gray=other person"


def _pt(p, w, h):
    return int(p[0] * w), int(p[1] * h)


def draw_overlay(frame, used_hands, selector=None, full=True):
    """
    used_hands : 인식에 쓰는 손 [(21,3), ...]
    selector   : person.PrimaryUserSelector (없으면 손만 그린다)
    full       : False면 예전처럼 사용하는 손만 초록 점으로 그린다
    """
    h, w = frame.shape[:2]
    if full and selector is not None:
        b = selector.box
        cv2.rectangle(frame, (int(b[0] * w), int(b[1] * h)),
                      (int(b[2] * w) - 1, int(b[3] * h) - 1), WHITE, 1)

        for pose in selector.poses:
            main = pose is selector.primary
            for j, p in enumerate(pose):
                if main and j in POSE_USED:
                    cv2.circle(frame, _pt(p, w, h), 6, CYAN, -1)
                elif main and j in (L_WRIST, R_WRIST):
                    cv2.circle(frame, _pt(p, w, h), 6, MAGENTA, -1)
                else:
                    cv2.circle(frame, _pt(p, w, h), 2, YELLOW if main else GRAY, -1)
            if main:
                # 거리는 화면 비율(0~1) 기준이라 픽셀로는 타원이 된다
                r = WRIST_MATCH * _body(pose)[1]
                for j in (L_WRIST, R_WRIST):
                    cv2.ellipse(frame, _pt(pose[j], w, h), (int(r * w), int(r * h)),
                                0, 0, 360, MAGENTA, 1)

        for hand in selector.rejected:
            for p in hand:
                cv2.circle(frame, _pt(p, w, h), 3, RED, -1)

    for hand in used_hands:
        for p in hand:
            cv2.circle(frame, _pt(p, w, h), 3, GREEN, -1)
    return frame
