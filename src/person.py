"""
person.py — 화면에 여러 사람이 있을 때 '내 앞의 한 사람'의 손만 남긴다. (조원 4)

10주차 중간발표 피드백 3
  화면 전체에서 손을 찾으면 옆·뒤 사람의 손이 섞여 특징이 오염되거나 다른 단어가 확정된다.

규칙
  1) 어깨 중심이 가이드 프레임 안에 있는 사람만 후보로 본다
  2) 주 사용자: 처음에는 어깨 폭이 가장 크고(카메라에 가장 가까움) 화면 중앙에 가까운 사람.
     한 번 정하면 그 사람이 프레임 안에 있는 동안 바꾸지 않는다 (더 가까운 사람이 와도 유지)
  3) 손은 '손목이 가장 가까운 사람'(Pose 15·16번)에게 배정하고, 주 사용자의 손만 남긴다.
     뒤에 선 사람은 몸이 가려져 상체가 검출되지 않는 경우가 많으므로, 다른 사람이 검출되지
     않아도 주 사용자 손목에서 '어깨 폭 x WRIST_MATCH'보다 먼 손은 버린다
  4) 가이드 프레임 밖의 손은 버린다
  5) 후보가 없으면(사용자가 프레임을 벗어남) 손을 모두 버린다 -> 대기, 다시 들어오면 새로 고른다

사람이 1명이고 손이 가이드 안에 있으면 결과가 이전과 같다(손 최대 2개, 그 사람의 상체).
특징 벡터 규격(146차원)은 그대로라 재학습이 필요 없다.
좌표는 거울 반전 후의 화면 비율(0~1)이라 브라우저에 보이는 영상과 같은 기준이다.
"""

import numpy as np

# 가이드 프레임 (x0, y0, x1, y1), 화면 비율 0~1.
# 브라우저는 /api/status의 guide_box 값으로 같은 위치에 프레임을 그린다.
GUIDE_BOX = (0.10, 0.0, 0.90, 1.0)
MAX_PEOPLE = 3          # 상체를 최대 몇 명까지 찾을 것인가
MAX_HANDS_SCAN = 4      # 다른 사람 손이 자리를 차지해도 사용자 손을 놓치지 않도록 넉넉히 찾는다
KEEP_HANDS = 2          # 특징 벡터에 들어가는 손 개수 (146차원 규격)
TRACK_DIST = 0.5        # 직전 주 사용자와 어깨 중심이 '어깨 폭 x 이 값' 안이면 같은 사람으로 본다
WRIST_MATCH = 0.7       # 손의 손목이 주 사용자 손목(Pose 15·16)에서 '어깨 폭 x 이 값' 안이어야 그 사람 손
#   손 검출기의 손목과 상체 검출기의 손목은 같은 점이라 보통 0.1~0.3 안에 들어온다.
#   여유를 두되, 어깨 너머로 보이는 뒤 사람 손은 걸러지도록 정했다 (11주차 실측으로 조정)

L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 11, 12, 15, 16


def _inside(pt, box) -> bool:
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def _body(pose):
    """(어깨 중심 xy, 어깨 폭)"""
    center = (pose[L_SHOULDER, :2] + pose[R_SHOULDER, :2]) / 2
    width = float(np.linalg.norm(pose[L_SHOULDER, :2] - pose[R_SHOULDER, :2]))
    return center, max(width, 1e-6)


class PrimaryUserSelector:
    """
    사용법
        sel = PrimaryUserSelector()
        hands, poses = landmarker.process_all(frame)
        hands, pose = sel.select(hands, poses)       # 주 사용자의 손과 상체만
        vec = build_feature_vector(hands, pose)
    """

    def __init__(self, guide_box=GUIDE_BOX, keep_hands=KEEP_HANDS):
        self.box = guide_box
        self.keep = keep_hands
        self.prev = None            # 직전 주 사용자의 어깨 중심
        # 직전 프레임 기록 (화면 확인용, overlay.py)
        self.rejected = []          # 버린 손
        self.poses = []             # 검출된 모든 사람의 상체
        self.primary = None         # 주 사용자의 상체 (없으면 None)
        self.n_people = 0           # 검출된 사람 수

    def reset(self):
        self.prev = None
        self.rejected, self.poses, self.primary = [], [], None

    def select(self, hands, poses):
        self.n_people = len(poses)
        self.poses = list(poses)
        kept, self.rejected = self._split(hands, poses)
        self.primary = kept[1]
        return kept

    def _split(self, hands, poses):
        """((주 사용자 손, 상체), 버린 손)"""
        inside = [h for h in hands if _inside(h[0], self.box)]
        out = [h for h in hands if not _inside(h[0], self.box)]
        if not poses:
            # 상체를 못 찾으면(카메라에 너무 가까움 등) 예전처럼 손만 쓴다
            return (inside[:self.keep], None), out + inside[self.keep:]

        bodies = [_body(p) for p in poses]
        cand = [i for i, (c, _w) in enumerate(bodies) if _inside(c, self.box)]
        if not cand:
            self.prev = None
            return ([], None), hands

        idx = self._pick(cand, bodies)
        self.prev = bodies[idx][0]
        width = bodies[idx][1]
        mine = [h for h in inside
                if self._owner(h, poses, bodies) == idx
                and self._wrist_dist(h, poses[idx]) <= WRIST_MATCH * width]
        rest = [h for h in hands if not any(h is m for m in mine[:self.keep])]
        return (mine[:self.keep], poses[idx]), rest

    @staticmethod
    def _wrist_dist(hand, pose):
        """손의 손목과 그 사람 손목(Pose 15·16) 중 가까운 쪽까지의 거리 (화면 비율)"""
        wrist = hand[0, :2]
        return float(min(np.linalg.norm(wrist - pose[L_WRIST, :2]),
                         np.linalg.norm(wrist - pose[R_WRIST, :2])))

    def _pick(self, cand, bodies):
        if self.prev is not None:
            near = min(cand, key=lambda i: np.linalg.norm(bodies[i][0] - self.prev))
            c, w = bodies[near]
            if np.linalg.norm(c - self.prev) <= TRACK_DIST * w:
                return near
        cx = (self.box[0] + self.box[2]) / 2
        return max(cand, key=lambda i: bodies[i][1] - 0.5 * abs(bodies[i][0][0] - cx))

    @staticmethod
    def _owner(hand, poses, bodies):
        """손목이 가장 가까운 사람의 번호. 거리는 그 사람 어깨 폭으로 나눠 비교한다."""
        wrist = hand[0, :2]
        dists = [min(np.linalg.norm(wrist - p[L_WRIST, :2]),
                     np.linalg.norm(wrist - p[R_WRIST, :2])) / bodies[i][1]
                 for i, p in enumerate(poses)]
        return int(np.argmin(dists))
