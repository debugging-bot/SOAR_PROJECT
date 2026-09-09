"""
landmarker.py — MediaPipe를 감싸는 유일한 파일.

다른 파일은 MediaPipe를 직접 부르지 않고 이 파일의 Landmarker 클래스만 쓴다.
MediaPipe 버전이 바뀌어도 이 파일 하나만 고치면 된다.

MediaPipe는 최근 버전에서 옛날 방식(mp.solutions.hands)을 없애고
새 방식(mp.tasks.vision.HandLandmarker)으로 넘어갔다.
설치된 버전이 어느 쪽인지 팀원마다 다를 수 있으므로,
이 파일은 새 방식을 먼저 시도하고 안 되면 옛 방식으로 자동으로 넘어간다.
어느 쪽이든 바깥에서 쓰는 방법은 완전히 같다.
"""

import numpy as np

from config import CAM_HEIGHT, CAM_WIDTH, HAND_TASK, POSE_TASK


def _to_array(landmark_list, n: int) -> np.ndarray:
    """MediaPipe 랜드마크 객체 목록 -> (n, 3) numpy 배열"""
    arr = np.zeros((n, 3), dtype=np.float32)
    for i, lm in enumerate(landmark_list):
        if i >= n:
            break
        arr[i] = (lm.x, lm.y, lm.z)
    return arr


class Landmarker:
    """
    사용법
        lm = Landmarker()
        hands, pose = lm.process(frame_bgr)   # frame_bgr: OpenCV 이미지
        lm.close()

    반환값
        hands : [(21,3), ...]  최대 2개. 못 찾으면 빈 리스트.
        pose  : (33,3) 또는 None
    """

    def __init__(self, max_hands: int = 2):
        import mediapipe as mp
        self.mp = mp
        self.backend = None
        self._ts = 0            # VIDEO 모드는 타임스탬프가 항상 증가해야 한다
        self.max_hands = max_hands

        # ---------- 1순위: 새 방식(Tasks API) ----------
        try:
            self._init_tasks()
            self.backend = "tasks"
            print("[landmarker] MediaPipe Tasks API 사용")
            return
        except Exception as e:            # noqa: BLE001
            tasks_err = e

        # ---------- 2순위: 옛 방식(solutions) ----------
        try:
            self._init_legacy()
            self.backend = "legacy"
            print("[landmarker] MediaPipe solutions(구버전) API 사용")
            return
        except Exception as e:            # noqa: BLE001
            raise RuntimeError(
                "MediaPipe를 초기화하지 못했습니다.\n"
                f"  - Tasks API 실패: {tasks_err}\n"
                f"  - solutions API 실패: {e}\n"
                "해결: python setup/download_models.py 를 실행해 모델 파일을 받고,\n"
                "      python setup/check_env.py 로 설치 상태를 확인하세요."
            ) from e

    # ------------------------------------------------------------ 초기화
    def _init_tasks(self):
        mp = self.mp
        if not HAND_TASK.exists() or not POSE_TASK.exists():
            raise FileNotFoundError(
                f"모델 파일이 없습니다: {HAND_TASK.name}, {POSE_TASK.name} "
                "-> python setup/download_models.py 먼저 실행"
            )

        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        hand_opts = mp.tasks.vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_TASK)),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=self.max_hands,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        pose_opts = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(POSE_TASK)),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hand = mp.tasks.vision.HandLandmarker.create_from_options(hand_opts)
        self.pose = mp.tasks.vision.PoseLandmarker.create_from_options(pose_opts)

    def _init_legacy(self):
        mp = self.mp
        self.hand = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    # ------------------------------------------------------------ 처리
    def process(self, frame_bgr):
        """OpenCV BGR 이미지 한 장 -> (hands, pose)"""
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self.backend == "tasks":
            return self._process_tasks(rgb)
        return self._process_legacy(rgb)

    def _process_tasks(self, rgb):
        mp = self.mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._ts += 33                      # 30fps 가정, 항상 증가하기만 하면 된다

        hres = self.hand.detect_for_video(mp_image, self._ts)
        pres = self.pose.detect_for_video(mp_image, self._ts)

        hands = [_to_array(h, 21) for h in (hres.hand_landmarks or [])]
        pose = None
        if pres.pose_landmarks:
            pose = _to_array(pres.pose_landmarks[0], 33)
        return hands, pose

    def _process_legacy(self, rgb):
        rgb.flags.writeable = False
        hres = self.hand.process(rgb)
        pres = self.pose.process(rgb)
        rgb.flags.writeable = True

        hands = []
        if hres.multi_hand_landmarks:
            hands = [_to_array(h.landmark, 21) for h in hres.multi_hand_landmarks]
        pose = None
        if pres.pose_landmarks:
            pose = _to_array(pres.pose_landmarks.landmark, 33)
        return hands, pose

    # ------------------------------------------------------------ 정리
    def close(self):
        for obj in (getattr(self, "hand", None), getattr(self, "pose", None)):
            try:
                obj.close()
            except Exception:             # noqa: BLE001
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def open_camera():
    """
    웹캠을 연다. 열리지 않으면 원인을 알려 주고 종료한다.
    (윈도우에서 카메라가 느리게 열리는 문제를 피하려고 CAP_DSHOW를 먼저 시도한다)
    """
    import sys
    import cv2
    from config import CAM_INDEX

    backends = []
    if sys.platform.startswith("win"):
        backends.append(cv2.CAP_DSHOW)
    backends.append(cv2.CAP_ANY)

    for be in backends:
        cap = cv2.VideoCapture(CAM_INDEX, be)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
            return cap
        cap.release()

    raise RuntimeError(
        f"웹캠(index={CAM_INDEX})을 열 수 없습니다.\n"
        "확인할 것: 다른 프로그램(줌, 팀즈)이 카메라를 쓰고 있지 않은지,\n"
        "          노트북 카메라 셔터가 닫혀 있지 않은지,\n"
        "          config.py의 CAM_INDEX를 1이나 2로 바꿔 볼 것."
    )
