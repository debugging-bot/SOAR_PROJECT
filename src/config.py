"""
config.py — 팀 전체가 공유하는 상수.

이 파일의 값은 5장 '데이터 규격'과 1:1로 대응한다.
여기 값을 바꾸면 이미 모은 데이터와 학습된 모델이 전부 무효가 되므로,
3주차에 확정한 뒤에는 전원 합의 없이 절대 수정하지 않는다.
"""

from pathlib import Path

# ---------------------------------------------------------------- 경로
# 이 파일(src/config.py)의 부모의 부모 = 프로젝트 루트(teammate/)
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"            # 학습 데이터(.npy) 저장 위치
MODEL_DIR = ROOT / "models"         # mediapipe .task 파일과 학습된 모델
LABELS_PATH = ROOT / "labels.json"  # 클래스 번호 <-> 한국어 단어

HAND_TASK = MODEL_DIR / "hand_landmarker.task"
POSE_TASK = MODEL_DIR / "pose_landmarker_lite.task"
KERAS_MODEL = MODEL_DIR / "sign_model.keras"

# ---------------------------------------------------------------- 데이터 규격
SEQ_LEN = 30          # 한 시퀀스 = 30프레임 (약 1초)
FEATURE_DIM = 146     # 프레임당 특징 차원

HAND_SHAPE_DIM = 63   # 21관절 x 3(xyz), 손목 기준 상대좌표 / 손 크기
HAND_POS_DIM = 3      # 손목이 어깨 중심에서 떨어진 정도 / 어깨 너비
HAND_FLAG_DIM = 1     # 이 손이 검출되었는가 (1 또는 0)
HAND_DIM = HAND_SHAPE_DIM + HAND_POS_DIM + HAND_FLAG_DIM   # = 67
POSE_DIM = 12         # 양 어깨 + 양 팔꿈치 4점 x 3(xyz)

assert HAND_DIM * 2 + POSE_DIM == FEATURE_DIM, "146차원 구성이 맞지 않습니다"

# MediaPipe Pose(33점)에서 우리가 쓰는 관절 번호
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_ELBOW = 13
POSE_RIGHT_ELBOW = 14
POSE_USED = [POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER,
             POSE_LEFT_ELBOW, POSE_RIGHT_ELBOW]

# 상체가 검출되지 않은 프레임에서 쓰는 대체값.
# 학습과 실시간에서 반드시 같은 값을 써야 하므로 상수로 못 박는다.
FALLBACK_CENTER = (0.5, 0.5, 0.0)   # 화면 정중앙
FALLBACK_WIDTH = 0.25               # 어깨 너비를 화면 폭의 25%로 가정
EPS = 1e-6

# ---------------------------------------------------------------- 카메라
CAM_INDEX = 0         # 웹캠이 여러 개면 0, 1, 2 로 바꿔 본다
CAM_WIDTH = 640
CAM_HEIGHT = 480
MIRROR = True         # 화면을 거울처럼 좌우 반전할지. 반드시 True로 고정.

# ---------------------------------------------------------------- 확정 로직
CONF_THRESHOLD = 0.85   # 이 확률 미만이면 무시
CONSECUTIVE_N = 5       # 같은 단어가 연속 몇 번 나와야 확정인가
COOLDOWN_SEC = 1.5      # 확정 직후 이 시간 동안은 새로 확정하지 않음
PREDICT_EVERY = 3       # 몇 프레임마다 한 번 예측할 것인가 (부하 조절)

# ---------------------------------------------------------------- 학습
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 1e-3
VAL_RATIO = 0.2
RANDOM_SEED = 42
