"""
check_env.py — 내 컴퓨터가 이 프로젝트를 돌릴 수 있는지 하나씩 확인한다.

실행 (프로젝트 루트 teammate/ 에서):
    python setup/check_env.py

전부 [OK]가 나와야 1주차 완료 기준을 만족한 것이다.
[FAIL]이 나오면 바로 아래에 적힌 해결 방법을 그대로 따라 하면 된다.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

results = []


def check(name, fn, fix=""):
    try:
        detail = fn()
        print(f"[ OK ] {name}  {detail or ''}")
        results.append(True)
    except Exception as e:                          # noqa: BLE001
        print(f"[FAIL] {name}")
        print(f"       원인: {e}")
        if fix:
            for line in fix.strip().split("\n"):
                print(f"       해결: {line}")
        results.append(False)


# ------------------------------------------------------------- 1. 파이썬
def c_python():
    v = sys.version_info
    if v.major != 3 or not (9 <= v.minor <= 12):
        raise RuntimeError(f"파이썬 {v.major}.{v.minor} 은(는) MediaPipe 지원 밖입니다")
    return f"(python {v.major}.{v.minor}.{v.micro})"


check("파이썬 버전", c_python,
      "python.org에서 3.11을 설치하고 가상환경을 다시 만드세요.\n"
      "3.13 이상은 MediaPipe 설치 파일이 아직 없습니다.")


# ------------------------------------------------------------- 2. 라이브러리
def c_numpy():
    import numpy
    return f"(numpy {numpy.__version__})"


def c_cv2():
    import cv2
    return f"(opencv {cv2.__version__})"


def c_mediapipe():
    import mediapipe
    return f"(mediapipe {mediapipe.__version__})"


def c_tf():
    import tensorflow as tf
    return f"(tensorflow {tf.__version__})"


def c_sklearn():
    import sklearn
    return f"(scikit-learn {sklearn.__version__})"


check("numpy", c_numpy, "pip install numpy")
check("opencv", c_cv2, "pip install opencv-python")
check("mediapipe", c_mediapipe, "pip install mediapipe")
check("tensorflow", c_tf,
      "pip install tensorflow\n"
      "설치가 오래 걸립니다(약 600MB). 실패하면 pip install tensorflow-cpu 를 시도하세요.")
check("scikit-learn", c_sklearn, "pip install scikit-learn")


# ------------------------------------------------------------- 3. 모델 파일
def c_models():
    from config import HAND_TASK, POSE_TASK
    missing = [p.name for p in (HAND_TASK, POSE_TASK) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{', '.join(missing)} 없음")
    return "(hand + pose .task)"


check("MediaPipe 모델 파일", c_models, "python setup/download_models.py")


# ------------------------------------------------------------- 4. 우리 코드
def c_features():
    import numpy as np
    from features import build_feature_vector
    hands = [np.random.rand(21, 3).astype(np.float32)]
    pose = np.random.rand(33, 3).astype(np.float32)
    vec = build_feature_vector(hands, pose)
    if vec.shape != (146,):
        raise RuntimeError(f"shape {vec.shape}")
    return "(146차원 생성 확인)"


check("특징 추출 코드", c_features,
      "src 폴더 안에서 실행 중인지, config.py와 features.py가 있는지 확인하세요.")


# ------------------------------------------------------------- 5. 웹캠
def c_camera():
    import cv2
    from config import CAM_INDEX
    cap = cv2.VideoCapture(CAM_INDEX)
    ok = cap.isOpened()
    if ok:
        ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("카메라에서 프레임을 읽지 못했습니다")
    return f"(해상도 {frame.shape[1]}x{frame.shape[0]})"


check("웹캠", c_camera,
      "줌/팀즈 등 카메라를 쓰는 프로그램을 모두 끄세요.\n"
      "노트북 카메라 셔터가 닫혀 있는지 확인하세요.\n"
      "외장 웹캠이면 src/config.py의 CAM_INDEX를 1 또는 2로 바꿔 보세요.")


# ------------------------------------------------------------- 결과
print("-" * 60)
if all(results):
    print("전부 통과했습니다. 다음 단계로 진행하세요.")
    raise SystemExit(0)
print(f"{results.count(False)}개 항목이 실패했습니다. 위의 '해결'을 따라 하고 다시 실행하세요.")
raise SystemExit(1)
