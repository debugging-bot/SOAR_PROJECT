# TeamMate — 창구 거치형 한국수어 번역 시스템

웹캠 앞에서 한국수어 단어를 하면, 화면에 큰 글씨가 뜨고 스피커로 한국어 음성이 나옵니다.

---

## 처음 한 번만 하는 것 (1주차, 전원)

### 1. 파이썬 3.11 설치
[python.org/downloads](https://www.python.org/downloads/) 에서 **3.11** 설치.
설치 화면에서 **Add Python to PATH** 체크박스를 반드시 켤 것.
3.13 이상은 MediaPipe 설치 파일이 아직 없어서 안 됩니다.

### 2. 코드 받고 가상환경 만들기

```bash
git clone <우리 저장소 주소>
cd teammate

python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux
```

프롬프트 앞에 `(venv)` 가 보이면 성공입니다.
**터미널을 새로 열 때마다 activate를 다시 해야 합니다.**

### 3. 라이브러리 설치

```bash
pip install -r requirements.txt
```

tensorflow가 커서 5~15분 걸립니다. 실패하면 `pip install tensorflow-cpu` 로 대체하세요.

### 4. MediaPipe 모델 파일 받기

```bash
python setup/download_models.py
```

### 5. 확인

```bash
python setup/check_env.py     # 전부 [OK] 여야 함
python tests/smoke_test.py    # 실패 0개여야 함
```

여기까지 통과하면 1주차 완료 기준을 만족한 것입니다.

---

## 매일 쓰는 명령어

| 하려는 일 | 명령어 |
|---|---|
| 환경 점검 | `python setup/check_env.py` |
| 코드 점검 (카메라 없이) | `python tests/smoke_test.py` |
| 단어 목록 보기 | `python src/labels.py` |
| 데이터 촬영 | `python src/collect.py --word pain --person kim --speed normal --count 30` |
| 데이터 현황 | `python src/dataset.py` |
| 학습 | `python src/train.py` |
| 정직한 평가 | `python src/train.py --holdout jung` |
| 혼동행렬 보기 | `python src/evaluate.py` |
| 실시간 인식 (콘솔) | `python src/realtime.py` |
| 전체 시스템 (브라우저) | `python src/server.py` → http://localhost:8000 |
| 화면만 먼저 보기 | `python src/server.py --demo` |

---

## 폴더 구조

```
teammate/
├─ labels.json           인식할 단어 목록          (조원 1이 관리)
├─ requirements.txt
├─ setup/
│  ├─ check_env.py       내 컴퓨터 점검
│  └─ download_models.py MediaPipe 모델 내려받기
├─ src/
│  ├─ config.py          팀 공통 상수 — 함부로 고치지 말 것
│  ├─ sampler.py         프레임 샘플링 (촬영/실시간 공통)  ★건드리면 정확도 깨짐
│  ├─ labels.py          단어 목록 읽기
│  ├─ landmarker.py      MediaPipe 감싸기        (조원 2)
│  ├─ features.py        좌표 -> 146차원 특징     (조원 2)  ★핵심
│  ├─ augment.py         데이터 증강             (조원 2)
│  ├─ collect.py         촬영 도구               (조원 1, 2)
│  ├─ dataset.py         데이터 로드/분할        (조원 3)
│  ├─ train.py           모델 학습               (조원 3)
│  ├─ evaluate.py        혼동행렬 평가           (조원 3)
│  ├─ confirm.py         인식 확정 로직          (조원 4)
│  ├─ realtime.py        실시간 인식(콘솔)       (조원 4)
│  └─ server.py          FastAPI + WebSocket     (조원 4)
├─ web/index.html        고령 친화 화면 + TTS     (조원 5)
├─ tests/smoke_test.py   카메라 없이 코드 점검
├─ data/                 학습 데이터 (.npy) — git에 올리지 않음
└─ models/               .task, .keras — git에 올리지 않음
```

---

## 자주 나는 오류

| 증상 | 해결 |
|---|---|
| `ModuleNotFoundError: No module named 'cv2'` | `venv` activate를 안 한 것. 다시 activate 후 설치 |
| `모델 파일이 없습니다` | `python setup/download_models.py` |
| `웹캠을 열 수 없습니다` | 줌·팀즈 끄기 / 노트북 카메라 셔터 확인 / `src/config.py`의 `CAM_INDEX`를 1, 2로 바꿔 보기 |
| 촬영 시간이 짧게 느껴짐 | 녹화 창은 `src/config.py`의 `SEQ_LEN x FRAME_STRIDE / 30`초입니다. 현재 30 x 2 / 30 = 2.0초. 값을 바꾸면 이미 찍은 데이터는 쓸 수 없습니다 |
| 화면 글씨가 네모(□)로 보임 | OpenCV는 한글을 못 그림. 정상이며, 브라우저 화면(`server.py`)에서는 정상 출력됨 |
| 학습은 잘됐는데 실시간에서만 못 맞춤 | 학습과 실시간이 같은 `features.build_feature_vector`를 쓰는지 확인 (가장 흔한 원인) |
| `pip install mediapipe` 실패 | 파이썬 3.13을 쓰고 있을 가능성. `python --version`으로 확인하고 3.11로 다시 |
| Colab에서 학습한 모델이 안 열림 | Colab과 내 PC의 tensorflow 버전이 다름. `pip install tensorflow==<Colab 버전>` |
| 음성이 안 나옴 | 브라우저는 사용자가 한 번 클릭해야 소리를 냄. 첫 화면의 **시작하기** 버튼을 눌렀는지 확인 |

---

## 개발 규칙

- `main` 브랜치에 직접 push 금지. `feature/이름` 에서 작업 → `dev` 로 합치기
- `src/config.py`의 값은 3주차 확정 이후 전원 합의 없이 수정 금지
- `labels.json`의 `id` 순서는 절대 바꾸지 않는다 (모델이 전부 틀린 답을 냄)
- 프레임 건너뛰기는 `src/sampler.py`의 `FrameSampler`만 쓴다. 촬영·실시간이 달라지면 "학습은 잘되는데 실시간에서만 못 맞추는" 문제가 생긴다
- `data/`, `models/`, `venv/` 는 `.gitignore`에 있으므로 올라가지 않는다
- 코드를 고친 뒤에는 push 전에 `python tests/smoke_test.py` 를 돌린다
