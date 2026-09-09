"""
train.py — LSTM 모델을 학습한다. (조원 3)

실행 (프로젝트 루트 teammate/ 에서)
    python src/train.py                       # 기본: 무작위 8:2 분할
    python src/train.py --holdout jung        # 정세영 촬영본을 통째로 검증용으로
    python src/train.py --aug 5 --epochs 300  # 증강 5배, 더 오래

Colab에서 돌리려면
    1) 구글 드라이브에 teammate 폴더를 통째로 올린다
    2) 런타임 > 런타임 유형 변경 > 하드웨어 가속기: GPU
    3) 노트북 셀에서
         from google.colab import drive; drive.mount('/content/drive')
         %cd /content/drive/MyDrive/teammate
         !pip install -q mediapipe
         !python src/train.py
    4) 학습이 끝나면 models/sign_model.keras 를 내려받아 각자 PC의 models/에 넣는다
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from augment import augment_dataset
from config import (BATCH_SIZE, EPOCHS, FEATURE_DIM, KERAS_MODEL, LEARNING_RATE,
                    MODEL_DIR, RANDOM_SEED, SEQ_LEN)
from dataset import load_dataset, split_by_person, split_random
from labels import load_labels


def build_model(num_classes: int):
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(SEQ_LEN, FEATURE_DIM)),   # 30프레임 x 146차원
        layers.Masking(mask_value=0.0),               # 전부 0인 프레임은 건너뛴다
        layers.LSTM(96, return_sequences=True),       # 순서를 기억하며 읽는다
        layers.Dropout(0.3),                          # 통째로 외우는 것을 막는다
        layers.LSTM(64),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),  # 단어별 확률, 합=1
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", type=int, default=3, help="증강 배수 (0이면 증강 안 함)")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    ap.add_argument("--holdout", default=None, help="검증용으로 뺄 촬영자 이름")
    args = ap.parse_args()

    np.random.seed(RANDOM_SEED)
    import tensorflow as tf
    tf.random.set_seed(RANDOM_SEED)
    print(f"TensorFlow {tf.__version__}")

    words = load_labels()
    num_classes = len(words)

    X, y, meta = load_dataset()
    if args.holdout:
        Xtr, Xva, ytr, yva = split_by_person(X, y, meta, args.holdout)
        print(f"검증: '{args.holdout}' 촬영본 전체 ({len(yva)}개) — 처음 보는 사람 기준")
    else:
        Xtr, Xva, ytr, yva = split_random(X, y)
        print(f"검증: 무작위 {len(yva)}개")

    # 증강은 학습용에만 적용한다. 검증용에 넣으면 성능이 부풀려진다.
    if args.aug > 0:
        before = len(Xtr)
        Xtr, ytr = augment_dataset(Xtr, ytr, n_aug=args.aug, seed=RANDOM_SEED)
        print(f"증강: {before} -> {len(Xtr)}개")

    print(f"학습 {Xtr.shape}, 검증 {Xva.shape}, 클래스 {num_classes}개")

    # 단어별 개수가 다를 때 적은 단어를 더 중요하게 본다
    counts = np.bincount(ytr, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    class_weight = {i: float(counts.sum() / (num_classes * counts[i]))
                    for i in range(num_classes)}

    from tensorflow import keras
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model = build_model(num_classes)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=25,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=10, min_lr=1e-5, verbose=1),
        keras.callbacks.ModelCheckpoint(str(KERAS_MODEL), monitor="val_accuracy",
                                        save_best_only=True, verbose=0),
    ]

    hist = model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=args.epochs,
        batch_size=args.batch,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    best = float(max(hist.history["val_accuracy"]))
    model.save(KERAS_MODEL)
    print(f"\n최고 검증 정확도: {best*100:.1f}%")
    print(f"모델 저장: {KERAS_MODEL}")

    (MODEL_DIR / "train_report.json").write_text(json.dumps({
        "tensorflow": tf.__version__,
        "num_classes": num_classes,
        "train_size": int(len(ytr)),
        "val_size": int(len(yva)),
        "augment": args.aug,
        "holdout": args.holdout,
        "best_val_accuracy": best,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    if best < 0.80:
        print("\n[주의] 80% 미만입니다. 다음 주로 넘어가지 말고 아래를 확인하세요.")
        print("  1) python src/dataset.py 로 단어별 개수가 150개 이상인지")
        print("  2) python src/evaluate.py 로 어떤 단어끼리 헷갈리는지")
        print("  3) 헷갈리는 두 단어는 동작이 실제로 비슷한 것이므로 하나를 교체")
    if best > 0.99:
        print("\n[주의] 99% 이상이면 의심해야 합니다.")
        print("  같은 촬영본이 학습과 검증에 함께 들어갔을 수 있습니다.")
        print("  python src/train.py --holdout <이름> 으로 다시 확인하세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
