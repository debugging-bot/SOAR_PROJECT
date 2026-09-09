"""
evaluate.py — 학습된 모델이 어떤 단어를 헷갈리는지 본다. (조원 3)

실행
    python src/evaluate.py
    python src/evaluate.py --holdout jung

정확도 숫자 하나만 보지 말고 반드시 혼동행렬을 볼 것.
"어느 단어가 문제인지"를 알아야 다음 주에 무엇을 고칠지 정할 수 있다.
결과 그림은 models/confusion_matrix.png 로 저장된다.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import KERAS_MODEL, MODEL_DIR
from dataset import load_dataset, split_by_person, split_random
from labels import load_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default=None)
    args = ap.parse_args()

    if not KERAS_MODEL.exists():
        print(f"모델 파일이 없습니다: {KERAS_MODEL}\n먼저 python src/train.py 를 실행하세요.")
        return 1

    from tensorflow import keras
    from sklearn.metrics import classification_report, confusion_matrix

    words = load_labels()
    names = [w["ko"] for w in words]

    X, y, meta = load_dataset()
    if args.holdout:
        _, Xva, _, yva = split_by_person(X, y, meta, args.holdout)
    else:
        _, Xva, _, yva = split_random(X, y)

    model = keras.models.load_model(KERAS_MODEL)
    prob = model.predict(Xva, verbose=0)
    pred = prob.argmax(axis=1)

    acc = float((pred == yva).mean())
    print(f"\n전체 정확도: {acc*100:.1f}%  (검증 {len(yva)}개)\n")
    print(classification_report(yva, pred, labels=list(range(len(names))),
                                target_names=names, zero_division=0))

    cm = confusion_matrix(yva, pred, labels=list(range(len(names))))
    print("혼동행렬 (행=정답, 열=예측)")
    w = max(len(n) for n in names) + 1
    print(" " * w + "".join(f"{i:>5d}" for i in range(len(names))))
    for i, row in enumerate(cm):
        print(f"{names[i]:<{w}}" + "".join(f"{v:>5d}" for v in row))

    print("\n[가장 많이 헷갈리는 쌍]")
    pairs = []
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j and cm[i][j] > 0:
                pairs.append((cm[i][j], names[i], names[j]))
    for n, a, b in sorted(pairs, reverse=True)[:5]:
        print(f"  '{a}' -> '{b}' 로 {n}번 잘못 인식")
    if not pairs:
        print("  없음")

    # 신뢰도 임계값을 얼마로 둘지 판단하는 근거
    conf = prob.max(axis=1)
    right, wrong = conf[pred == yva], conf[pred != yva]
    print(f"\n맞힌 경우 평균 확신도 {right.mean():.2f}")
    if len(wrong):
        print(f"틀린 경우 평균 확신도 {wrong.mean():.2f}")
        for t in (0.7, 0.8, 0.85, 0.9, 0.95):
            keep = conf >= t
            if keep.sum():
                p = (pred[keep] == yva[keep]).mean()
                print(f"  임계값 {t:.2f} -> 통과 {keep.mean()*100:4.1f}%, "
                      f"그중 정확도 {p*100:5.1f}%")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(range(len(names)))
        ax.set_yticklabels(range(len(names)))
        ax.set_xlabel("predicted (class id)")
        ax.set_ylabel("true (class id)")
        for i in range(len(names)):
            for j in range(len(names)):
                if cm[i][j]:
                    ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=8)
        out = MODEL_DIR / "confusion_matrix.png"
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        print(f"\n그림 저장: {out}")
    except ImportError:
        print("\n(matplotlib이 없어 그림은 건너뜁니다: pip install matplotlib)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
