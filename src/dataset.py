"""
dataset.py — data/ 폴더의 .npy들을 모아 학습용 배열로 만든다. (조원 3)

단독 실행하면 현재 모은 데이터의 통계를 보여 준다.
    python src/dataset.py
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_DIR, FEATURE_DIM, RANDOM_SEED, SEQ_LEN, VAL_RATIO
from labels import load_labels, slug_to_id


def load_dataset(data_dir: Path = DATA_DIR, verbose: bool = True):
    """
    return
        X      : (N, 30, 146) float32
        y      : (N,) int64          정답 클래스 번호
        meta   : list[dict]          {'person':..., 'speed':..., 'path':...}
    """
    words = load_labels()
    s2i = slug_to_id(words)

    X, y, meta = [], [], []
    bad = []

    for slug, cls in s2i.items():
        folder = data_dir / slug
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.npy")):
            arr = np.load(path)
            if arr.shape != (SEQ_LEN, FEATURE_DIM):
                bad.append((path, arr.shape))
                continue
            X.append(arr.astype(np.float32))
            y.append(cls)
            parts = path.stem.split("_")
            meta.append({
                "person": parts[0] if parts else "unknown",
                "speed": parts[1] if len(parts) > 1 else "unknown",
                "path": str(path),
            })

    if not X:
        raise RuntimeError(
            f"{data_dir} 안에 학습 데이터가 없습니다.\n"
            "먼저 python src/collect.py --word ... 로 촬영하세요."
        )

    X = np.stack(X)
    y = np.array(y, dtype=np.int64)

    if verbose:
        print(f"불러온 시퀀스 {len(y)}개, X.shape={X.shape}, y.shape={y.shape}")
        if bad:
            print(f"[경고] 규격이 안 맞아 제외한 파일 {len(bad)}개:")
            for p, s in bad[:5]:
                print(f"   {p.name}  shape={s}  (기대값 {(SEQ_LEN, FEATURE_DIM)})")
    return X, y, meta


def split_random(X, y, val_ratio=VAL_RATIO, seed=RANDOM_SEED):
    """단어별 비율을 유지하면서 무작위로 나눈다 (기본 방식)."""
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=val_ratio,
                            random_state=seed, stratify=y)


def split_by_person(X, y, meta, holdout: str):
    """
    특정 촬영자를 통째로 검증용으로 뺀다.
    '처음 보는 사람'에게도 통하는지를 보는 정직한 평가 방법이다.
    발표 때 이 숫자를 함께 제시하면 설득력이 크게 올라간다.
    """
    persons = np.array([m["person"] for m in meta])
    mask = persons == holdout
    if mask.sum() == 0:
        raise ValueError(f"'{holdout}' 촬영자의 데이터가 없습니다. "
                         f"있는 사람: {sorted(set(persons))}")
    return X[~mask], X[mask], y[~mask], y[mask]


def describe(X, y, meta):
    words = load_labels()
    id2ko = {w["id"]: w["ko"] for w in words}
    print("\n[단어별 개수]")
    for cls, n in sorted(Counter(y.tolist()).items()):
        speeds = Counter(m["speed"] for m, c in zip(meta, y) if c == cls)
        slow = speeds.get("slow", 0)
        mark = "  <-- 부족" if n < 150 or slow < n / 3 else ""
        print(f"  {cls:2d} {id2ko.get(cls, '?'):<8} 총 {n:4d}  (slow {slow:3d}){mark}")

    print("\n[촬영자별 개수]")
    for p, n in sorted(Counter(m["person"] for m in meta).items()):
        print(f"  {p:<10} {n:5d}")

    missing = [w["ko"] for w in words if w["id"] not in set(y.tolist())]
    if missing:
        print(f"\n[경고] 아직 한 장도 안 찍은 단어: {', '.join(missing)}")


if __name__ == "__main__":
    X, y, meta = load_dataset()
    describe(X, y, meta)
