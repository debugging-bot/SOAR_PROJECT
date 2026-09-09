"""
labels.py — labels.json을 읽어 오는 도우미.

labels.json 형식 (조원 1이 관리)
{
  "words": [
    {"id": 0, "slug": "pain", "ko": "아프다", "display": "아파요"},
    ...
  ]
}

- id      : 모델의 출력 번호. **한 번 정하면 절대 바꾸지 않는다.**
            중간에 순서를 바꾸면 이미 학습한 모델이 전부 틀린 답을 낸다.
- slug    : 데이터 폴더 이름(영문). data/pain/ 처럼 쓰인다.
- ko      : 표준 단어
- display : 화면과 음성에 나갈 문장
"""

import json

from config import LABELS_PATH


def load_labels(path=None) -> list[dict]:
    path = path or LABELS_PATH
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    words = data["words"]
    words.sort(key=lambda w: w["id"])
    for i, w in enumerate(words):
        if w["id"] != i:
            raise ValueError(
                f"labels.json의 id는 0부터 빈틈없이 이어져야 합니다. {i}번이 없습니다."
            )
    return words


def slug_to_id(words: list[dict]) -> dict[str, int]:
    return {w["slug"]: w["id"] for w in words}


def id_to_display(words: list[dict]) -> dict[int, str]:
    return {w["id"]: w.get("display") or w["ko"] for w in words}


def num_classes(words: list[dict] | None = None) -> int:
    return len(words if words is not None else load_labels())


if __name__ == "__main__":
    ws = load_labels()
    print(f"단어 {len(ws)}개")
    for w in ws:
        print(f"  {w['id']:2d}  {w['slug']:<12} {w['ko']:<8} -> {w.get('display', '')}")
