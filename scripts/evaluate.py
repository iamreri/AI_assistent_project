# -*- coding: utf-8 -*-
"""
Сравнение вариантов чанкинга на тестовых вопросах (recall@5).

Поиск: TF-IDF по текстам чанков. Вопрос считается «найденным», если
в топ-5 попал чанк, содержащий хотя бы одно ожидаемое ключевое слово.

Запуск:
  python scripts/evaluate.py \
      --chunks data/chunks/chunks_fixed.jsonl data/chunks/chunks_semantic.jsonl \
      --questions scripts/questions.csv
"""

import argparse
import csv
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_chunks(path):
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def load_questions(path):
    with open(path, encoding="utf-8-sig") as f:
        return [(r["question"], [k.strip().lower() for k in r["expected_keywords"].split(";") if k.strip()])
                for r in csv.DictReader(f)]


def recall_at_k(chunks, questions, k=5):
    texts = [c["text"] for c in chunks]
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
    X = vec.fit_transform(texts)
    hits = 0
    details = []
    for q, keywords in questions:
        scores = cosine_similarity(vec.transform([q]), X).ravel()
        top = scores.argsort()[::-1][:k]
        top_text = " ".join(texts[i] for i in top).lower()
        found = [kw for kw in keywords if kw in top_text]
        ok = bool(found)
        hits += ok
        details.append((q, ok, [chunks[i]["chunk_id"] for i in top]))
    return hits / max(1, len(questions)), details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", nargs="+", required=True)
    ap.add_argument("--questions", default="scripts/questions.csv")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    questions = load_questions(args.questions)
    print(f"Вопросов: {len(questions)}, метрика: recall@{args.k}\n")
    results = {}
    for path in args.chunks:
        chunks = load_chunks(path)
        name = Path(path).stem
        r, details = recall_at_k(chunks, questions, args.k)
        results[name] = r
        print(f"== {name}: {len(chunks)} чанков, recall@{args.k} = {r:.0%}")
        for q, ok, ids in details:
            mark = "OK " if ok else "MISS"
            print(f"  [{mark}] {q}")
        print()

    if len(results) == 2:
        best = max(results, key=results.get)
        print(f"Вывод: лучше вариант «{best}» ({results[best]:.0%}). "
              f"Зафиксируйте итог и обоснование в docs/data_preparation.md.")


if __name__ == "__main__":
    main()
