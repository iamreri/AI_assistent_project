import csv
import time
from src.rag.pipeline import RAGPipeline

print("Загрузка ассистента...")
rag = RAGPipeline()

with open("benchmark.csv", "r", encoding="utf-8") as f:
    questions = list(csv.DictReader(f))

with open("benchmark_results.csv", "w", encoding="utf-8", newline="") as f:
    fieldnames = ["id", "category", "question", "expected_behavior", "ai_answer", "ai_sources"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for i, row in enumerate(questions, 1):
        print(f"[{i}/100] Вопрос: {row['question'][:40]}...")
        try:
            result = rag.ask(row["question"], top_k=4)
            answer = result["answer"]
            sources = ";".join([c['metadata'].get('document_id', '') for c in result['chunks']])
        except Exception as e:
            answer = f"ОШИБКА: {e}"
            sources = ""
        
        writer.writerow({
            "id": row["id"], "category": row["category"], "question": row["question"],
            "expected_behavior": row["expected_behavior"], "ai_answer": answer, "ai_sources": sources
        })
        time.sleep(3)
print("Готово! Все ответы в benchmark_results.csv")
