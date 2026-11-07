from src.rag.pipeline import RAGPipeline

# 1. Инициализация
rag = RAGPipeline()

# 2. Индексация (если база пустая)
rag.index_data()

# 3. Задаем вопрос
question = "Какие условия для получения повышенной стипендии?"
print(f"\nВопрос: {question}")

# 4. Получаем ответ
result = rag.ask(question, top_k=4)

# 5. Выводим результат
print("\n--- ОТВЕТ ИИ ---")
print(result["answer"])
print("\n--- ИСТОЧНИКИ ---")
for i, c in enumerate(result["chunks"], 1):
    print(f"{i}. {c['metadata'].get('document_title')} (файл: {c['metadata'].get('document_id')})")