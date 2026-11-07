from src.rag.pipeline import RAGPipeline
import time

# Вопрос для теста (можешь поменять на любой другой)
question = "Что нужно сделать для получения места в общежитии?"

def run_experiment():
    print(f"Вопрос: {question}\n")
    rag = RAGPipeline() # Инициализация (база уже собрана)

    for k in [2, 4, 8]:
        print("="*50)
        print(f"ЗАПУСК: top_k = {k}")
        print("="*50)
        
        start_time = time.time()
        result = rag.ask(question, top_k=k)
        end_time = time.time()

        print(f"ОТВЕТ ИИ:\n{result['answer']}\n")
        print(f"ИСТОЧНИКИ (всего {len(result['chunks'])}):")
        for i, c in enumerate(result['chunks'], 1):
            print(f"{i}. {c['metadata'].get('document_title')}")
        
        print(f"\nВремя выполнения: {end_time - start_time:.2f} сек.\n")

if __name__ == "__main__":
    run_experiment()