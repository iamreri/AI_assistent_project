import glob
import chromadb
from src.embeddings.encoder import Embedder
from src.generation.llm import LLMGenerator

def split_text(text, chunk_size):
    """Простое разбиение текста на куски фиксированного размера"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def run_experiment():
    question = "Что нужно сделать для получения места в общежитии?"
    print(f"Вопрос: {question}\n")

    # Загружаем все очищенные тексты из нашей папки
    texts = []
    for file in glob.glob("data/documents/clean/*.md"):
        with open(file, "r", encoding="utf-8") as f:
            texts.append(f.read())
    full_text = "\n\n".join(texts)

    embedder = Embedder()
    llm = LLMGenerator()

    for size in [300, 500, 800]:
        print("="*50)
        print(f"ЗАПУСК: размер чанка = {size} символов")
        print("="*50)

        # Нарезаем текст на куски заданного размера
        chunks = split_text(full_text, size)
        
        # Считаем эмбеддинги
        print(f"Считаем эмбеддинги для {len(chunks)} кусков (это может занять минуту)...")
        embeds = embedder.encode(chunks)

        # Создаем временную базу в памяти, чтобы не портить основную
        client = chromadb.Client() 
        coll = client.create_collection(name=f"test_{size}")
        coll.add(embeddings=embeds, documents=chunks, ids=[str(i) for i in range(len(chunks))])

        # Ищем top-4 куска
        query_emb = embedder.encode([question])[0]
        results = coll.query(query_embeddings=[query_emb], n_results=4)
        
        retrieved_chunks = [{"text": doc, "metadata": {"document_title": "Фрагмент"}} for doc in results['documents'][0]]
        
        # Генерируем ответ
        answer = llm.generate(question, retrieved_chunks)
        print(f"ОТВЕТ ИИ:\n{answer}\n")

if __name__ == "__main__":
    run_experiment()