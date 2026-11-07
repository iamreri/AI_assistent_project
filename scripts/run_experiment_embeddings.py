import json
import chromadb
from sentence_transformers import SentenceTransformer
from src.generation.llm import LLMGenerator

def run_experiment():
    question = "Какие условия для получения повышенной стипендии?"
    print(f"Вопрос: {question}\n")

    chunks = []
    with open("data/chunks/chunks_semantic.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
            
    texts = [c["text"] for c in chunks]
    metadatas = [{k: v for k, v in c.items() if k != "text"} for c in chunks]
    ids = [str(c["chunk_id"]) for c in chunks]

    llm = LLMGenerator()
    
    models = [
        "intfloat/multilingual-e5-small",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ]

    for model_name in models:
        print("="*50)
        print(f"МОДЕЛЬ: {model_name}")
        print("="*50)

        embedder = SentenceTransformer(model_name)
        
        # Создаем временный клиент в памяти (каждый раз новый)
        client = chromadb.EphemeralClient() 
        coll_name = f"test_{model_name.split('/')[-1].replace('-', '_')}"
        coll = client.create_collection(name=coll_name)
        
        print(f"Считаем эмбеддинги для {len(texts)} чанков (может занять минуту)...")
        embeds = embedder.encode(texts, normalize_embeddings=True).tolist()
        coll.add(embeddings=embeds, documents=texts, metadatas=metadatas, ids=ids)

        query_emb = embedder.encode([question], normalize_embeddings=True).tolist()
        results = coll.query(query_embeddings=query_emb, n_results=4)
        
        retrieved_chunks = []
        for i in range(len(results['ids'][0])):
            retrieved_chunks.append({
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i]
            })
        
        answer = llm.generate(question, retrieved_chunks)
        print(f"ОТВЕТ ИИ:\n{answer}\n")
        
        print("ИСТОЧНИКИ:")
        for i, c in enumerate(retrieved_chunks, 1):
            print(f"{i}. {c['metadata'].get('document_title')}")
        print("\n")

if __name__ == "__main__":
    run_experiment()
