import chromadb
import json

class VectorStore:
    def __init__(self, db_path="./chroma_db", collection_name="knowledge_base"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def index_data(self, chunks_path: str, embedder):
        if self.collection.count() > 0:
            print(f"В базе уже {self.collection.count()} чанков.")
            return

        chunks = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))

        texts = [c["text"] for c in chunks]
        ids = [str(c["chunk_id"]) for c in chunks]
        metadatas = [{k: v for k, v in c.items() if k != "text"} for c in chunks]
        embeddings = embedder.encode(texts)

        self.collection.add(embeddings=embeddings, documents=texts, metadatas=metadatas, ids=ids)
        print(f"Проиндексировано {len(texts)} чанков.")

    def search(self, query_embedding, top_k=4):
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        chunks = []
        for i in range(len(results['ids'][0])):
            chunks.append({
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i]
            })
        return chunks
