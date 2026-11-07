from src.embeddings.encoder import Embedder
from src.retrieval.vector_store import VectorStore
from src.generation.llm import LLMGenerator

class RAGPipeline:
    def __init__(self):
        print("Инициализация компонентов...")
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self.llm = LLMGenerator()

    def index_data(self, chunks_path="data/chunks/chunks_semantic.jsonl"):
        self.vector_store.index_data(chunks_path, self.embedder)

    def ask(self, query: str, top_k: int = 4):
        # 1. Эмбеддинг вопроса
        query_emb = self.embedder.encode([query])[0]
        # 2. Поиск в базе
        chunks = self.vector_store.search(query_emb, top_k=top_k)
        # 3. Генерация ответа
        answer = self.llm.generate(query, chunks)
        return {"answer": answer, "chunks": chunks}
