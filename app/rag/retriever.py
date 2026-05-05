from typing import List, Tuple
import numpy as np
from app.db.vector_store import InMemoryVectorStore
from app.rag.embedder import embed_texts


class Retriever:
    def __init__(self, store: InMemoryVectorStore):
        self.store = store

    def add_documents(self, chunks: List[str]):
        embs = embed_texts(chunks)
        self.store.add(embs, chunks)

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        q_emb = embed_texts([query])
        return self.store.search(q_emb, k=k)
