from typing import List, Tuple
import faiss
import numpy as np


class InMemoryVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.texts: List[str] = []

    def add(self, embeddings: np.ndarray, texts: List[str]):
        self.index.add(embeddings.astype("float32"))
        self.texts.extend(texts)

    def search(self, query_emb: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        D, I = self.index.search(query_emb.astype("float32"), k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            if idx < 0 or idx >= len(self.texts):
                continue
            results.append((self.texts[idx], float(dist)))
        return results
