from typing import List
from app.rag.chunker.py import simple_chunk
from app.rag.retriever import Retriever
from app.rag.generator import simple_llm_generate


class RAGPipeline:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def ingest_document(self, text: str):
        chunks = simple_chunk(text)
        self.retriever.add_documents(chunks)

    def answer(self, query: str, k: int = 5) -> str:
        results = self.retriever.retrieve(query, k=k)
        contexts = [r[0] for r in results]
        return simple_llm_generate(query, contexts)
