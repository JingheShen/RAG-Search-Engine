from app.db.vector_store import InMemoryVectorStore
from app.rag.retriever import Retriever
from app.rag.pipeline import RAGPipeline


def test_rag_pipeline_smoke():
    store = InMemoryVectorStore(dim=384)
    retriever = Retriever(store)
    rag = RAGPipeline(retriever)

    rag.ingest_document("This is a test document about machine learning and RAG systems.")
    ans = rag.answer("What is this document about?")
    assert "mocked" in ans
