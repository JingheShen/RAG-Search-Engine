from fastapi import FastAPI
from app.core.config import settings
from app.db.vector_store import InMemoryVectorStore
from app.rag.retriever import Retriever
from app.rag.pipeline import RAGPipeline
from app.api import ingest, query, health

# global singletons
vector_store = InMemoryVectorStore(dim=settings.VECTOR_DIM)
retriever = Retriever(store=vector_store)
rag_pipeline = RAGPipeline(retriever=retriever)

app = FastAPI(title=settings.APP_NAME)

app.include_router(health.router, prefix="/api")
app.include_router(ingest.router, prefix="/api")
app.include_router(query.router, prefix="/api")
