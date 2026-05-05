from fastapi import APIRouter
from pydantic import BaseModel
from app.main import rag_pipeline


router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


@router.post("/query")
def query(req: QueryRequest):
    answer = rag_pipeline.answer(req.query, k=req.top_k)
    return {"answer": answer}
