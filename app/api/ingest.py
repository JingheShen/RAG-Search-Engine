from fastapi import APIRouter
from pydantic import BaseModel
from app.main import rag_pipeline


router = APIRouter()


class IngestRequest(BaseModel):
    text: str


@router.post("/ingest")
def ingest(req: IngestRequest):
    rag_pipeline.ingest_document(req.text)
    return {"status": "ok", "message": "Document ingested"}
