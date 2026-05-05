from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from app.core.config import settings

_model = SentenceTransformer(settings.EMBEDDING_MODEL)


def embed_texts(texts: List[str]) -> np.ndarray:
    return _model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
