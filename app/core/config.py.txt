from pydantic import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "RAG Search Engine"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_DIM: int = 384

    class Config:
        env_file = ".env"


settings = Settings()
