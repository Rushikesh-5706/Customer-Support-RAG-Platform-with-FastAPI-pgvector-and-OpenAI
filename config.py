from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    database_url: str
    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "gpt-4o-mini"
    chunk_size: int = 512
    chunk_overlap: int = 50
    hybrid_alpha: float = 0.7
    top_k: int = 5

    class Config:
        env_file = ".env"


settings = Settings()
