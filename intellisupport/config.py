from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str
    database_url: str
    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "gpt-4o-mini"
    chunk_size: int = 512
    chunk_overlap: int = 50
    hybrid_alpha: float = 0.7
    top_k: int = 5

    model_config = SettingsConfigDict(
        # In pydantic-settings v2, later files override earlier ones.
        # ../.env is loaded first (Docker root, lower priority).
        # .env is loaded second (local, higher priority — wins).
        env_file=("../.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
