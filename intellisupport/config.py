from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = ""
    database_url: str = "postgresql://postgres:postgres@localhost:5432/intellisupport"
    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "gpt-4o-mini"
    chunk_size: int = 512
    chunk_overlap: int = 50
    hybrid_alpha: float = 0.7
    top_k: int = 5

    model_config = SettingsConfigDict(
        env_file=(".env",),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
