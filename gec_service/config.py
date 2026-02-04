from pydantic import BaseSettings


class Settings(BaseSettings):
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    TOP_K: int = 5
    CACHE_THRESHOLD: float = 0.95
    RETRIEVAL_ENABLED: bool = True
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_API_KEY: str | None = None
    INDEX_PATH: str = "./data/index.npz"

    class Config:
        env_file = ".env"


settings = Settings()
