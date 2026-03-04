"""
Centralized configuration via pydantic-settings.

All environment variables are loaded here and exposed as typed attributes.
Import `settings` from this module wherever config is needed.
"""

import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

_script_dir = Path(__file__).resolve().parent.parent  # demo/
for _env_path in [_script_dir / ".env", _script_dir.parent / ".env", Path.cwd() / ".env"]:
    if _env_path.is_file():
        load_dotenv(_env_path, override=True)
        break
else:
    load_dotenv(override=True)


class Settings(BaseSettings):
    # MongoDB
    mongodb_url: str = Field("mongodb://localhost:27017", alias="MONGODB_URL")
    # Qdrant
    qdrant_url: str = Field("http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(None, alias="QDRANT_API_KEY")
    # OpenAI
    openai_api_key: Optional[str] = Field(None, alias="OPENAI_API_KEY")
    openai_model: str = Field("gpt-5-mini", alias="OPENAI_MODEL")
    openai_vision_model: str = Field("gpt-4o", alias="OPENAI_VISION_MODEL")
    embedding_model: str = Field("text-embedding-3-large", alias="EMBEDDING_MODEL")
    embedding_dim: int = 3072
    # JWT
    jwt_secret: str = Field(..., alias="JWT_SECRET", min_length=32)
    jwt_algorithm: str = "HS256"
    jwt_access_expire_minutes: int = Field(30, alias="JWT_ACCESS_EXPIRE_MINUTES")
    jwt_refresh_expire_days: int = Field(7, alias="JWT_REFRESH_EXPIRE_DAYS")
    # Redis
    redis_url: str = Field("redis://localhost:6379/0", alias="REDIS_URL")
    # Celery
    celery_broker_url: str = Field("redis://localhost:6379/1", alias="CELERY_BROKER_URL")
    # CORS
    cors_origins: str = Field(
        "http://localhost:8000,http://127.0.0.1:8000", alias="CORS_ORIGINS"
    )
    # Weather
    openweathermap_api_key: str = Field("", alias="OPENWEATHERMAP_API_KEY")
    # S3
    s3_bucket_name: str = Field("", alias="S3_BUCKET_NAME")
    s3_endpoint_url: str = Field("", alias="S3_ENDPOINT_URL")
    # Logging
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    class Config:
        env_file = ".env"
        extra = "ignore"

    BASE_DIR: Path = _script_dir


settings = Settings()
