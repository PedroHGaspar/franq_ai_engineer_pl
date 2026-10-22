from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    openai_api_key: SecretStr = SecretStr("")
    openai_model: str = "gpt-5.4-mini-2026-03-17"
    enable_paid_api: bool = False
    database_path: Path = Path("anexo_desafio_1.db")
    max_model_calls: int = Field(16, ge=1, le=30)
    max_iterations: int = Field(16, ge=1, le=30)
    max_queries: int = Field(10, ge=1, le=20)
    max_corrections: int = Field(3, ge=0, le=5)
    max_rows: int = Field(500, ge=1, le=5000)
    max_result_bytes: int = Field(60_000, ge=1000, le=500_000)
    sql_timeout_seconds: float = Field(3, gt=0, le=30)
    max_output_tokens: int = Field(2500, ge=256, le=8000)
    max_input_tokens: int = Field(20000, ge=1000, le=32000)
    api_budget_usd: float = Field(0.50, gt=0, le=5)
    input_price_per_million: float = Field(0.75, ge=0)
    output_price_per_million: float = Field(4.50, ge=0)
    local_dir: Path = Path(".local")
