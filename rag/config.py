"""Runtime settings. Every value can be overridden by an env var or `.env`."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # Paths. DATA_DIR is overridable so the HF Space can point it at /home/user/data.
    data_dir: Path = ROOT / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def parsed_dir(self) -> Path:
        return self.data_dir / "parsed"

    @property
    def chunks_path(self) -> Path:
        return self.data_dir / "chunks" / "chunks.jsonl"

    @property
    def index_path(self) -> Path:
        return self.data_dir / "index.sqlite"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.yaml"

    # Models
    embed_model: str = "BAAI/bge-small-en-v1.5"
    llm_model: str = "claude-haiku-4-5-20251001"
    anthropic_api_key: str = ""

    # Retrieval: fuse 30 BM25 + 30 dense candidates, return 6.
    top_k: int = 6
    bm25_k: int = 30
    dense_k: int = 30
    rrf_k: int = 60  # standard RRF constant; damps the influence of rank-1 outliers

    # API guards
    max_question_chars: int = 500
    rate_limit: str = "10/minute"


settings = Settings()
