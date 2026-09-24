"""Runtime settings. Every value can be overridden by an env var or `.env`."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # Paths. DATA_DIR is overridable so the HF Space can point it at /home/user/data.
    data_dir: Path = ROOT / "data"
    # Point tests/chunking at another backend's output, e.g. data/parsed_alt/mineru
    parsed_dir_override: Path | None = None

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def parsed_dir(self) -> Path:
        return self.parsed_dir_override or self.data_dir / "parsed"

    @property
    def chunks_path(self) -> Path:
        return self.data_dir / "chunks" / "chunks.jsonl"

    @property
    def index_path(self) -> Path:
        return self.data_dir / "index.sqlite"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.yaml"

    # Parser backend for the core corpus: pymupdf | mineru | hybrid (rag/parse/backends.py)
    # mineru won the parse-level bake-off (eval/results/parsers_20260924_1635.md).
    # User uploads must not use it: ~8 s/page on CPU is too slow for the free Space.
    parse_backend: str = "mineru"

    # Models
    embed_model: str = "BAAI/bge-small-en-v1.5"
    llm_model: str = "claude-haiku-4-5-20251001"
    # BGE retrieval instruction, prepended to queries only (rag/embed.py). "" disables.
    query_instruction: str = "Represent this sentence for searching relevant passages: "
    anthropic_api_key: str = ""

    # Retrieval: fuse 30 BM25 + 30 dense candidates, return 6.
    top_k: int = 6
    bm25_k: int = 30
    dense_k: int = 30
    rrf_k: int = 60  # standard RRF constant; damps the influence of rank-1 outliers
    # Diversity cap (rag/retrieve.per_paper_cap): max chunks per paper in top_k.
    # "adaptive" tightens to 2 only for list-style questions; see eval/results.
    max_per_paper: int = 3
    cap_mode: str = "adaptive"  # eval/results/retrieval_*_1735: best coverage/depth trade-off

    # API guards
    max_question_chars: int = 500
    rate_limit: str = "10/minute"


settings = Settings()
