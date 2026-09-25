"""Private data on the HF Space (PLAN_ADDENDUM 11.4).

The Space repo is public, so it never holds the index: index.sqlite contains text
from papers that aren't open access. It lives in a private HF dataset repo and is
downloaded into DATA_DIR on boot. Upload it with scripts/deploy_space.py.

push_index / push_pdf keep user uploads (Phase 13) in the dataset too. All three
are no-ops when HF_DATASET_REPO is unset (local use).
"""

import logging
from pathlib import Path

from rag.config import settings

log = logging.getLogger("thesis_rag")
INDEX_FILE = "index.sqlite"


def pull() -> None:
    """Download the index from the private dataset. No-op when no dataset is
    configured (local development uses data/index.sqlite as built)."""
    if not settings.hf_dataset_repo:
        return
    from huggingface_hub import hf_hub_download

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=settings.hf_dataset_repo,
        filename=INDEX_FILE,
        repo_type="dataset",
        token=settings.hf_token or None,
        local_dir=settings.data_dir,
    )
    log.info("pulled %s from %s", path, settings.hf_dataset_repo)


def _upload(local: Path, path_in_repo: str) -> None:
    """Best effort: a failed sync is logged, never shown to the user as a failed upload."""
    if not settings.hf_dataset_repo or not local.exists():
        return
    try:
        from huggingface_hub import HfApi

        HfApi(token=settings.hf_token or None).upload_file(
            path_or_fileobj=str(local),
            path_in_repo=path_in_repo,
            repo_id=settings.hf_dataset_repo,
            repo_type="dataset",
            commit_message=f"sync {path_in_repo}",
        )
    except Exception:
        log.exception("dataset sync failed for %s", path_in_repo)


def push_index() -> None:
    """After an ingest or delete, so uploads survive a restart of a hosted demo."""
    _upload(settings.index_path, INDEX_FILE)


def push_pdf(local: Path, path_in_repo: str) -> None:
    _upload(local, path_in_repo)
