"""Private data on the HF Space (PLAN_ADDENDUM 11.4).

The Space repo is public, so it never holds the index: index.sqlite contains text
from papers that aren't open access. It lives in a private HF dataset repo and is
downloaded into DATA_DIR on boot. Upload it with scripts/deploy_space.py.

push_index / push_pdf (user uploads) arrive with Phase 13.
"""

import logging

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
