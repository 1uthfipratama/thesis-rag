"""Deploy to Hugging Face (PLAN_ADDENDUM Phase 11). Needs `hf auth login` first.

    uv run python scripts/deploy_space.py            # index -> private dataset, code -> Space
    uv run python scripts/deploy_space.py --code     # code only (after a code change)

Two repos under your HF account:
- dataset <user>/thesis-rag-data (PRIVATE): index.sqlite, which holds text from
  papers that aren't open access. Never the Space.
- Space <user>/thesis-rag (public, Docker): Dockerfile, requirements, rag/, app/,
  data/manifest.yaml, and deploy/SPACE_README.md as its README.

Secrets (ANTHROPIC_API_KEY, ACCESS_CODE, HF_TOKEN) are set by hand in the Space
settings; this script never reads or sends them.
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

ROOT = Path(__file__).resolve().parent.parent
SPACE_FILES = ["Dockerfile", "requirements.txt", "data/manifest.yaml"]
SPACE_DIRS = ["rag", "app"]
SKIP = ("__pycache__", ".pyc")


def space_operations() -> list[CommitOperationAdd]:
    ops = [CommitOperationAdd("README.md", str(ROOT / "deploy" / "SPACE_README.md"))]
    ops += [CommitOperationAdd(f, str(ROOT / f)) for f in SPACE_FILES]
    for d in SPACE_DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_file() and not any(s in str(p) for s in SKIP):
                ops.append(CommitOperationAdd(p.relative_to(ROOT).as_posix(), str(p)))
    return ops


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", action="store_true", help="skip the index upload")
    ap.add_argument("--space", default="thesis-rag")
    ap.add_argument("--dataset", default="thesis-rag-data")
    args = ap.parse_args()

    api = HfApi()
    user = api.whoami()["name"]
    space, dataset = f"{user}/{args.space}", f"{user}/{args.dataset}"

    if not args.code:
        index = ROOT / "data" / "index.sqlite"
        if not index.exists():
            sys.exit("data/index.sqlite missing: run `make index` first")
        api.create_repo(dataset, repo_type="dataset", private=True, exist_ok=True)
        if not api.repo_info(dataset, repo_type="dataset").private:
            sys.exit(f"{dataset} is PUBLIC; refusing to upload the index. Make it private.")
        api.upload_file(
            path_or_fileobj=str(index),
            path_in_repo="index.sqlite",
            repo_id=dataset,
            repo_type="dataset",
            commit_message="index.sqlite",
        )
        print(f"index  -> https://huggingface.co/datasets/{dataset} (private)")

    api.create_repo(space, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
    api.add_space_variable(space, "HF_DATASET_REPO", dataset)
    ops = space_operations()
    api.create_commit(space, repo_type="space", operations=ops, commit_message="deploy")
    print(f"code   -> https://huggingface.co/spaces/{space} ({len(ops)} files)")
    print(f"app    -> https://{user.lower()}-{args.space}.hf.space")


if __name__ == "__main__":
    main()
