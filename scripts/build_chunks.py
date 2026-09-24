"""data/parsed/*.json -> data/chunks/chunks.jsonl, with a size report.

Sizes are in the embedding model's tokens (rag/tokens.py); n_tokens is the full
embed_text as the embedder sees it, so "over 512" means "would be truncated".
"""

from collections import Counter

from rag.chunk import chunk_doc
from rag.config import settings
from rag.manifest import load_manifest
from rag.schemas import ParsedDoc
from rag.tokens import EMBED_MAX_TOKENS


def main() -> None:
    chunks = []
    for paper in load_manifest():
        doc = ParsedDoc.model_validate_json(
            (settings.parsed_dir / f"{paper.id}.json").read_text(encoding="utf-8")
        )
        cs = chunk_doc(paper, doc)
        kinds = Counter(c.kind for c in cs)
        print(
            f"{paper.id}  {len(cs):3d} chunks  (prose {kinds['prose']:3d}, "
            f"table {kinds['table']:2d}, card {kinds['paper_card']})  "
            f"max {max(c.n_tokens for c in cs)} tok  [{doc.backend}]"
        )
        chunks += cs

    settings.chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with settings.chunks_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")

    lens = sorted(c.n_tokens for c in chunks)
    print(f"\ntotal {len(chunks)} chunks -> {settings.chunks_path}")
    print(f"embedder tokens (embed_text): median {lens[len(lens) // 2]}, max {lens[-1]}")
    edges = [0, 100, 200, 300, 400, 450, EMBED_MAX_TOKENS + 1]
    for lo, hi in zip(edges, edges[1:], strict=False):
        n = sum(lo <= x < hi for x in lens)
        print(f"  {lo:3d}-{hi - 1:3d}  {n:4d}  {'#' * (n // 5)}")
    over = sum(x > EMBED_MAX_TOKENS for x in lens)
    refs = sum(c.section == "references" for c in chunks)
    dup = len(chunks) - len({c.chunk_id for c in chunks})
    print(
        f"over {EMBED_MAX_TOKENS} (truncated by embedder): {over}   "
        f"references: {refs}   dup ids: {dup}"
    )


if __name__ == "__main__":
    main()
