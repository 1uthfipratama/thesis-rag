"""data/parsed/*.json -> data/chunks/chunks.jsonl, with a size report.

uv run python scripts/build_chunks.py          # build + report
uv run python scripts/build_chunks.py --bge    # also count chunks the embedder truncates
"""

import argparse
from collections import Counter

from rag.chunk import MAX_TOKENS, chunk_doc
from rag.config import settings
from rag.manifest import load_manifest
from rag.schemas import ParsedDoc

BGE_MAX = 512  # bge-small-en-v1.5 max sequence length (its own WordPiece tokens)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bge", action="store_true", help="measure truncation by the embedder")
    args = ap.parse_args()

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

    print(f"\ntotal {len(chunks)} chunks -> {settings.chunks_path}")
    edges = [0, 100, 200, 300, 400, 500, 600, MAX_TOKENS + 1]
    print("token histogram (cl100k, text only):")
    for lo, hi in zip(edges, edges[1:], strict=False):
        n = sum(lo <= c.n_tokens < hi for c in chunks)
        print(f"  {lo:3d}-{hi - 1:3d}  {n:4d}  {'#' * (n // 5)}")
    over = [c.chunk_id for c in chunks if c.n_tokens > MAX_TOKENS]
    refs = [c.chunk_id for c in chunks if c.section == "references"]
    dup = len(chunks) - len({c.chunk_id for c in chunks})
    print(
        f"over {MAX_TOKENS}: {len(over)}   section==references: {len(refs)}   duplicate ids: {dup}"
    )

    if args.bge:
        from fastembed import TextEmbedding

        tok = TextEmbedding(settings.embed_model).model.tokenizer
        tok.no_truncation()
        lens = [len(tok.encode(c.embed_text).ids) for c in chunks]
        cut = [(c.chunk_id, n) for c, n in zip(chunks, lens, strict=True) if n > BGE_MAX]
        print(
            f"\nembedder view ({settings.embed_model}): median {sorted(lens)[len(lens) // 2]} "
            f"tokens, {len(cut)}/{len(chunks)} chunks exceed {BGE_MAX} and would be truncated"
        )


if __name__ == "__main__":
    main()
