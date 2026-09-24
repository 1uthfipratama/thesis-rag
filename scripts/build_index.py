"""data/chunks/chunks.jsonl -> data/index.sqlite (rebuilt from scratch), plus the
Phase 4 smoke check: a Lotka-Volterra query must find p12 via BM25 and dense."""

import time

from rag.config import settings
from rag.index import build, connect
from rag.manifest import load_manifest
from rag.retrieve import bm25, dense, describe
from rag.schemas import Chunk

SMOKE = "Lotka-Volterra Adams-Bashforth"


def main() -> None:
    chunks = [
        Chunk.model_validate_json(line)
        for line in settings.chunks_path.read_text(encoding="utf-8").splitlines()
    ]
    t0 = time.perf_counter()
    meta = build(chunks, load_manifest())
    print(f"built {settings.index_path} in {time.perf_counter() - t0:.1f}s")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    db = connect(readonly=True)
    counts = {
        t: db.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        for t in ("papers", "chunks", "fts_chunks", "vec_chunks")
    }
    print(f"\nrows: {counts}  (chunks.jsonl: {len(chunks)})")

    print(f"\nsmoke query: {SMOKE!r}")
    for name, fn in (("bm25", bm25), ("dense", dense)):
        hits = fn(db, SMOKE, 5)
        rows = describe(db, [r for r, _ in hits])
        print(f"  {name}:")
        for (_, score), (cid, _pid, _sec, page) in zip(hits, rows, strict=True):
            print(f"    {score:8.4f}  {cid:<28} p.{page}")


if __name__ == "__main__":
    main()
