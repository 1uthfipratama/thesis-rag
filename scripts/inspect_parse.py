"""Human review of one parsed paper: layout, section tree, tables, drop counts."""

import sys
from collections import Counter

from rag.config import settings
from rag.manifest import load_manifest
from rag.schemas import ParsedDoc


def main(pid: str) -> None:
    doc = ParsedDoc.model_validate_json(
        (settings.parsed_dir / f"{pid}.json").read_text(encoding="utf-8")
    )
    paper = next(p for p in load_manifest() if p.id == pid)
    print(f"{pid} — {doc.title}\n")

    layouts = Counter(v.split("+")[0] for v in doc.page_layouts.values())
    expected = paper.layout.removesuffix("_sidebar")
    detected = layouts.most_common(1)[0][0]
    flag = "" if detected == expected else "   <-- MISMATCH"
    print(f"layout: manifest={paper.layout}  detected={dict(layouts)}{flag}")
    print(
        "  per page:",
        " ".join(f"{k}:{v.replace('_column', 'col')}" for k, v in doc.page_layouts.items()),
    )

    print("\nsections:")
    for s in doc.sections:
        words = sum(len(b.text.split()) for b in s.blocks)
        print(f"  [{s.label:<17}] p.{s.page_start:<3} {words:5d}w  {s.heading[:70]}")
        text = " ".join(b.text for b in s.blocks if b.kind != "equation")
        print(f"      {text[:300]!r}")

    print(f"\ntables ({len(doc.tables)}):")
    for t in doc.tables:
        print(
            f"  p.{t.page:<3} {t.n_rows:3d}x{t.n_cols:<2d} [{t.section}] "
            f"{t.caption[:80] or '(no caption)'}"
        )
    print(
        f"\nreferences: {len(doc.references)}   data_availability: {doc.data_availability[:120]!r}"
    )
    print(f"dropped: {dict(sorted(doc.dropped.items()))}")


if __name__ == "__main__":
    main(sys.argv[1])
