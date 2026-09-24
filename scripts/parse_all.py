"""Parse every manifest paper -> data/parsed/{id}.json + {id}.md."""

import argparse
import time

from rag.config import settings
from rag.manifest import load_manifest
from rag.parse.pipeline import parse_pdf
from rag.schemas import ParsedDoc


def to_markdown(doc: ParsedDoc) -> str:
    """Human-readable rendering: what will actually be indexed, section by section."""
    out = [f"# {doc.paper_id} — {doc.title}\n"]
    for s in doc.sections:
        out.append(f"\n## [{s.label}] {s.heading}  (p.{s.page_start})\n")
        for b in s.blocks:
            prefix = {"caption": "_caption:_ ", "footnote": "_footnote:_ ", "equation": ""}.get(
                b.kind, ""
            )
            out.append(f"{prefix}{b.text}  <!-- p.{b.page} -->\n")
    out.append("\n## Tables\n")
    for t in doc.tables:
        out.append(
            f"\n**{t.caption or '(no caption)'}** — p.{t.page}, section: {t.section}\n\n"
            f"{t.markdown}\n"
        )
    if doc.data_availability:
        out.append(f"\n## Data availability (metadata)\n\n{doc.data_availability}\n")
    out.append(f"\n## References ({len(doc.references)}, metadata only)\n")
    out += [f"- {r}" for r in doc.references[:5]]
    if len(doc.references) > 5:
        out.append(f"- … {len(doc.references) - 5} more")
    out.append(f"\n\n## Dropped\n\n{doc.dropped}\n")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", help="paper ids; default all")
    args = ap.parse_args()
    settings.parsed_dir.mkdir(parents=True, exist_ok=True)
    for paper in load_manifest():
        if args.ids and paper.id not in args.ids:
            continue
        t0 = time.perf_counter()
        doc = parse_pdf(paper, settings.raw_dir / paper.file)
        (settings.parsed_dir / f"{paper.id}.json").write_text(
            doc.model_dump_json(indent=1), encoding="utf-8"
        )
        (settings.parsed_dir / f"{paper.id}.md").write_text(to_markdown(doc), encoding="utf-8")
        labels = ",".join(dict.fromkeys(s.label for s in doc.sections))
        print(
            f"{paper.id} {time.perf_counter() - t0:4.1f}s  sections={len(doc.sections):2d} "
            f"tables={len(doc.tables):2d} refs={len(doc.references):3d}  [{labels}]"
        )


if __name__ == "__main__":
    main()
