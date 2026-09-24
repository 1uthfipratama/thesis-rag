"""Check every manifest entry has its PDF in data/raw and the sha256 matches."""

import sys

from rag.manifest import load_manifest, validate

papers = load_manifest()
problems = validate(papers)
for p in papers:
    print(f"{p.id}  {p.sha256[:12]}  {p.layout:<20} {p.short_cite}")
if problems:
    print("\nFAIL", *problems, sep="\n  ")
    sys.exit(1)
print(f"\nOK: {len(papers)} papers, all files present, all sha256 match.")
