import pytest

from rag.config import settings
from rag.manifest import load_manifest, validate


def test_manifest_schema() -> None:
    papers = load_manifest()
    assert [p.id for p in papers] == [f"p{i:02d}" for i in range(1, 20)]
    assert next(p for p in papers if p.id == "p12").drop_pages == [1]


# data/raw is gitignored, so this only runs where the PDFs exist.
@pytest.mark.skipif(not settings.raw_dir.exists(), reason="no local PDFs")
def test_manifest_matches_raw_files() -> None:
    assert validate(load_manifest()) == []


def test_short_cite_follows_apa_et_al_rule() -> None:
    for p in load_manifest():
        n = len(p.authors)
        assert p.short_cite.endswith(f" {p.year}"), p.id
        name = p.short_cite.removesuffix(f" {p.year}")
        if n == 1:
            assert " & " not in name and "et al." not in name, p.id
        elif n == 2:
            assert " & " in name and "et al." not in name, p.id
        else:
            # "et al." unless every surname is listed to disambiguate a collision
            assert name.endswith(" et al.") or name.count(",") + 1 == n - 1, p.id


def test_short_cites_are_unique() -> None:
    cites = [p.short_cite for p in load_manifest()]
    assert len(set(cites)) == len(cites)
