"""Phase 5: RRF, named-paper detection, diversity cap, filters, neighbours."""

import pytest

from rag.config import settings
from rag.retrieve import MAX_PER_PAPER, named_papers, neighbours, rrf, search


def test_rrf_rewards_agreement_between_lists() -> None:
    fused = dict(rrf([[1, 2, 3], [3, 1, 4]], k=60))
    assert fused[1] == pytest.approx(1 / 61 + 1 / 62)
    assert max(fused, key=fused.get) == 1  # rank 1 + rank 2 beats rank 3 + rank 1
    assert fused[4] == pytest.approx(1 / 63)


@pytest.mark.parametrize(
    ("q", "expected"),
    [
        ("What sample did Smales use?", {"p14"}),
        ("What did Lyocsa and Todorova find?", {"p10"}),  # diacritics folded
        ("Compare p12 and p13", {"p12", "p13"}),
        ("Noviantri 2023 fit MAPE", {"p02", "p13"}),  # both 2023: still ambiguous
        ("Noviantri 2026 algorithm", {"p05"}),
        ("Chandra et al. 2023", {"p02", "p13"}),  # p13's alternative author order
        ("Which of Li et al.'s four dynamic models did best?", {"p01"}),  # 2-letter surname
        ("Is the lithium price in the corpus?", set()),  # "li" inside a word: not a name
        ("Which papers use GARCH?", set()),
    ],
)
def test_named_papers(q: str, expected: set[str]) -> None:
    assert named_papers(q) == expected


corpus = pytest.mark.skipif(not settings.index_path.exists(), reason="run `make index` first")


@pytest.fixture(scope="module")
def db():
    from rag.index import connect

    conn = connect(readonly=True)
    yield conn
    conn.close()


@corpus
def test_diversity_cap_on_unnamed_question(db) -> None:
    hits = search(db, "which papers use the geopolitical risk index", collections=["core"])
    assert len(hits) == settings.top_k
    counts = {}
    for h in hits:
        counts[h.paper_id] = counts.get(h.paper_id, 0) + 1
    assert max(counts.values()) <= MAX_PER_PAPER


@corpus
def test_cap_lifted_when_one_paper_is_named(db) -> None:
    hits = search(db, "Smales geopolitical risk GARCH oil stock volatility", collections=["core"])
    assert sum(h.paper_id == "p14" for h in hits) > MAX_PER_PAPER


@corpus
def test_filters(db) -> None:
    assert all(h.paper_id == "p13" for h in search(db, "MAPE forecast", paper_ids=["p13"]))
    assert search(db, "MAPE forecast", collections=["user"]) == []  # no uploads yet
    assert search(db, "MAPE forecast", collections=["core"])


@corpus
def test_neighbours_stay_in_section(db) -> None:
    hit = next(h for h in search(db, "Heston model volatility process") if h.kind == "prose")
    for seq, _ in neighbours(db, hit):
        row = db.execute(
            "SELECT paper_id, heading FROM chunks WHERE paper_id = ? AND seq = ?",
            (hit.paper_id, seq),
        ).fetchone()
        assert row == (hit.paper_id, hit.heading)
        assert abs(seq - hit.seq) == 1


@pytest.mark.parametrize(
    ("q", "mode", "expected"),
    [
        ("Which papers use GARCH?", "adaptive", 2),  # list-style: breadth
        ("What MAPE did the model get?", "adaptive", 6),  # single topic: full depth
        ("Across the repeated forecast experiments, what share was accurate?", "adaptive", 6),
        ("What step-by-step methods show up across the collection?", "adaptive", 2),
        ("What sample did Smales use?", "adaptive", 6),  # one paper named: no cap
        ("Which papers use GARCH?", "fixed", 3),
    ],
)
def test_per_paper_cap(q: str, mode: str, expected: int) -> None:
    from rag.retrieve import per_paper_cap

    assert per_paper_cap(q, top_k=6, mode=mode, base=3) == expected


@corpus
def test_top_of_each_retriever_is_guaranteed(db) -> None:
    """q56: the equation chunk is dense rank 1 but BM25 rank 20; plain RRF put it 7th."""
    q = (
        "What stochastic differential equation defines the SP-SPDE stock price model "
        "in the pantograph-delay paper?"
    )
    assert "p06:methodology:003" in [h.chunk_id for h in search(db, q, collections=["core"])]
