from rag.config import settings


def test_defaults_match_plan() -> None:
    assert settings.top_k == 6
    assert settings.rrf_k == 60
    assert settings.embed_model == "BAAI/bge-small-en-v1.5"
