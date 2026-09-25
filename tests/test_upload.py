"""Phase 13: PDF uploads, on a temporary copy of the index (the real one is untouched)."""

import shutil

import pymupdf
import pytest

from rag import upload
from rag.config import settings


def pdf_bytes(text: str = "", pages: int = 1, password: str = "") -> bytes:
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=9)
    kw = {"encryption": pymupdf.PDF_ENCRYPT_AES_256, "user_pw": password} if password else {}
    return doc.tobytes(**kw)


PROSE = "Stock prices were modelled with a logistic equation and fitted by least squares. " * 12


@pytest.mark.parametrize(
    ("data", "reason"),
    [
        (b"hello, not a pdf", "isn't a PDF"),
        (pdf_bytes(PROSE, password="x"), "password-protected"),
        (pdf_bytes(PROSE, pages=upload.MAX_PAGES + 1), "pages only"),
        (pdf_bytes("", pages=3), "scanned PDF"),  # no text layer
    ],
    ids=["not-pdf", "encrypted", "too-long", "scanned"],
)
def test_validation_rejects(data: bytes, reason: str) -> None:
    with pytest.raises(upload.Rejected, match=reason):
        upload.validate(data, "x.pdf")


def test_validation_accepts_text_pdf_and_reads_metadata() -> None:
    checked = upload.validate(pdf_bytes(PROSE), "my_paper.pdf")
    assert checked.n_pages == 1 and len(checked.sha256) == 64
    assert checked.authors == ["Unknown author"] and checked.title


SAMPLE = settings.raw_dir / "p19.pdf"
needs_corpus = pytest.mark.skipif(
    not (settings.index_path.exists() and SAMPLE.exists()), reason="needs index + raw PDFs"
)


@pytest.fixture()
def tmp_index(tmp_path, monkeypatch):
    from rag.index import connect

    path = tmp_path / "index.sqlite"
    shutil.copy(settings.index_path, path)
    shutil.copy(settings.manifest_path, tmp_path / "manifest.yaml")
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "hf_dataset_repo", "")  # no dataset sync in tests
    db = connect(path)
    upload.ensure_tables(db)
    yield db
    db.close()


@needs_corpus
def test_upload_ingest_search_delete(tmp_index) -> None:
    """The Phase 13 acceptance check, offline: a core paper uploaded as a user
    document answers user-scoped searches with u01, and deletion removes it."""
    from rag.retrieve import search

    db = tmp_index
    core_before = db.execute("SELECT count(*) FROM chunks WHERE collection='core'").fetchone()[0]
    data = SAMPLE.read_bytes()
    job, pid = upload.create(db, data, "enow.pdf")
    assert pid == "u01" and job is not None
    upload.ingest(job.job_id, settings.index_path)
    assert upload.get_job(job.job_id)["status"] == "ready"
    assert upload.documents(db)[0]["status"] == "ready"

    again, same = upload.create(db, data, "copy.pdf")  # duplicate: no second ingest
    assert again is None and same == "u01"

    q = "Hurst exponent long memory stock markets"
    user_hits = search(db, q, collections=["user"])
    assert user_hits and {h.paper_id for h in user_hits} == {"u01"}
    assert all(h.paper_id != "u01" for h in search(db, q, collections=["core"]))
    n_core = db.execute("SELECT count(*) FROM chunks WHERE collection='core'").fetchone()[0]
    assert n_core == core_before  # the gold-set corpus is untouched

    assert upload.delete(db, "u01")
    assert db.execute("SELECT count(*) FROM chunks WHERE paper_id='u01'").fetchone()[0] == 0
    assert not search(db, q, collections=["user"])
    assert not upload.delete(db, "p19")  # core papers can't be deleted
    assert not (settings.data_dir / "uploads" / "u01.pdf").exists()


@needs_corpus
def test_upload_api_guards(monkeypatch, tmp_path) -> None:
    from fastapi.testclient import TestClient

    from app import main

    monkeypatch.setattr(settings, "access_code", "c")
    main.limiter.reset()
    with TestClient(main.app) as client:

        def post(data: bytes, code: str = "c"):
            files = {"file": ("x.pdf", data, "application/pdf")}
            return client.post("/api/upload", files=files, headers={"x-access-code": code})

        assert post(pdf_bytes(PROSE), code="wrong").status_code == 401
        r = post(b"not a pdf")
        assert r.status_code == 400 and "isn't a PDF" in r.json()["error"]
        assert client.delete("/api/documents/u01").status_code == 401
        assert client.get("/api/documents").status_code == 200
        assert client.get("/api/jobs/nope").status_code == 404
