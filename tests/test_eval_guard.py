"""PLAN_ADDENDUM 13.1: evaluation must only ever see the core corpus, so uploads
can never change the reported numbers."""

import importlib.util
from pathlib import Path


def _load(name: str):
    path = Path(__file__).resolve().parent.parent / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_retrieval_eval_is_locked_to_core() -> None:
    assert _load("eval_retrieval").COLLECTIONS == ["core"]
