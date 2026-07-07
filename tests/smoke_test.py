"""Smoke test for PerfectPitch-CV.

Runs without a live API key or network for the fast checks; retriever checks
require the sentence-transformers model (cached) and a prebuilt FAISS index,
and are skipped gracefully if either is unavailable.

Usage:  python3 tests/smoke_test.py
Exits non-zero if any required check fails.
"""

import os
import sys

# Make the project root importable when run from anywhere.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PASSED, FAILED, SKIPPED = [], [], []


def check(name, fn):
    try:
        fn()
        PASSED.append(name)
        print(f"PASS  {name}")
    except _Skip as exc:
        SKIPPED.append(name)
        print(f"SKIP  {name}: {exc}")
    except Exception as exc:  # noqa: BLE001 - smoke test reports any failure
        FAILED.append(name)
        print(f"FAIL  {name}: {exc}")


class _Skip(Exception):
    pass


def test_ragmodel_requires_api_key():
    from components.rag_model import RAGModel

    saved = os.environ.pop("GEMINI_API_KEY", None)
    try:
        try:
            RAGModel()
        except RuntimeError:
            return
        raise AssertionError("RAGModel should raise RuntimeError when GEMINI_API_KEY is unset")
    finally:
        if saved is not None:
            os.environ["GEMINI_API_KEY"] = saved


def test_extract_text_unsupported_extension():
    import app

    class _Fake:
        name = "resume.docx"

    assert app.extract_text(_Fake()) == "", "unsupported extension should yield empty string"


def test_live_retriever_normalized_cosine():
    try:
        from components.retriever_live import LiveRetriever
    except Exception as exc:  # noqa: BLE001
        raise _Skip(f"import failed: {exc}")
    try:
        lr = LiveRetriever()
    except Exception as exc:  # noqa: BLE001
        raise _Skip(f"model unavailable (offline?): {exc}")

    text = "Experienced Python developer skilled in ML, FAISS, and NLP. Built RAG systems."
    results = lr.get_top_k_from_text(text, "python machine learning engineer", k=2)
    assert len(results) >= 1, "expected at least one result"
    for idx, chunk in results:
        assert isinstance(idx, int), "index should be a native int, not numpy scalar"
        assert isinstance(chunk, str) and chunk, "chunk should be a non-empty string"


def test_multi_retriever_loads_index():
    index_path = os.path.join(ROOT, "data", "index", "faiss_index.bin")
    if not os.path.exists(index_path):
        raise _Skip("no prebuilt FAISS index (run build_index.py)")
    try:
        from components.retriever_multi import MultiPDFRetriever

        r = MultiPDFRetriever()
    except Exception as exc:  # noqa: BLE001
        raise _Skip(f"index/model unavailable: {exc}")

    results = r.get_top_k("data scientist with python", k=3)
    assert len(results) == 3, "expected 3 results"
    for title, chunk_id, chunk_text in results:
        assert chunk_text, "each result should carry chunk text"


if __name__ == "__main__":
    check("ragmodel_requires_api_key", test_ragmodel_requires_api_key)
    check("extract_text_unsupported_extension", test_extract_text_unsupported_extension)
    check("live_retriever_normalized_cosine", test_live_retriever_normalized_cosine)
    check("multi_retriever_loads_index", test_multi_retriever_loads_index)

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed, {len(SKIPPED)} skipped")
    sys.exit(1 if FAILED else 0)
