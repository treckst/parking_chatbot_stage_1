from types import SimpleNamespace

import pytest

from eval.evaluate_rag import (
    build_report,
    evaluate_retriever,
    mrr_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    summarize,
)


def test_recall_and_precision_at_k():
    retrieved = ["a.pdf", "b.pdf", "c.pdf", "d.pdf"]
    relevant = {"b.pdf", "d.pdf"}

    assert recall_at_k(retrieved, relevant, k=1) == pytest.approx(0.0)
    assert recall_at_k(retrieved, relevant, k=2) == pytest.approx(0.5)
    assert recall_at_k(retrieved, relevant, k=4) == pytest.approx(1.0)

    assert precision_at_k(retrieved, relevant, k=1) == pytest.approx(0.0)
    assert precision_at_k(retrieved, relevant, k=2) == pytest.approx(0.5)
    assert precision_at_k(retrieved, relevant, k=4) == pytest.approx(0.5)


def test_mrr_at_k():
    assert mrr_at_k(["a.pdf", "b.pdf", "c.pdf"], {"a.pdf"}, k=3) == pytest.approx(1.0)
    assert mrr_at_k(["a.pdf", "b.pdf", "c.pdf"], {"b.pdf"}, k=3) == pytest.approx(0.5)
    assert mrr_at_k(["a.pdf", "b.pdf", "c.pdf"], {"z.pdf"}, k=3) == pytest.approx(0.0)


def test_percentile():
    assert percentile([10.0, 20.0, 30.0, 40.0, 50.0], 50) == pytest.approx(30.0)
    assert percentile([10.0, 20.0, 30.0, 40.0, 50.0], 95) == pytest.approx(48.0)
    assert percentile([], 95) == pytest.approx(0.0)


def test_evaluate_retriever_and_report():
    class FakeRetriever:
        def invoke(self, _query: str):
            return [
                SimpleNamespace(page_content="doc1", metadata={"source": "pdfs/a.pdf"}),
                SimpleNamespace(page_content="doc2", metadata={"source": "pdfs/b.pdf"}),
            ]

    dataset = [{"query": "test question", "relevant_sources": ["a.pdf"]}]
    rows = evaluate_retriever(FakeRetriever(), dataset, redact_fn=lambda text: text)
    summary = summarize(rows)
    report = build_report(rows, summary)

    assert summary["recall@1"] == pytest.approx(1.0)
    assert summary["precision@4"] == pytest.approx(0.25)
    assert summary["mrr@4"] == pytest.approx(1.0)
    assert "# RAG Evaluation Report" in report
    assert "test question" in report
