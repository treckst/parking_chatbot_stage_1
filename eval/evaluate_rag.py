"""Minimal, reusable baseline for RAG retrieval quality and latency evaluation.

Run:
    python -m eval.evaluate_rag
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import time

EVAL_DIR = Path(__file__).parent
DATASET_PATH = EVAL_DIR / "dataset.json"
REPORT_PATH = EVAL_DIR / "report.md"
K_VALUES = (1, 4)


# --- Core retrieval & performance metrics (pure Python, easy to reuse) ---

def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant documents found in the top-k results."""
    if not relevant:
        return 0.0
    hits = set(retrieved[:k]) & relevant
    return len(hits) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of top-k retrieved documents that are relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / k


def mrr_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Reciprocal rank of the first relevant document within top-k."""
    for rank, doc in enumerate(retrieved[:k], start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile (p in 0..100)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * (p / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)


# --- Evaluation runner ---

def _extract_unique_sources(docs) -> list[str]:
    seen: list[str] = []
    for doc in docs:
        name = Path(doc.metadata.get("source", "")).name
        if name and name not in seen:
            seen.append(name)
    return seen


def evaluate_retriever(retriever, dataset: list[dict], redact_fn=None) -> list[dict]:
    """Run every evaluation query through the retriever and score it."""
    rows: list[dict] = []
    for item in dataset:
        query = item["query"]
        relevant = set(item["relevant_sources"])

        start = time.perf_counter()
        docs = retriever.invoke(query)
        if redact_fn is not None:
            for doc in docs:
                redact_fn(doc.page_content)
        latency_ms = (time.perf_counter() - start) * 1000.0

        retrieved = _extract_unique_sources(docs)
        entry = {
            "query": query,
            "relevant": sorted(relevant),
            "retrieved": retrieved,
            "latency_ms": latency_ms,
        }
        for k in K_VALUES:
            entry[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
            entry[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
            entry[f"mrr@{k}"] = mrr_at_k(retrieved, relevant, k)
        rows.append(entry)
    return rows


def summarize(rows: list[dict]) -> dict[str, float]:
    """Aggregate accuracy and latency metrics across all queries."""
    n = len(rows) or 1
    latencies = [r["latency_ms"] for r in rows]
    summary: dict[str, float] = {
        "queries": float(len(rows)),
        "latency_avg_ms": sum(latencies) / n,
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "latency_max_ms": max(latencies, default=0.0),
    }
    for k in K_VALUES:
        summary[f"recall@{k}"] = sum(r[f"recall@{k}"] for r in rows) / n
        summary[f"precision@{k}"] = sum(r[f"precision@{k}"] for r in rows) / n
        summary[f"mrr@{k}"] = sum(r[f"mrr@{k}"] for r in rows) / n
    return summary


def build_report(rows: list[dict], summary: dict[str, float]) -> str:
    """Format evaluation metrics into a concise Markdown report."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# RAG Evaluation Report",
        "",
        f"- **Generated:** {timestamp}",
        "- **Pipeline:** Hybrid retriever (Chroma dense `0.60` + BM25 sparse `0.40`) + Presidio PII redaction",
        f"- **Dataset:** `{DATASET_PATH.name}` ({int(summary['queries'])} queries)",
        "",
        "## Retrieval Quality",
        "",
        "| Metric | @1 | @4 |",
        "| --- | ---: | ---: |",
        f"| Recall | {summary['recall@1']:.3f} | {summary['recall@4']:.3f} |",
        f"| Precision | {summary['precision@1']:.3f} | {summary['precision@4']:.3f} |",
        f"| MRR | {summary['mrr@1']:.3f} | {summary['mrr@4']:.3f} |",
        "",
        "## Retrieval Latency",
        "",
        "| Avg (ms) | p50 (ms) | p95 (ms) | Max (ms) |",
        "| ---: | ---: | ---: | ---: |",
        (
            f"| {summary['latency_avg_ms']:.1f} | {summary['latency_p50_ms']:.1f} "
            f"| {summary['latency_p95_ms']:.1f} | {summary['latency_max_ms']:.1f} |"
        ),
        "",
        "## Per-Query Breakdown",
        "",
        "| # | Query | Expected | Top-4 Retrieved | R@4 | P@4 | Latency (ms) |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for i, r in enumerate(rows, start=1):
        expected = ", ".join(r["relevant"])
        top4 = ", ".join(r["retrieved"][:4])
        lines.append(
            f"| {i} | {r['query']} | `{expected}` | `{top4}` "
            f"| {r['recall@4']:.2f} | {r['precision@4']:.2f} | {r['latency_ms']:.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()

    from src.functions import redact_pii
    from src.retrieval_graph import retriever

    # Warm up local PII model once so cold-start doesn't skew the first query.
    redact_pii("warmup")

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    rows = evaluate_retriever(retriever, dataset, redact_fn=redact_pii)
    summary = summarize(rows)
    report = build_report(rows, summary)

    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
