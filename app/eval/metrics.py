"""Чистая математика retrieval-метрик, вынесенная из evaluate_retrieval.py,
чтобы её же переиспользовал evaluate_routes.py (сравнение vector/graph/agentic)
без дублирования формул."""

from typing import List


def normalize_text(text: str) -> str:
    return text.lower().strip() if text else ""


def compute_retrieval_metrics(
    retrieved_titles: List[str],
    retrieved_contents: List[str],
    expected_titles: List[str],
    expected_quotes: List[str],
    k_values: List[int],
) -> dict:
    metrics = {}

    # ── Метрики по заголовкам (для всех K) ──
    for k in k_values:
        top_k_titles = retrieved_titles[:k]
        hit = any(
            any(normalize_text(t) == normalize_text(et) for et in expected_titles)
            for t in top_k_titles if t != "UNKNOWN"
        )
        found_count = sum(
            1 for t in top_k_titles if t != "UNKNOWN" and
            any(normalize_text(t) == normalize_text(et) for et in expected_titles)
        )
        mrr = next(
            (1.0 / (i + 1) for i, t in enumerate(top_k_titles) if t != "UNKNOWN" and
             any(normalize_text(t) == normalize_text(et) for et in expected_titles)),
            0.0,
        )

        metrics[f"title_hit@{k}"] = int(hit)
        metrics[f"title_recall@{k}"] = found_count / len(expected_titles) if expected_titles else 0.0
        metrics[f"title_precision@{k}"] = found_count / k
        metrics[f"title_mrr@{k}"] = mrr

    # ── Метрики по цитатам (для всех K) ──
    norm_expected_quotes = [normalize_text(q) for q in expected_quotes]
    norm_retrieved_contents = [normalize_text(c) for c in retrieved_contents]

    for k in k_values:
        top_k_contents = norm_retrieved_contents[:k]
        if norm_expected_quotes:
            found_quotes = set()
            for eq in norm_expected_quotes:
                if any(eq in chunk for chunk in top_k_contents):
                    found_quotes.add(eq)

            relevant_chunks = sum(
                1 for chunk in top_k_contents
                if any(eq in chunk for eq in norm_expected_quotes)
            )

            mrr_cit = next(
                (1.0 / (i + 1) for i, chunk in enumerate(top_k_contents)
                 if any(eq in chunk for eq in norm_expected_quotes)),
                0.0,
            )

            metrics[f"citation_recall@{k}"] = len(found_quotes) / len(norm_expected_quotes)
            metrics[f"citation_hit@{k}"] = 1 if found_quotes else 0
            metrics[f"citation_precision@{k}"] = relevant_chunks / k
            metrics[f"citation_mrr@{k}"] = mrr_cit
        else:
            metrics.update({
                f"citation_recall@{k}": 0.0, f"citation_hit@{k}": 0,
                f"citation_precision@{k}": 0.0, f"citation_mrr@{k}": 0.0,
            })

    return metrics
