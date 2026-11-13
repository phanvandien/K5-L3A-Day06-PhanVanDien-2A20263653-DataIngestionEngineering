"""Evaluate retrieved evidence and preserve per-query traces.

Only the automatic retrieval metrics live here. Chunk coherence, metadata
correctness and grounding quality are human labels and are never inferred
from string matching.
"""
import re
import unicodedata
from collections import defaultdict
from typing import Any

COLUMN_FIELDS = ("document_id", "source", "page")


def match_text(text: str) -> str:
    """Form used for evidence matching: NFC, single spaces, case-insensitive."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip().casefold()


def gold_options(question: dict[str, Any]) -> list[dict[str, Any]]:
    """The main gold plus any documented ``alternatives`` (same three keys)."""
    options = []
    for raw in [question, *question.get("alternatives", [])]:
        pages = raw["gold_page"]
        options.append({
            "source_id": raw["gold_source_id"],
            "pages": set(pages if isinstance(pages, list) else [pages]),
            "quote": match_text(raw["evidence_quote"]),
        })
    return options


def _field(hit: dict[str, Any], key: str) -> Any:
    return hit.get(key) if key in COLUMN_FIELDS else hit.get("metadata", {}).get(key)


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """A chunk is a hit when it is on a gold source/page AND contains the whole
    evidence quote of that same gold option. Hits must be ordered by rank."""
    options = gold_options(question)
    filters = question.get("filter") or {}
    judged = []
    for rank, hit in enumerate(hits, start=1):
        text = match_text(hit["text"])
        on_gold_page = any(
            hit["document_id"] == o["source_id"] and hit["page"] in o["pages"] for o in options
        )
        is_hit = any(
            hit["document_id"] == o["source_id"] and hit["page"] in o["pages"] and o["quote"] in text
            for o in options
        )
        judged.append({
            **hit, "rank": rank, "on_gold_page": on_gold_page, "is_hit": is_hit,
            "matches_filter": all(_field(hit, k) == v for k, v in filters.items()) if filters else None,
        })
    top3 = judged[:3]
    first = next((h["rank"] for h in top3 if h["is_hit"]), None)
    return {
        "id": question["id"],
        "split": question["split"],
        "type": question.get("type"),
        "question": question["question"],
        "expected_answer": question.get("expected_answer"),
        "gold_source_id": question["gold_source_id"],
        "gold_page": question["gold_page"],
        "evidence_quote": question["evidence_quote"],
        "filter": filters or None,
        "hit_at_1": bool(top3) and top3[0]["is_hit"],
        "hit_at_3": first is not None,
        "first_hit_rank": first,
        # Gold page retrieved but no single chunk holds the full quote: the
        # evidence was probably cut across chunks. Counts as a miss.
        "evidence_split_suspected": first is None and any(h["on_gold_page"] for h in top3),
        "filter_respected": all(h["matches_filter"] for h in top3) if filters and top3 else None,
        "hits": judged,
    }


def _ratio(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator}" if denominator else "N/A"


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Hit@1 and Evidence Hit@3 by strategy and dev/test split.

    Each trace needs a ``strategy`` key. Ratios are reported as "n/N" strings
    next to the raw counts; filter compliance is "N/A" without filter questions.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        groups[(trace["strategy"], trace["split"])].append(trace)
    summary: dict[str, Any] = {}
    for (strategy, split), items in sorted(groups.items()):
        with_filter = [t for t in items if t["filter"]]
        summary.setdefault(strategy, {})[split] = {
            "n_questions": len(items),
            "hit_at_1_count": sum(t["hit_at_1"] for t in items),
            "hit_at_3_count": sum(t["hit_at_3"] for t in items),
            "hit_at_1": _ratio(sum(t["hit_at_1"] for t in items), len(items)),
            "hit_at_3": _ratio(sum(t["hit_at_3"] for t in items), len(items)),
            "evidence_split_suspected": [t["id"] for t in items if t["evidence_split_suspected"]],
            "missed_at_3": [t["id"] for t in items if not t["hit_at_3"]],
            "filter_questions": len(with_filter),
            "filter_respected": _ratio(sum(bool(t["filter_respected"]) for t in with_filter),
                                       len(with_filter)),
        }
    return summary