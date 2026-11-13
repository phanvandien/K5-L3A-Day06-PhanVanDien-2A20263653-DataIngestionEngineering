"""Day 06 pipeline: ingest -> chunk -> embed -> store -> retrieve -> evaluate.

Run from ``starter/``:

    python -m src.run --split dev    # tune on dev only; test is not touched
    python -m src.run --split all    # final run, also writes manual-audit sheets

Inputs (relative to the repo root):
    data/manifest.json          corpus manifest, see data/manifest.example.json
    benchmark/questions.json    questions with gold, see benchmark/questions.example.json
Outputs go to results/.
"""
import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from . import chunking
from .embeddings import DEFAULT_MODEL, MyEmbedder
from .evaluation import evaluate_query, gold_options, match_text, summarize
from .ingestion import load_pdf
from .models import Chunk, DocumentPage

ROOT = Path(__file__).resolve().parents[2]
STARTER = ROOT / "starter"
MANIFEST = ROOT / "data" / "manifest.json"
BENCHMARK = ROOT / "benchmark" / "questions.json"
RESULTS = ROOT / "results"

TOP_K = 3
MIN_CONTENT_PAGES = 30
MIN_PAGE_CHARS = 50  # a listed page with less normalized text is logged as empty

# A vs B changes only chunking. C searches B's index (same chunks, same vectors)
# and differs from B only by applying the question's metadata filter.
STRATEGIES: dict[str, dict[str, Any]] = {
    "A": {"index": "A", "chunking": "fixed_size", "size": 500, "overlap": 50, "use_filter": False},
    "B": {"index": "B", "chunking": "by_paragraph", "max_size": 800, "use_filter": False},
    "C": {"index": "B", "chunking": "by_paragraph", "max_size": 800, "use_filter": True},
}


def make_store(dimensions: int):
    """VECTOR_STORE=chroma (default, no Docker needed) or pgvector. No fallback
    between them: the chosen backend must work or the run fails."""
    kind = (os.getenv("VECTOR_STORE") or "chroma").lower()
    if kind == "chroma":
        from .chroma_store import ChromaStore
        return ChromaStore(STARTER / "chroma_db", dimensions)
    if kind == "pgvector":
        from .store import PgVectorStore
        return PgVectorStore(os.environ["DATABASE_URL"], dimensions)
    raise ValueError(f"unknown VECTOR_STORE={kind!r}")


def sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_pages(spec: Any) -> list[int]:
    """[3, 4, "10-12"] or "3-4,10-12" -> [3, 4, 10, 11, 12]."""
    parts = spec.split(",") if isinstance(spec, str) else spec
    pages: list[int] = []
    for part in parts:
        if isinstance(part, int):
            pages.append(part)
        else:
            first, _, last = part.strip().partition("-")
            pages.extend(range(int(first), int(last or first) + 1))
    return pages


def load_corpus(manifest: dict[str, Any]) -> tuple[list[DocumentPage], list[dict[str, Any]]]:
    """Return the usable content pages and a log entry for every listed page."""
    pages: list[DocumentPage] = []
    log: list[dict[str, Any]] = []
    for doc in manifest["documents"]:
        path = ROOT / doc["local_path"]
        if not path.is_file():
            raise FileNotFoundError(f"{doc['source_id']}: missing file {path}")
        by_number = {p.page: p for p in load_pdf(path, doc["source_id"], doc["source_path_or_url"])}
        sections = [(s["section"], set(parse_pages(s["pages"]))) for s in doc.get("sections", [])]
        for number in parse_pages(doc["content_pages"]):
            page = by_number.get(number)
            entry = {"document_id": doc["source_id"], "page": number, "status": "ok", "chars": 0}
            if page is None:
                entry["status"] = "missing_page"
            elif page.metadata.get("extraction_error"):
                entry.update(status="extraction_error", detail=page.metadata["extraction_error"])
            elif len(page.text) < MIN_PAGE_CHARS:
                entry.update(status="empty_or_too_short", chars=len(page.text))
            else:
                entry["chars"] = len(page.text)
                # Metadata comes only from the manifest: document-level fields
                # plus the section whose page range contains this page.
                page.metadata = dict(doc.get("metadata", {}))
                for name, numbers in sections:
                    if number in numbers:
                        page.metadata["section"] = name
                pages.append(page)
            log.append(entry)
    return pages, log


def load_benchmark() -> list[dict[str, Any]]:
    data = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    return data["questions"] if isinstance(data, dict) else data


def validate_benchmark(questions: list[dict[str, Any]], pages: list[DocumentPage]) -> list[str]:
    """Size/split rules from BENCHMARK.md and: every evidence quote must exist
    on its gold page after normalization."""
    problems: list[str] = []
    ids = [q["id"] for q in questions]
    if len(set(ids)) != len(ids):
        problems.append("duplicate question ids")
    dev = [q for q in questions if q["split"] == "dev"]
    test = [q for q in questions if q["split"] == "test"]
    filtered = [q for q in questions if q.get("filter")]
    if len(questions) < 20 or len(dev) < 15 or len(test) < 5:
        problems.append(f"need >=20 questions, >=15 dev, >=5 test; got {len(questions)}/{len(dev)}/{len(test)}")
    if len(dev) + len(test) != len(questions):
        problems.append("every question needs split 'dev' or 'test'")
    if len(filtered) < 3 or not any(q["split"] == "test" for q in filtered):
        problems.append("need >=3 questions with a filter, at least 1 of them in test")
    page_text = {(p.document_id, p.page): match_text(p.text) for p in pages}
    filter_values: dict[str, set] = {}
    for page in pages:
        for key, value in page.metadata.items():
            filter_values.setdefault(key, set()).add(value)
    for q in questions:
        for option in gold_options(q):
            if not any(option["quote"] in page_text.get((option["source_id"], n), "") for n in option["pages"]):
                problems.append(f"{q['id']}: evidence_quote not found on {option['source_id']} page(s) {sorted(option['pages'])}")
        for key, value in (q.get("filter") or {}).items():
            if key not in ("document_id", "source", "page") and value not in filter_values.get(key, set()):
                problems.append(f"{q['id']}: filter {key}={value!r} matches no page metadata")
    return problems


def build_chunks(config: dict[str, Any], pages: list[DocumentPage]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        if config["chunking"] == "fixed_size":
            chunks.extend(chunking.fixed_size(page, config["size"], config["overlap"]))
        else:
            chunks.extend(chunking.by_paragraph(page, config["max_size"]))
    return chunks


def chunk_stats(chunks: list[Chunk]) -> dict[str, Any]:
    lengths = [len(c.text) for c in chunks]
    return {
        "chunk_count": len(chunks), "total_chars": sum(lengths), "min_chars": min(lengths),
        "median_chars": statistics.median(lengths), "mean_chars": round(statistics.mean(lengths), 1),
        "max_chars": max(lengths),
    }


def write_csv_once(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    """Audit sheets hold human labels, so an existing file is never overwritten."""
    if path.exists():
        print(f"  kept existing {path.relative_to(ROOT)} (delete it to regenerate)")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  wrote {path.relative_to(ROOT)}")


def write_audit_sheets(pages: list[DocumentPage], index_chunks: dict[str, list[Chunk]],
                       traces: list[dict[str, Any]]) -> None:
    """Empty sheets for the three manual assessments; labels are filled by hand."""
    # Coherence: the first 10 content pages in sorted order, position = middle
    # of the page. Chosen by rule, before any score is seen.
    rows = []
    for page in sorted(pages, key=lambda p: (p.document_id, p.page))[:10]:
        middle = len(page.text) // 2
        for name, config in STRATEGIES.items():
            chunk = next(
                (c for c in index_chunks[config["index"]]
                 if c.document_id == page.document_id and c.page == page.page
                 and c.metadata["char_start"] <= middle < c.metadata["char_end"]), None)
            rows.append([f"{page.document_id}:p{page.page}", middle, name,
                         chunk.chunk_id if chunk else "", chunk.text if chunk else "", "", ""])
    write_csv_once(RESULTS / "audit_coherence.csv",
                   ["position", "char_pos", "strategy", "chunk_id", "chunk_text",
                    "coherence_0_2", "reason"], rows)

    test = [t for t in traces if t["split"] == "test"]
    write_csv_once(
        RESULTS / "audit_metadata.csv",
        ["strategy", "question_id", "filter", "rank", "chunk_id", "source", "page", "metadata",
         "matches_filter_auto", "metadata_complete_and_correct_0_1", "error_note"],
        [[t["strategy"], t["id"], json.dumps(t["filter"], ensure_ascii=False), h["rank"],
          h["chunk_id"], h["source"], h["page"], json.dumps(h["metadata"], ensure_ascii=False),
          h["matches_filter"], "", ""] for t in test for h in t["hits"]])
    write_csv_once(
        RESULTS / "audit_grounding.csv",
        ["strategy", "question_id", "question", "expected_answer", "top3_chunk_ids", "top3_text",
         "evidence_hit_at_3_auto", "grounding_0_2", "reason"],
        [[t["strategy"], t["id"], t["question"], t["expected_answer"],
          " | ".join(h["chunk_id"] for h in t["hits"]),
          "\n-----\n".join(h["text"] for h in t["hits"]), t["hit_at_3"], "", ""] for t in test])


def package_versions() -> dict[str, str]:
    found = {}
    for name in ("sentence-transformers", "transformers", "torch", "torchvision", "pillow",
                 "chromadb", "pypdf"):
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = "not installed"
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--split", choices=["dev", "all"], default="dev",
                        help="dev: evaluate dev questions only (default). all: dev + test + audit sheets.")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(STARTER / ".env")
    RESULTS.mkdir(exist_ok=True)

    # 1. Corpus and benchmark ------------------------------------------------
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    pages, ingest_log = load_corpus(manifest)
    write_json(RESULTS / "ingest_log.json", ingest_log)
    bad = [e for e in ingest_log if e["status"] != "ok"]
    print(f"Corpus: {len(pages)} content pages, {sum(len(p.text) for p in pages)} chars, "
          f"{len(bad)} listed pages skipped (see results/ingest_log.json)")
    if len(pages) < MIN_CONTENT_PAGES:
        sys.exit(f"Only {len(pages)} usable content pages; the lab needs {MIN_CONTENT_PAGES}.")

    questions = load_benchmark()
    problems = validate_benchmark(questions, pages)
    write_json(RESULTS / "benchmark_validation.json", {"questions": len(questions), "problems": problems})
    if problems:
        sys.exit("Benchmark invalid, fix before running:\n- " + "\n- ".join(problems))
    selected = [q for q in questions if args.split == "all" or q["split"] == "dev"]
    print(f"Benchmark: {len(questions)} questions valid; evaluating {len(selected)} ({args.split})")

    # 2. Model and store (any failure stops the run; no mock fallback) --------
    embedder = MyEmbedder(os.getenv("EMBEDDING_MODEL") or DEFAULT_MODEL)
    store = make_store(embedder.dimensions)
    store.create_schema()

    # 3. Chunk, embed, store; re-ingest once to prove there are no duplicates --
    index_chunks: dict[str, list[Chunk]] = {}
    index_stats: dict[str, dict[str, Any]] = {}
    for name in sorted({config["index"] for config in STRATEGIES.values()}):
        started = time.perf_counter()
        chunks = build_chunks(STRATEGIES[name], pages)
        chunk_seconds = time.perf_counter() - started
        started = time.perf_counter()
        vectors = embedder.embed_documents([c.text for c in chunks])
        embed_seconds = time.perf_counter() - started
        started = time.perf_counter()
        store.upsert(name, chunks, vectors)
        store_seconds = time.perf_counter() - started
        count_first = store.count(name)
        store.upsert(name, chunks, vectors)
        count_second = store.count(name)
        index_chunks[name] = chunks
        index_stats[name] = {
            **chunk_stats(chunks),
            "chunk_seconds": round(chunk_seconds, 3), "embed_seconds": round(embed_seconds, 3),
            "store_seconds": round(store_seconds, 3),
            "rows_after_ingest": count_first, "rows_after_reingest": count_second,
            "reingest_no_duplicates": count_first == count_second == len(chunks),
        }
        print(f"Index {name}: {index_stats[name]}")

    # 4. Retrieve top-3 for every strategy with the same query vectors --------
    query_vectors = {q["id"]: embedder.embed_query(q["question"]) for q in selected}
    traces: list[dict[str, Any]] = []
    for name, config in STRATEGIES.items():
        with (RESULTS / f"trace_{name}.jsonl").open("w", encoding="utf-8") as handle:
            for q in selected:
                filters = q.get("filter") if config["use_filter"] else None
                hits = store.search(config["index"], query_vectors[q["id"]], TOP_K, filters or None)
                trace = {"strategy": name, "filter_applied": filters or None, **evaluate_query(q, hits)}
                traces.append(trace)
                handle.write(json.dumps(trace, ensure_ascii=False) + "\n")

    # 5. Metrics, run info, audit sheets -------------------------------------
    summary = summarize(traces)
    write_json(RESULTS / "metrics.json", summary)
    write_json(RESULTS / "run_info.json", {
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split_evaluated": args.split,
        "python": platform.python_version(), "packages": package_versions(),
        "embedding": {"provider": "local sentence-transformers", "model": embedder.model_name,
                      "dimensions": embedder.dimensions, "normalized": True,
                      "query_prefix": embedder.query_prefix, "document_prefix": embedder.document_prefix,
                      "cost_usd": 0},
        "vector_store": store.describe(),
        "top_k": TOP_K, "strategies": STRATEGIES, "indexes": index_stats,
        "content_pages": len(pages),
        "content_chars": sum(len(p.text) for p in pages),
        "source_files_sha256": {d["source_id"]: sha256(ROOT / d["local_path"])
                                for d in manifest["documents"]},
    })
    lines = ["| Strategy | Split | Hit@1 | Evidence Hit@3 | Chunk count | Ingest s | Filter respected |",
             "|---|---|---|---|---|---|---|"]
    for name, splits in summary.items():
        stats = index_stats[STRATEGIES[name]["index"]]
        seconds = round(stats["chunk_seconds"] + stats["embed_seconds"] + stats["store_seconds"], 2)
        for split, m in splits.items():
            lines.append(f"| {name} | {split} | {m['hit_at_1']} | {m['hit_at_3']} | "
                         f"{stats['chunk_count']} | {seconds} | {m['filter_respected']} |")
    (RESULTS / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if args.split == "all":
        print("Manual audit sheets:")
        write_audit_sheets(pages, index_chunks, traces)


if __name__ == "__main__":
    main()