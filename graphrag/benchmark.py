from __future__ import annotations

import csv
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from graphrag.benchmark_questions import BENCHMARK_QUESTIONS
from graphrag.config import Settings
from graphrag.documents import build_chunks, load_documents
from graphrag.flat_rag import FlatRetriever
from graphrag.llm import LlmClient
from graphrag.models import RetrievalResult
from graphrag.networkx_store import NetworkXStore


@dataclass(frozen=True, slots=True)
class BenchmarkRow:
    index: int
    question: str
    expected_docs: str
    flat_docs: str
    graph_docs: str
    flat_recall: float
    graph_recall: float
    flat_retrieval_ms: float
    graph_retrieval_ms: float
    flat_generation_ms: float
    graph_generation_ms: float
    flat_context_chars: int
    graph_context_chars: int
    flat_answer: str
    graph_answer: str


@dataclass(frozen=True, slots=True)
class BenchmarkCost:
    chunk_count: int
    extraction_prompt_chars: int
    precomputed_triples: int
    triple_csv_bytes: int
    graph_import_ms: float
    graphml_bytes: int


def _recall(retrieved: tuple[str, ...], expected: tuple[str, ...]) -> float:
    return len(set(retrieved) & set(expected)) / len(expected)


def _answer(client: LlmClient, question: str, result: RetrievalResult) -> str:
    prompt = (
        "Answer concisely using only the context. If insufficient, say so. "
        "Cite document ids in square brackets.\n\n"
        f"Question: {question}\n\nContext:\n{result.context}"
    )
    return client.complete(prompt, max_tokens=250)


def run_benchmark(
    settings: Settings,
    triples_csv: Path,
    output_dir: Path,
) -> list[BenchmarkRow]:
    chunks = build_chunks(
        load_documents(settings.dataset_dir),
        settings.chunk_size,
        settings.chunk_overlap,
    )
    flat = FlatRetriever(chunks)
    started = time.perf_counter()
    graph = NetworkXStore.from_triples_csv(triples_csv, settings.graph_path)
    graph_import_ms = (time.perf_counter() - started) * 1000
    graph.save()
    cost = BenchmarkCost(
        chunk_count=len(chunks),
        extraction_prompt_chars=sum(min(len(chunk.text), 3500) for chunk in chunks),
        precomputed_triples=sum(
            1 for _ in csv.DictReader(triples_csv.open(encoding="utf-8-sig"))
        ),
        triple_csv_bytes=triples_csv.stat().st_size,
        graph_import_ms=graph_import_ms,
        graphml_bytes=settings.graph_path.stat().st_size,
    )
    rows: list[BenchmarkRow] = []
    with LlmClient(settings.llm) as client:
        for index, item in enumerate(BENCHMARK_QUESTIONS, start=1):
            started = time.perf_counter()
            flat_result = flat.retrieve(item.question)
            flat_retrieval_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            graph_result = graph.retrieve(item.question)
            graph_retrieval_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            flat_answer = _answer(client, item.question, flat_result)
            flat_generation_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            graph_answer = _answer(client, item.question, graph_result)
            graph_generation_ms = (time.perf_counter() - started) * 1000
            rows.append(
                BenchmarkRow(
                    index=index,
                    question=item.question,
                    expected_docs=", ".join(item.expected_docs),
                    flat_docs=", ".join(flat_result.doc_ids),
                    graph_docs=", ".join(graph_result.doc_ids),
                    flat_recall=_recall(flat_result.doc_ids, item.expected_docs),
                    graph_recall=_recall(graph_result.doc_ids, item.expected_docs),
                    flat_retrieval_ms=flat_retrieval_ms,
                    graph_retrieval_ms=graph_retrieval_ms,
                    flat_generation_ms=flat_generation_ms,
                    graph_generation_ms=graph_generation_ms,
                    flat_context_chars=len(flat_result.context),
                    graph_context_chars=len(graph_result.context),
                    flat_answer=flat_answer,
                    graph_answer=graph_answer,
                )
            )
            print(f"[{index}/20] benchmarked")
    _write_results(rows, output_dir, cost)
    return rows


def _write_results(
    rows: list[BenchmarkRow],
    output_dir: Path,
    cost: BenchmarkCost,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "benchmark_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=BenchmarkRow.__annotations__)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    flat_recall = statistics.mean(row.flat_recall for row in rows)
    graph_recall = statistics.mean(row.graph_recall for row in rows)
    flat_retrieval = statistics.mean(row.flat_retrieval_ms for row in rows)
    graph_retrieval = statistics.mean(row.graph_retrieval_ms for row in rows)
    flat_generation = statistics.mean(row.flat_generation_ms for row in rows)
    graph_generation = statistics.mean(row.graph_generation_ms for row in rows)
    flat_context = statistics.mean(row.flat_context_chars for row in rows)
    graph_context = statistics.mean(row.graph_context_chars for row in rows)
    flat_wins = sum(row.flat_recall > row.graph_recall for row in rows)
    graph_wins = sum(row.graph_recall > row.flat_recall for row in rows)
    ties = len(rows) - flat_wins - graph_wins
    report = f"""# Flat RAG vs GraphRAG benchmark

Corpus: 70 documents. Questions: 20. Same LLM for answer generation.

| Metric | Flat RAG | GraphRAG |
|---|---:|---:|
| Mean expected-document recall | {flat_recall:.1%} | {graph_recall:.1%} |
| Mean retrieval latency | {flat_retrieval:.2f} ms | {graph_retrieval:.2f} ms |
| Mean generation latency | {flat_generation:.2f} ms | {graph_generation:.2f} ms |
| Mean context size | {flat_context:.0f} chars | {graph_context:.0f} chars |

Recall wins: Flat {flat_wins}, GraphRAG {graph_wins}, ties {ties}.

## Graph construction cost

- Existing graph input: {cost.precomputed_triples:,} triples from `dataset/triples/graphrag_edges_neo4j.csv`
  ({cost.triple_csv_bytes:,} bytes).
- Loading triples into NetworkX took {cost.graph_import_ms:.2f} ms and wrote a
  {cost.graphml_bytes:,}-byte GraphML file. No database service is required.
- Building from raw text would require {cost.chunk_count:,} extraction calls,
  covering about {cost.extraction_prompt_chars:,} prompt characters before response tokens.
- GraphRAG adds extraction, normalization, persistence, and refresh cost. Flat RAG only
  chunks and indexes text, so it is materially cheaper to build and update.
- The graph cost is justified when questions require joining relationships across
  documents; for direct factual lookup, Flat RAG is usually the cheaper baseline.
"""
    (output_dir / "benchmark_report.md").write_text(report, encoding="utf-8")
