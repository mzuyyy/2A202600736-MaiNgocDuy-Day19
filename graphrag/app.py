from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

from graphrag.config import Settings
from graphrag.documents import build_chunks, load_documents
from graphrag.extraction import InvalidTripleResponseError, extract_chunk
from graphrag.llm import LlmClient
from graphrag.networkx_store import NetworkXStore

TRIPLE_FIELDS = (
    ":START_ID",
    "predicate",
    ":END_ID",
    "subject_type",
    "object_type",
    "source_doc",
    "source_chunk",
    "evidence",
)


def check_runtime(settings: Settings) -> tuple[int, str]:
    documents = load_documents(settings.dataset_dir)
    if not documents:
        return 1, f"No .txt files found in {settings.dataset_dir}"

    with LlmClient(settings.llm) as client:
        response = client.complete("Reply with exactly OK.", max_tokens=8)

    return 0, (
        f"dataset={len(documents)} documents | graph={settings.graph_path} | "
        f"llm={response[:40]}"
    )


def ingest(
    settings: Settings,
    *,
    max_docs: int | None,
    progress: Callable[[str], None] = print,
) -> tuple[int, int, Path]:
    documents = load_documents(settings.dataset_dir, max_docs=max_docs)
    chunks = build_chunks(
        documents,
        size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    stored = 0
    failed = 0
    store = NetworkXStore(settings.graph_path)
    store.clear()
    settings.triples_path.parent.mkdir(parents=True, exist_ok=True)
    with settings.triples_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=TRIPLE_FIELDS)
        writer.writeheader()
        for index, chunk in enumerate(chunks, start=1):
            try:
                triples = extract_chunk(settings.llm, chunk)
            except InvalidTripleResponseError:
                failed += 1
                progress(f"[{index}/{len(chunks)}] invalid extraction: {chunk.chunk_id}")
                continue
            stored += store.upsert(chunk, triples)
            for triple in triples:
                writer.writerow(
                    {
                        ":START_ID": triple.subject,
                        "predicate": triple.relation,
                        ":END_ID": triple.object,
                        "subject_type": triple.subject_type,
                        "object_type": triple.object_type,
                        "source_doc": chunk.doc_id,
                        "source_chunk": chunk.chunk_id,
                        "evidence": triple.evidence,
                    }
                )
            progress(
                f"[{index}/{len(chunks)}] {chunk.chunk_id}: {len(triples)} triples"
            )
    store.save()
    return stored, failed, settings.triples_path


def answer(settings: Settings, question: str) -> str:
    facts = NetworkXStore(settings.graph_path).facts_for_question(question)

    if not facts:
        return "Không tìm thấy dữ kiện liên quan trong graph local."

    context = "\n".join(
        f"({fact.source}) -[{fact.relation}]-> ({fact.target}) [{fact.doc_id}]"
        for fact in facts
    )
    prompt = (
        "Answer the question only from the graph facts below. "
        "If facts are insufficient, say so. Include relevant document ids.\n\n"
        f"Question: {question}\n\nGraph facts:\n{context}"
    )
    with LlmClient(settings.llm) as client:
        return client.complete(prompt, max_tokens=500)
