from __future__ import annotations

import re
import csv
from collections.abc import Iterable
from pathlib import Path

import networkx as nx

from graphrag.models import Chunk, GraphFact, RetrievalResult, Triple


class NetworkXStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._graph = (
            nx.read_graphml(path, force_multigraph=True)
            if path.exists()
            else nx.MultiDiGraph()
        )

    def upsert(self, chunk: Chunk, triples: Iterable[Triple]) -> int:
        count = 0
        for index, triple in enumerate(triples):
            source = triple.subject.strip()
            target = triple.object.strip()
            self._graph.add_node(source, type=triple.subject_type.strip())
            self._graph.add_node(target, type=triple.object_type.strip())
            self._graph.add_edge(
                source,
                target,
                key=f"{chunk.chunk_id}:{index}",
                relation=triple.relation.strip(),
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                evidence=triple.evidence.strip(),
            )
            count += 1
        return count

    def clear(self) -> None:
        self._graph.clear()

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(self._graph, self._path)

    @classmethod
    def from_triples_csv(cls, path: Path, graph_path: Path) -> NetworkXStore:
        store = cls(graph_path)
        store._graph.clear()
        with path.open(encoding="utf-8-sig", newline="") as csv_file:
            for index, row in enumerate(csv.DictReader(csv_file)):
                source = row[":START_ID"].strip()
                target = row[":END_ID"].strip()
                store._graph.add_edge(
                    source,
                    target,
                    key=str(index),
                    relation=row["predicate"].strip(),
                    doc_id=row["source_doc"].strip(),
                    evidence=row["evidence"].strip(),
                )
        return store

    def retrieve(self, question: str, *, limit: int = 20) -> RetrievalResult:
        tokens = {
            token.lower()
            for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}", question)
        }
        ranked: list[tuple[int, str, str, dict[str, str]]] = []
        for source, target, data in self._graph.edges(data=True):
            searchable = " ".join(
                (
                    str(source),
                    str(target),
                    str(data.get("relation", "")),
                    str(data.get("evidence", "")),
                )
            ).lower()
            ranked.append(
                (
                    sum(token in searchable for token in tokens),
                    str(source),
                    str(target),
                    data,
                )
            )
        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = [item for item in ranked[:limit] if item[0] > 0]
        return RetrievalResult(
            context="\n".join(
                f"({source}) -[{data.get('relation', '')}]-> ({target}) "
                f"[{data.get('doc_id', '')}] Evidence: {data.get('evidence', '')}"
                for _, source, target, data in selected
            ),
            doc_ids=tuple(
                dict.fromkeys(str(data.get("doc_id", "")) for *_, data in selected)
            ),
        )

    def facts_for_question(
        self,
        question: str,
        *,
        limit: int = 40,
    ) -> list[GraphFact]:
        tokens = {
            token.lower()
            for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}", question)
        }
        facts: list[GraphFact] = []
        for source, target, data in self._graph.edges(data=True):
            searchable = " ".join(
                (str(source), str(target), str(data.get("relation", "")))
            ).lower()
            if tokens and not any(token in searchable for token in tokens):
                continue
            facts.append(
                GraphFact(
                    source=str(source),
                    relation=str(data.get("relation", "")),
                    target=str(target),
                    doc_id=str(data.get("doc_id", "")),
                )
            )
            if len(facts) >= limit:
                break
        return facts
