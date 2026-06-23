from pathlib import Path

from graphrag.benchmark import BENCHMARK_QUESTIONS
from graphrag.documents import build_chunks
from graphrag.flat_rag import FlatRetriever
from graphrag.models import Chunk, Document, Triple
from graphrag.networkx_store import NetworkXStore
from graphrag.visualize import draw_graph


def test_benchmark_has_twenty_questions() -> None:
    # Given / When
    questions = BENCHMARK_QUESTIONS

    # Then
    assert len(questions) == 20
    assert len({item.question for item in questions}) == 20
    assert sum(len(item.expected_docs) for item in questions) / len(questions) >= 4


def test_flat_retriever_returns_relevant_document() -> None:
    # Given
    chunks = build_chunks(
        [
            Document("ev.txt", "EV charging infrastructure supports adoption."),
            Document("other.txt", "Quarterly software revenue increased."),
        ],
        size=100,
        overlap=0,
    )

    # When
    result = FlatRetriever(chunks).retrieve("electric vehicle charging", limit=1)

    # Then
    assert result.doc_ids == ("ev.txt",)
    assert "charging infrastructure" in result.context


def test_benchmark_expected_documents_exist() -> None:
    # Given
    dataset_dir = Path("dataset")
    existing = {path.name for path in dataset_dir.glob("*.txt")}

    # When
    expected = {
        doc_id
        for question in BENCHMARK_QUESTIONS
        for doc_id in question.expected_docs
    }

    # Then
    assert expected <= existing


def test_draw_graph_writes_png_from_graphml(tmp_path: Path) -> None:
    # Given
    graph_path = tmp_path / "graph.graphml"
    output_path = tmp_path / "graph.png"
    chunk = Chunk(chunk_id="doc_1.txt::ch0", doc_id="doc_1.txt", text="Tesla")
    store = NetworkXStore(graph_path)
    store.upsert(
        chunk,
        (
            Triple(subject="Tesla", relation="builds", object="EVs"),
            Triple(subject="EVs", relation="need", object="charging"),
        ),
    )
    store.save()

    # When
    written = draw_graph(graph_path, output_path, max_nodes=10)

    # Then
    assert written == output_path
    assert output_path.read_bytes().startswith(b"\x89PNG")
