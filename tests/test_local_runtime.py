from pathlib import Path

import pytest

from graphrag.config import LlmSettings, Settings
from graphrag.documents import chunk_text, load_documents
from graphrag.app import ingest
from graphrag.extraction import extraction_to_triple
from graphrag.models import Chunk, Triple
from graphrag.networkx_store import NetworkXStore
from graphrag_local import run
from langextract import data as lx_data
from pydantic import SecretStr


def test_settings_loads_local_graph_and_llm_from_env_file(tmp_path: Path) -> None:
    # Given
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            (
                "BTL_API_KEY=test-key",
                "BTL_API_BASE_URL=https://example.test/v1",
            )
        ),
        encoding="utf-8",
    )

    # When
    settings = Settings.from_env(env_file=env_file, project_dir=tmp_path)

    # Then
    assert settings.llm.base_url == "https://example.test/v1"
    assert settings.dataset_dir == tmp_path / "dataset"
    assert settings.graph_path == tmp_path / "graph.graphml"
    assert settings.triples_path == tmp_path / "dataset/triples/graphrag_edges_langextract.csv"


def test_load_documents_reads_local_dataset_in_natural_order(tmp_path: Path) -> None:
    # Given
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "doc_10.txt").write_text("ten", encoding="utf-8")
    (dataset_dir / "doc_2.txt").write_text("two", encoding="utf-8")

    # When
    documents = load_documents(dataset_dir)

    # Then
    assert [document.doc_id for document in documents] == ["doc_2.txt", "doc_10.txt"]


def test_chunk_text_rejects_overlap_that_cannot_advance() -> None:
    # Given
    text = "abcdef"

    # When / Then
    with pytest.raises(ValueError, match="overlap"):
        chunk_text(text, size=3, overlap=3)


def test_cli_reports_connection_error_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given
    def fail_check(_settings: Settings) -> tuple[int, str]:
        raise ConnectionError("service unavailable")

    monkeypatch.setattr("graphrag_local.check_runtime", fail_check)

    # When
    exit_code = run(["check"])

    # Then
    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err == "error: service unavailable\n"


def test_extraction_to_triple_reads_langextract_attributes() -> None:
    # Given
    extraction = lx_data.Extraction(
        extraction_class="triple",
        extraction_text="Tesla builds electric vehicles",
        attributes={
            "subject": "Tesla",
            "subject_type": "company",
            "relation": "builds",
            "object": "electric vehicles",
            "object_type": "product_category",
        },
    )

    # When
    triple = extraction_to_triple(extraction)

    # Then
    assert triple == Triple(
        subject="Tesla",
        subject_type="company",
        relation="builds",
        object="electric vehicles",
        object_type="product_category",
        evidence="Tesla builds electric vehicles",
    )


def test_extraction_to_triple_skips_missing_required_attribute() -> None:
    # Given
    extraction = lx_data.Extraction(
        extraction_class="triple",
        extraction_text="Tesla builds electric vehicles",
        attributes={
            "subject": "Tesla",
            "relation": "builds",
        },
    )

    # When
    triple = extraction_to_triple(extraction)

    # Then
    assert triple is None


def test_networkx_store_persists_and_finds_facts(tmp_path: Path) -> None:
    # Given
    graph_path = tmp_path / "graph.graphml"
    chunk = Chunk(chunk_id="doc_1.txt::ch0", doc_id="doc_1.txt", text="Tesla")
    triple = Triple(
        subject="Tesla",
        subject_type="company",
        relation="operates_in",
        object="EV market",
        object_type="sector",
    )

    # When
    store = NetworkXStore(graph_path)
    store.upsert(chunk, (triple,))
    store.save()
    facts = NetworkXStore(graph_path).facts_for_question("Tesla market")

    # Then
    assert len(facts) == 1
    assert facts[0].source == "Tesla"
    assert facts[0].target == "EV market"


def test_networkx_store_keeps_multiple_relations_from_same_chunk(
    tmp_path: Path,
) -> None:
    # Given
    graph_path = tmp_path / "graph.graphml"
    chunk = Chunk(chunk_id="doc_1.txt::ch0", doc_id="doc_1.txt", text="Tesla")
    triples = (
        Triple(subject="Tesla", relation="builds", object="Model 3"),
        Triple(subject="Tesla", relation="sells", object="Model 3"),
    )

    # When
    store = NetworkXStore(graph_path)
    store.upsert(chunk, triples)
    store.save()
    facts = NetworkXStore(graph_path).facts_for_question("Tesla")

    # Then
    assert {fact.relation for fact in facts} == {"builds", "sells"}


def test_ingest_rebuilds_graph_and_writes_langextract_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "doc_1.txt").write_text("Tesla builds EVs.", encoding="utf-8")
    graph_path = tmp_path / "graph.graphml"
    triples_path = tmp_path / "dataset/triples/langextract.csv"
    old_chunk = Chunk(chunk_id="old::ch0", doc_id="old", text="Old")
    old_triple = Triple(subject="Old", relation="mentions", object="Stale")
    stale_store = NetworkXStore(graph_path)
    stale_store.upsert(old_chunk, (old_triple,))
    stale_store.save()

    settings = Settings(
        llm=LlmSettings(
            api_key=SecretStr("test-key"),
            base_url="https://example.test/v1",
        ),
        dataset_dir=dataset_dir,
        graph_path=graph_path,
        triples_path=triples_path,
        chunk_size=1000,
        chunk_overlap=0,
    )

    def fake_extract(_settings: LlmSettings, _chunk: Chunk) -> tuple[Triple, ...]:
        return (
            Triple(
                subject="Tesla",
                relation="builds",
                object="EVs",
                evidence="Tesla builds EVs",
            ),
        )

    monkeypatch.setattr("graphrag.app.extract_chunk", fake_extract)

    # When
    stored, failed, written_path = ingest(settings, max_docs=None, progress=lambda _: None)
    facts = NetworkXStore(graph_path).facts_for_question("Old Tesla EVs")

    # Then
    assert (stored, failed, written_path) == (1, 0, triples_path)
    assert [fact.source for fact in facts] == ["Tesla"]
    assert "Tesla,builds,EVs" in triples_path.read_text(encoding="utf-8")
