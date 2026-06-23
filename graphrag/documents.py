from __future__ import annotations

import re
from pathlib import Path

from graphrag.models import Chunk, Document


def _natural_key(path: Path) -> tuple[str | int, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    )


def load_documents(directory: Path, max_docs: int | None = None) -> list[Document]:
    paths = sorted(directory.glob("*.txt"), key=_natural_key)
    if max_docs is not None:
        paths = paths[:max_docs]
    return [
        Document(
            doc_id=path.name,
            text=path.read_text(encoding="utf-8", errors="ignore"),
        )
        for path in paths
    ]


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0:
        raise InvalidChunkSettingsError(size=size, overlap=overlap)
    if overlap < 0 or overlap >= size:
        raise InvalidChunkSettingsError(size=size, overlap=overlap)

    clean_text = text.strip()
    chunks: list[str] = []
    start = 0
    while start < len(clean_text):
        end = min(start + size, len(clean_text))
        chunks.append(clean_text[start:end])
        if end == len(clean_text):
            break
        start = end - overlap
    return chunks


def build_chunks(
    documents: list[Document],
    size: int,
    overlap: int,
) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"{document.doc_id}::ch{index}",
            doc_id=document.doc_id,
            text=text,
        )
        for document in documents
        for index, text in enumerate(chunk_text(document.text, size, overlap))
    ]


class InvalidChunkSettingsError(ValueError):
    def __init__(self, size: int, overlap: int) -> None:
        self.size = size
        self.overlap = overlap
        super().__init__(
            f"chunk overlap must satisfy 0 <= overlap < size; "
            f"received size={size}, overlap={overlap}"
        )
