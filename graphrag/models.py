from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    text: str


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


class Triple(BaseModel):
    model_config = ConfigDict(frozen=True)

    subject: str = Field(min_length=1)
    subject_type: str = ""
    relation: str = Field(min_length=1)
    object: str = Field(min_length=1)
    object_type: str = ""
    evidence: str = ""


class TripleBatch(BaseModel):
    model_config = ConfigDict(frozen=True)

    triples: tuple[Triple, ...]


@dataclass(frozen=True, slots=True)
class GraphFact:
    source: str
    relation: str
    target: str
    doc_id: str


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    context: str
    doc_ids: tuple[str, ...]
