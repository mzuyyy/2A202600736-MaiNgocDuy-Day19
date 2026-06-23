from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

import langextract as lx
from langextract import data as lx_data
from langextract.factory import ModelConfig

from graphrag.config import LlmSettings
from graphrag.models import Chunk, Triple, TripleBatch


EXTRACTION_PROMPT = """Extract important knowledge-graph triples from the text.
Use exact source evidence for each extraction.
Each extraction must have class "triple" and attributes:
subject, subject_type, relation, object, object_type.
Use concise named entities and snake_case relations.
Return at most 5 triples.
"""
AttrValue: TypeAlias = str | list[str] | None
Attributes: TypeAlias = Mapping[str, AttrValue]


EXAMPLES = [
    lx_data.ExampleData(
        text="Tesla builds electric vehicles in California.",
        extractions=[
            lx_data.Extraction(
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
        ],
    )
]


def parse_triples(raw: str) -> tuple[Triple, ...]:
    return TripleBatch.model_validate_json(raw).triples


def build_extraction_prompt(text: str) -> str:
    return f"{EXTRACTION_PROMPT}\n\nText:\n{text[:3500]}"


def extract_chunk(settings: LlmSettings, chunk: Chunk) -> tuple[Triple, ...]:
    result = lx.extract(
        text_or_documents=chunk.text,
        prompt_description=EXTRACTION_PROMPT,
        examples=EXAMPLES,
        config=ModelConfig(
            model_id=settings.model,
            provider="openai",
            provider_kwargs={
                "api_key": settings.api_key.get_secret_value(),
                "base_url": settings.base_url,
                "max_output_tokens": 4096,
            },
        ),
        max_char_buffer=3500,
        batch_length=1,
        max_workers=1,
        temperature=0.0,
        show_progress=False,
    )
    if isinstance(result, list):
        return tuple(
            triple
            for document in result
            for triple in _triples_from_extractions(document.extractions or [])
        )
    return _triples_from_extractions(result.extractions or [])


def _triples_from_extractions(
    extractions: list[lx_data.Extraction],
) -> tuple[Triple, ...]:
    return tuple(
        triple
        for extraction in extractions
        if extraction.extraction_class == "triple"
        for triple in (extraction_to_triple(extraction),)
        if triple is not None
    )


def extraction_to_triple(extraction: lx_data.Extraction) -> Triple | None:
    attributes = extraction.attributes or {}
    subject = _attribute(attributes, "subject")
    relation = _attribute(attributes, "relation")
    target = _attribute(attributes, "object")
    if not subject or not relation or not target:
        return None
    return Triple(
        subject=subject,
        subject_type=_attribute(attributes, "subject_type"),
        relation=relation,
        object=target,
        object_type=_attribute(attributes, "object_type"),
        evidence=extraction.extraction_text,
    )


def _attribute(attributes: Attributes, key: str) -> str:
    value = attributes.get(key, "")
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(item for item in value if item).strip()
    return value.strip()


class InvalidTripleResponseError(RuntimeError):
    def __init__(self, raw: str) -> None:
        self.raw = raw
        super().__init__("LLM returned invalid triple JSON")
