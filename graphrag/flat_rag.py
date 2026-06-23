from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from graphrag.models import Chunk, RetrievalResult


class FlatRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(
            chunk.text for chunk in chunks
        )

    def retrieve(self, question: str, *, limit: int = 6) -> RetrievalResult:
        query = self._vectorizer.transform([question])
        scores = cosine_similarity(query, self._matrix).ravel()
        indices = scores.argsort()[::-1][:limit]
        selected = [self._chunks[int(index)] for index in indices]
        return RetrievalResult(
            context="\n\n".join(
                f"[{chunk.doc_id}] {chunk.text}" for chunk in selected
            ),
            doc_ids=tuple(dict.fromkeys(chunk.doc_id for chunk in selected)),
        )
