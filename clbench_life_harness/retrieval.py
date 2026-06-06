from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .chunking import Chunk

TOKEN_RE = re.compile(r"[A-Za-z0-9_$#@][A-Za-z0-9_$#@'./:-]*")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "can",
    "could",
    "do",
    "for",
    "from",
    "give",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "list",
    "make",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "should",
    "that",
    "the",
    "their",
    "them",
    "these",
    "this",
    "to",
    "using",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
    "would",
    "you",
}


@dataclass(frozen=True)
class RetrievalHit:
    chunk: Chunk
    score: float
    matched_terms: tuple[str, ...]


def tokenize(text: str) -> list[str]:
    terms = [token.lower().strip("'\".,;:!?()[]{}") for token in TOKEN_RE.findall(text)]
    return [term for term in terms if len(term) >= 2 and term not in STOPWORDS]


def retrieve(chunks: list[Chunk], query: str, *, top_k: int = 12) -> list[RetrievalHit]:
    if not chunks:
        return []

    query_terms = tokenize(query)
    if not query_terms:
        query_terms = tokenize(query.lower())
    query_counts = Counter(query_terms)
    chunk_terms = [Counter(tokenize(chunk.text)) for chunk in chunks]

    doc_frequency: Counter[str] = Counter()
    for counts in chunk_terms:
        doc_frequency.update(counts.keys())

    total_docs = len(chunks)
    hits: list[RetrievalHit] = []
    for chunk, counts in zip(chunks, chunk_terms, strict=True):
        score = 0.0
        matched: list[str] = []
        chunk_len = max(sum(counts.values()), 1)
        for term, query_weight in query_counts.items():
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            idf = math.log((total_docs + 1) / (doc_frequency[term] + 0.5)) + 1
            score += query_weight * idf * (tf / (tf + 1.2 + 0.25 * chunk_len / 250))
            matched.append(term)

        phrase_bonus = _phrase_bonus(query, chunk.text)
        score += phrase_bonus
        if score > 0:
            hits.append(RetrievalHit(chunk=chunk, score=round(score, 4), matched_terms=tuple(matched[:20])))

    hits.sort(key=lambda hit: hit.score, reverse=True)
    return hits[:top_k]


def _phrase_bonus(query: str, text: str) -> float:
    compact_query = " ".join(query.lower().split())
    compact_text = " ".join(text.lower().split())
    quoted_phrases = re.findall(r"['\"]([^'\"]{3,80})['\"]", query)
    bonus = 0.0
    for phrase in quoted_phrases:
        if phrase.lower() in compact_text:
            bonus += 2.5

    query_numbers = set(re.findall(r"\b\d+(?:[./:-]\d+)*\b", compact_query))
    for number in query_numbers:
        if number in compact_text:
            bonus += 0.5
    return bonus

