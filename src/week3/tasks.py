"""Baseline NLP utilities for the Week 3 deliverable.

The project primary model remains Sentence-BERT retrieval. These utilities add
small, reproducible baselines for the other requested NLP task families:
NER, text classification, extractive summarization, RAG retrieval, semantic
similarity, and structured information extraction.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAG_DOCUMENTS_PATH = PROJECT_ROOT / "Data" / "processed" / "rag_documents.csv"

_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z-]{2,}")
_STOP_WORDS = {
    "about", "after", "again", "against", "also", "among", "because",
    "between", "could", "from", "have", "into", "more", "other", "over",
    "such", "than", "that", "their", "there", "these", "they", "this",
    "through", "using", "were", "which", "with", "would",
}


def extract_entities(text: str) -> dict[str, list[str]]:
    """Extract lightweight academic entities without requiring a new model.

    This is a transparent baseline NER implementation. It recognizes common
    academic organization names, person-like names, DOIs, URLs, email
    addresses, and publication years.
    """
    value = str(text or "")
    organization_pattern = re.compile(
        r"\b(?:[A-Z][\w&.-]*\s+){0,5}"
        r"(?:University|Institute|College|Laboratory|Laboratories|Lab|"
        r"Centre|Center|Department|Technology|Research)\b"
    )
    person_pattern = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b")
    entities = {
        "ORGANIZATION": organization_pattern.findall(value),
        "PERSON": person_pattern.findall(value),
        "DOI": [
            match.rstrip(".,;:)")
            for match in re.findall(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", value, re.I)
        ],
        "URL": re.findall(r"https?://[^\s)]+", value),
        "EMAIL": re.findall(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", value),
        "YEAR": re.findall(r"\b(?:19|20)\d{2}\b", value),
    }
    return {label: list(dict.fromkeys(items)) for label, items in entities.items()}


def extract_information(text: str, top_k: int = 10) -> dict[str, Any]:
    """Return structured academic metadata and salient terms from text."""
    value = str(text or "")
    tokens = [token.lower() for token in _TOKEN_PATTERN.findall(value)]
    term_counts = Counter(token for token in tokens if token not in _STOP_WORDS)
    return {
        "entities": extract_entities(value),
        "keywords": [term for term, _ in term_counts.most_common(top_k)],
        "character_count": len(value),
        "sentence_count": len([sentence for sentence in _split_sentences(value)]),
    }


def _split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in _SENTENCE_PATTERN.split(text.strip()) if sentence.strip()]


def summarize_text(text: str, max_sentences: int = 3) -> str:
    """Create an extractive summary by ranking sentences with TF-IDF weights."""
    sentences = _split_sentences(str(text or ""))
    if len(sentences) <= max_sentences:
        return " ".join(sentences)
    vectorizer = TfidfVectorizer(stop_words="english")
    sentence_matrix = vectorizer.fit_transform(sentences)
    sentence_scores = np.asarray(sentence_matrix.sum(axis=1)).ravel()
    selected = np.argsort(sentence_scores)[::-1][:max_sentences]
    return " ".join(sentences[index] for index in sorted(selected))


def train_text_classifier(texts: Sequence[str], labels: Sequence[str]) -> dict[str, Any]:
    """Train a reproducible TF-IDF plus logistic-regression classifier."""
    if len(texts) != len(labels) or len(texts) < 2:
        raise ValueError("texts and labels must have the same length and at least two rows")
    if len(set(labels)) < 2:
        raise ValueError("classification requires at least two distinct labels")
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))
    features = vectorizer.fit_transform(texts)
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(features, labels)
    return {"vectorizer": vectorizer, "classifier": classifier}


def classify_text(text: str, trained_model: dict[str, Any]) -> dict[str, Any]:
    """Classify one text with a model returned by ``train_text_classifier``."""
    vectorizer = trained_model["vectorizer"]
    classifier = trained_model["classifier"]
    features = vectorizer.transform([text])
    probabilities = classifier.predict_proba(features)[0]
    best_index = int(np.argmax(probabilities))
    return {
        "label": str(classifier.classes_[best_index]),
        "confidence": float(probabilities[best_index]),
        "probabilities": {
            str(label): float(probability)
            for label, probability in zip(classifier.classes_, probabilities)
        },
    }


def semantic_similarity(text_a: str, text_b: str, model: SentenceTransformer | None = None) -> float:
    """Calculate normalized Sentence-BERT cosine similarity for two texts."""
    encoder = model or SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = encoder.encode([str(text_a), str(text_b)], normalize_embeddings=True)
    return float(np.dot(embeddings[0], embeddings[1]))


def retrieve_rag(query: str, top_k: int = 5, documents_path: Path = RAG_DOCUMENTS_PATH) -> list[dict[str, Any]]:
    """Retrieve the most relevant RAG documents with sparse cosine similarity."""
    if not str(query).strip():
        return []
    documents = pd.read_csv(documents_path, keep_default_na=False)
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=50000,
    )
    search_text = documents["title_or_name"] + " " + documents["text"]
    document_matrix = vectorizer.fit_transform(search_text)
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, document_matrix).ravel()
    indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for rank, index in enumerate(indices, start=1):
        row = documents.iloc[int(index)]
        results.append({
            "rank": rank,
            "doc_id": row["doc_id"],
            "entity_type": row["entity_type"],
            "title_or_name": row["title_or_name"],
            "score": round(float(scores[index]), 4),
            "text": row["text"],
        })
    return results
