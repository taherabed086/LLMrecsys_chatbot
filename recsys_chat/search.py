"""Semantic search inside a single, user-selected dataset."""

import pandas as pd
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util

from recsys_chat.config import (
    CONFIDENCE_THRESHOLDS,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_MODELS,
    PROCESSED_DIR,
    TOP_K_MATCHES,
)


def embedding_model_name(dataset: str) -> str:
    return EMBEDDING_MODELS.get(dataset, DEFAULT_EMBEDDING_MODEL)


@st.cache_resource
def load_embedding_model(model_name: str) -> SentenceTransformer:
    """Small local model; no API cost per query."""
    return SentenceTransformer(model_name)


@st.cache_resource
def get_embeddings(dataset: str, titles: tuple) -> torch.Tensor:
    """One embedding per item title, saved to disk after the first run."""
    path = PROCESSED_DIR / f"embeddings_{dataset}.pt"
    if path.exists():
        return torch.load(path)

    model = load_embedding_model(embedding_model_name(dataset))
    vectors = model.encode(
        list(titles), convert_to_tensor=True, batch_size=32, show_progress_bar=True
    )
    torch.save(vectors, path)
    return vectors


def confidence_level(score: float) -> str:
    if score >= CONFIDENCE_THRESHOLDS["high"]:
        return "high"
    if score >= CONFIDENCE_THRESHOLDS["medium"]:
        return "medium"
    if score > CONFIDENCE_THRESHOLDS["low"]:
        return "low"
    return "none"


def search(query: str, items: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, str]:
    """Returns the closest items (with a 'score' column) and a confidence level."""
    vectors = get_embeddings(dataset, tuple(items["title"]))
    model = load_embedding_model(embedding_model_name(dataset))
    query_vector = model.encode(query, convert_to_tensor=True)

    scores = util.cos_sim(query_vector, vectors)[0]
    top = scores.topk(min(TOP_K_MATCHES, len(scores)))

    results = items.iloc[top.indices.tolist()].copy()
    results["score"] = [round(s, 3) for s in top.values.tolist()]
    return results, confidence_level(results["score"].iloc[0])