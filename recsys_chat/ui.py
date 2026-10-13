"""Reusable pieces of the interface: item cards and the sample grid."""

import pandas as pd
import streamlit as st

from recsys_chat.config import CONFIDENCE_LABELS


def render_card(item: pd.Series) -> None:
    """One item as a bordered card: image, title, rating, description."""
    with st.container(border=True):
        image_url = item.get("image_url")
        if isinstance(image_url, str) and image_url:
            st.image(image_url, use_container_width=True)

        st.markdown(f"**{item['title']}**")

        if pd.notna(item.get("rating")):
            st.caption(f"{item['rating']} ★")

        description = item.get("description")
        if isinstance(description, str) and description:
            st.caption(description[:220] + ("..." if len(description) > 220 else ""))

        if "score" in item and pd.notna(item["score"]):
            st.caption(f"similarity {item['score']}")


def render_cards(items: pd.DataFrame, columns: int = 3) -> None:
    """Lays items out in a grid of `columns` cards per row."""
    rows = [items.iloc[i : i + columns] for i in range(0, len(items), columns)]
    for row in rows:
        cols = st.columns(columns)
        for col, (_, item) in zip(cols, row.iterrows()):
            with col:
                render_card(item)


def confidence_badge(level: str, score: float) -> str:
    return (
        f'<span class="confidence-badge confidence-{level}">'
        f"{CONFIDENCE_LABELS[level]}, top score {score}</span>"
    )