"""Loading the cleaned item tables the app searches over."""

import pandas as pd
import streamlit as st

from recsys_chat.config import PROCESSED_DIR

REQUIRED_COLUMNS = ["item_id", "title", "description", "image_url", "rating"]


@st.cache_data
def load_items(dataset: str) -> pd.DataFrame:
    """Reads data/processed/items_<dataset>.csv and validates its columns."""
    path = PROCESSED_DIR / f"items_{dataset}.csv"
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")

    return df.dropna(subset=["title"]).reset_index(drop=True)