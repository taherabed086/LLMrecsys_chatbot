"""The one place the app reads the Anthropic API key from."""

import os

import streamlit as st


def _api_key() -> str | None:
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY")