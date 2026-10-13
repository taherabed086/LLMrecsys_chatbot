"""The one place the app calls Claude: turning chat into a search phrase."""

import os

import anthropic
import streamlit as st

from recsys_chat.config import CLAUDE_MODEL


def _api_key() -> str | None:
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY")


def extract_search_phrase(conversation: list[dict], dataset_label: str) -> str:
    """Summarizes the whole chat into one short English search phrase."""
    transcript = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in conversation
    )
    prompt = f"""A user is searching a catalog of: {dataset_label}.
The user may write in Arabic or English. Based on the whole conversation,
write one short English phrase (5 to 10 words) describing what they want,
suitable for semantic search. Respond with only the phrase.

Conversation:
{transcript}"""

    client = anthropic.Anthropic(api_key=_api_key())
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=40,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()