"""A small tool-using agent: Claude searches the catalog, then picks what to show."""

import json

import anthropic
import pandas as pd

from recsys_chat.config import CLAUDE_MODEL
from recsys_chat.llm import _api_key
from recsys_chat.search import confidence_level, search

MAX_STEPS = 4

SYSTEM_PROMPT = """You help a user find items in a catalog of: {dataset_label}.
Work in two phases. First use search_catalog to find candidates (you may search
again with a different phrase if the results look off). Then call answer_user
exactly once to finish.

Rules for answer_user:
- item_ids must be copied from the search results. Never invent ids or items.
- Include only items that really fit the request. If nothing fits, send an
  empty item_ids list and say so plainly in the message.
- Write the message in the same language as the user. Keep it short: name the
  best matches and say in one line why each fits. The app shows the item cards
  separately, so do not repeat full descriptions."""

SEARCH_TOOL = {
    "name": "search_catalog",
    "description": (
        "Semantic search over the selected catalog. Pass a short English "
        "phrase (5 to 10 words) describing what the user wants. Returns the "
        "closest items with an id and a similarity score between 0 and 1. "
        "A high score does not guarantee the item really fits, so check the titles."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Short English search phrase."}
        },
        "required": ["query"],
    },
}

ANSWER_TOOL = {
    "name": "answer_user",
    "description": (
        "Send the final reply to the user and choose which items to show as cards. "
        "Call this once, when you are done searching."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "The reply for the user."},
            "item_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ids of the recommended items. Empty if nothing fits.",
            },
        },
        "required": ["message", "item_ids"],
    },
}


def _format_for_claude(results: pd.DataFrame) -> str:
    """Compact JSON of the matches; descriptions are trimmed to save tokens."""
    rows = []
    for _, row in results.iterrows():
        rows.append(
            {
                "id": str(row["item_id"]),
                "title": row["title"],
                "rating": None if pd.isna(row.get("rating")) else row["rating"],
                "description": str(row.get("description") or "")[:200],
                "score": row["score"],
            }
        )
    return json.dumps(rows, ensure_ascii=False)


def _build_reply(args: dict, seen: dict):
    """Keeps only ids that really came from a search, then scores the choice."""
    wanted = dict.fromkeys(str(i) for i in args.get("item_ids", []))
    chosen = [seen[i] for i in wanted if i in seen]
    message = str(args.get("message", "")).strip()

    if not chosen:
        return message, None, None
    results = pd.DataFrame(chosen).reset_index(drop=True)
    return message, results, confidence_level(results["score"].max())


def run_agent(conversation: list[dict], items: pd.DataFrame, dataset: str, dataset_label: str):
    """Runs the tool loop.

    Returns (message, results, confidence). results is None when Claude
    decided that nothing in the catalog fits.
    """
    client = anthropic.Anthropic(api_key=_api_key())
    messages = [{"role": m["role"], "content": m["content"]} for m in conversation]
    seen = {}  # item_id -> row, everything the searches have returned so far

    for _ in range(MAX_STEPS):
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=700,
            system=SYSTEM_PROMPT.format(dataset_label=dataset_label),
            tools=[SEARCH_TOOL, ANSWER_TOOL],
            tool_choice={"type": "any"},  # every turn must end in a tool call
            messages=messages,
        )

        tool_calls = [b for b in response.content if b.type == "tool_use"]
        final = next((b for b in tool_calls if b.name == "answer_user"), None)
        if final is not None:
            return _build_reply(final.input, seen)

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in tool_calls:
            results, _ = search(block.input["query"], items, dataset)
            for _, row in results.iterrows():
                seen[str(row["item_id"])] = row
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": _format_for_claude(results),
                }
            )
        messages.append({"role": "user", "content": tool_results})

    return "I could not settle on a good match. Try rephrasing your request.", None, None