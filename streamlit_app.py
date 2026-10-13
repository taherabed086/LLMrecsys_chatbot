"""
Entry point. Pick a dataset, preview a sample of it, then chat to find
items by describing what you want in your own words.
"""

import streamlit as st

from recsys_chat.config import DATASETS, SAMPLE_SIZE, SUGGESTED_PROMPTS
from recsys_chat.data import load_items
from recsys_chat.llm import extract_search_phrase
from recsys_chat.search import search
from recsys_chat.ui import confidence_badge, render_cards

st.set_page_config(page_title="Smart Recommendation Assistant", page_icon="🛍️")

st.markdown(
    """
    <style>
    .confidence-badge {
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        font-size: 0.8rem; font-weight: 600; margin-bottom: 6px;
    }
    .confidence-high   { background: #d4f7dc; color: #1e7e34; }
    .confidence-medium { background: #fff3cd; color: #856404; }
    .confidence-low    { background: #ffe1d6; color: #a94442; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛍️ Smart Recommendation Assistant")

# --- 1. The user chooses where to search ---------------------------------
with st.sidebar:
    dataset = st.selectbox(
        "Search in", list(DATASETS), format_func=lambda key: DATASETS[key]
    )

# A new dataset means a new conversation and a new sample
if st.session_state.get("dataset") != dataset:
    st.session_state.dataset = dataset
    st.session_state.messages = []
    st.session_state.sample_seed = 0

items = load_items(dataset)

# --- 2. A sample of the chosen dataset -----------------------------------
st.subheader(f"A look inside: {DATASETS[dataset]}")
if st.button("Show another sample"):
    st.session_state.sample_seed += 1

sample = items.sample(
    min(SAMPLE_SIZE, len(items)), random_state=st.session_state.sample_seed
)
render_cards(sample)

st.divider()

# --- 3. The conversation --------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)
        if message.get("results") is not None:
            render_cards(message["results"])

clicked = None
if not st.session_state.messages:
    cols = st.columns(len(SUGGESTED_PROMPTS[dataset]))
    for col, text in zip(cols, SUGGESTED_PROMPTS[dataset]):
        if col.button(text, use_container_width=True):
            clicked = text

user_input = st.chat_input("Describe what you are looking for") or clicked

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            phrase = extract_search_phrase(
                st.session_state.messages, DATASETS[dataset]
            )
            results, level = search(phrase, items, dataset)

        if level == "none":
            text = f'No close match for "{phrase}". Try describing it differently.'
            results = None
        else:
            badge = confidence_badge(level, results["score"].iloc[0])
            text = f'{badge}\n\nInterpreted request: "{phrase}"'

        st.markdown(text, unsafe_allow_html=True)
        if results is not None:
            render_cards(results)

    st.session_state.messages.append(
        {"role": "assistant", "content": text, "results": results}
    )