# Smart Recommendation Assistant

A chatbot that recommends items from a real catalog. You pick a dataset, look at a sample of its items, then describe what you want in plain language. The app finds the closest matches by meaning, not by keywords, and tells you how confident it is in each one.

**Live demo:** <https://llmrecsyschatbot.streamlit.app>

## How it works

1. **Choose a dataset** in the sidebar (Amazon Beauty, Amazon Video Games, Yelp, or the Al-Nahal Al-Jawal honey and herbs store).
2. **Preview a sample.** A grid of cards shows real items with their picture, title, rating and description. "Show another sample" draws a new random set.
3. **Describe what you want**, for example "something for dry skin" or "a quiet place for a date night".
4. **Claude runs a small tool-use loop.** It calls a `search_catalog` tool (semantic search over the chosen dataset) as many times as it needs, then calls `answer_user` with a short reply and the ids of the items that really fit. It can only pick ids that a real search actually returned, so it cannot invent products that are not in the data.
5. **Semantic search** compares Claude's search phrase with every item in the chosen dataset using sentence embeddings and cosine similarity.
6. **Results come back as cards** with the description and a confidence label (strong, moderate or weak match) based on the similarity score.

```
user message -> Claude (search_catalog tool, one or more times) -> embeddings + cosine similarity
             -> Claude (answer_user: picks fitting ids) -> top matches + confidence
```

## Why semantic search

A keyword search for "dry skin" misses a product described as "moisturizer for parched skin". Embeddings turn both the query and the items into vectors that capture meaning, so the two end up close together even when they share no words.

## Why the user picks the dataset

An earlier version searched all datasets at once. A Yelp business called "Skin Care Studio" kept outranking real skin care products. Letting the user choose the dataset removes that mixing completely and also makes the sample preview possible.

## Datasets

| Dataset | Source | Items used |
|---|---|---|
| Amazon Beauty | Amazon Reviews 2023 (metadata) | 3,000 |
| Amazon Video Games | Amazon Reviews 2023 (metadata) | 3,000 |
| Yelp | Yelp Open Dataset (business.json) | 3,000 |
| Al-Nahal Al-Jawal | Scraped from the store's own product pages (`scrape_store.py`) | 286 |

Each dataset is converted to a table with the same core columns (`item_id, title, description, image_url, rating`), so the app code does not care where the data came from. A dataset can add extra columns on top (Al-Nahal Al-Jawal also has `price` and `category`); nothing elsewhere breaks, since the rest of the app only reads the core columns.

## Project structure

```
LLM_Recsys/
├── streamlit_app.py        # the app: sidebar, sample cards, chat loop
├── build_items.py          # one-time script that builds the item tables
├── scrape_store.py         # one-time script that scrapes Al-Nahal Al-Jawal
├── recsys_chat/
│   ├── config.py           # paths, model names, thresholds, suggested prompts,
│   │                       # per-dataset embedding models and extra agent rules
│   ├── data.py             # loads and validates the item tables
│   ├── search.py           # embeddings, cosine similarity, confidence levels
│   ├── agent.py            # the tool-use loop: search_catalog, then answer_user
│   ├── llm.py              # reads the Anthropic API key
│   └── ui.py               # item cards and confidence badge
├── data/
│   ├── raw/                # original downloads and scrapes (not in git)
│   └── processed/          # items_<dataset>.csv, embeddings_<dataset>.pt
├── tests/
│   └── test_logic.py       # unit tests for the confidence logic
├── requirements.txt
└── .streamlit/             # config.toml (secrets.toml stays local)
```

## Run locally

```bash
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:

```toml
ANTHROPIC_API_KEY = "your-key-here"
```

Then start the app:

```bash
python -m streamlit run streamlit_app.py
```

The first run downloads the embedding model and computes embeddings for the chosen dataset. They are saved to `data/processed/`, so later runs start quickly.

## Rebuild the item tables

Only needed if you want different data. Put the raw files under `data/raw/` and run:

```bash
python build_items.py
```

If you change the items, delete the matching `embeddings_*.pt` file so the embeddings are computed again.

## Tests

```bash
python -m pytest tests/test_logic.py -v
```

## Known limitations

- The confidence thresholds (0.55 and 0.35) were picked by trying values and checking results by eye. They have not been tuned against human judgments.
- The Amazon and Yelp datasets are a random sample of 3,000 items, not the full catalog. Al-Nahal Al-Jawal is small enough that all scraped items are kept.
- The Yelp table contains every kind of business, not only restaurants, and Yelp has no product photos in the source files.
- Search runs on the item title only. Adding the description to the embedding text would likely help.
- Some image links come from the original datasets and can stop working over time.
