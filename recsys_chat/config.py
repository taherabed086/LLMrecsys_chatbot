"""Central configuration for the recommendation chat app."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATASETS = {
    "beauty": "Beauty and Personal Care",
    "video_games": "Video Games",
    "yelp": "Restaurants and Local Businesses",
    "alnahal": "Al-Nahal Al-Jawal (Honey and Herbs)",
}

CLAUDE_MODEL = "claude-sonnet-4-5"

# Each dataset can pick its own embedding model. Datasets not listed here
# fall back to DEFAULT_EMBEDDING_MODEL.
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_MODELS = {
    "alnahal": "paraphrase-multilingual-MiniLM-L12-v2",
}

TOP_K_MATCHES = 5
SAMPLE_SIZE = 6

# Heuristic thresholds, picked by trying a few values on real queries,
# not from a formal evaluation. Worth revisiting with usage data.
CONFIDENCE_THRESHOLDS = {"high": 0.55, "medium": 0.35, "low": 0.0}

CONFIDENCE_LABELS = {
    "high": "Strong match",
    "medium": "Moderate match",
    "low": "Weak match",
}

SUGGESTED_PROMPTS = {
    "beauty": ["Something for dry, sensitive skin", "A gentle everyday moisturizer"],
    "video_games": ["A relaxing puzzle game", "A scary horror game"],
    "yelp": ["A cozy Italian restaurant", "Somewhere good for coffee"],
    "alnahal": ["عسل طبيعي للعائلة", "أعشاب تساعد على الهضم"],
}

# Extra per-dataset rules appended to the agent's system prompt.
# Datasets not listed here get no extra rules.
DATASET_RULES = {
    "alnahal": """This catalog sells honey, herbs, and other natural products. Do not make
medical claims: never say an item treats, cures, or prevents an illness, and
never cite medical or scientific evidence. For a mild wellness request (a
cough, a sore throat, digestion), you may still recommend a relevant item,
describing it by how people traditionally use it rather than as a cure (for
example: "often used to soothe a sore throat"). For anything that sounds
like a serious or persistent medical condition, say you cannot advise on
medical conditions and suggest seeing a doctor, instead of recommending a
product.""",
}