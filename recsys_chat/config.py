"""Central configuration for the recommendation chat app."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATASETS = {
    "beauty": "Beauty and Personal Care",
    "video_games": "Video Games",
    "yelp": "Restaurants and Local Businesses",
}

CLAUDE_MODEL = "claude-sonnet-4-5"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

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
}