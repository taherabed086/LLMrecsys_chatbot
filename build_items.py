"""
build_items.py

Builds data/processed/items_<dataset>.csv from the raw metadata files.
Run once from the project root:  python build_items.py

The app never imports this file. It only prepares the small tables the
app reads, so it lives next to the app instead of inside it.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
SAMPLE = 45000


def first_image(images):
    """Best image URL from Amazon's list of image dicts, or None."""
    if not isinstance(images, list):
        return None
    for img in sorted(images, key=lambda d: d.get("variant") != "MAIN"):
        url = img.get("hi_res") or img.get("large") or img.get("thumb")
        if url:
            return url
    return None


def amazon_items(path: Path) -> pd.DataFrame:
    """Turns an Amazon meta file (.jsonl or .jsonl.gz) into the items table."""
    compression = "gzip" if path.suffix == ".gz" else None
    meta = pd.read_json(path, lines=True, compression=compression)

    desc = meta["description"].apply(
        lambda d: " ".join(d) if isinstance(d, list) else None
    )
    feat = meta["features"].apply(
        lambda f: " ".join(f[:3]) if isinstance(f, list) and f else None
    )
    return pd.DataFrame({
        "item_id": meta["parent_asin"],
        "title": meta["title"],
        # Fall back to the first feature bullets when there is no description
        "description": desc.where(desc.str.len() > 0, feat),
        "image_url": meta["images"].apply(first_image),
        "rating": meta["average_rating"],
    })


def yelp_items(path: Path) -> pd.DataFrame:
    """Yelp has no photos or descriptions, so the description is built
    from real fields (categories, city, address, review count)."""
    biz = pd.read_json(path, lines=True)
    description = (
        biz["categories"].fillna("") + " · " + biz["city"].fillna("")
        + " · " + biz["address"].fillna("")
        + " · " + biz["review_count"].astype(str) + " reviews"
    )
    return pd.DataFrame({
        "item_id": biz["business_id"],
        "title": biz["name"],
        "description": description,
        "image_url": None,
        "rating": biz["stars"],
    })


def save(df: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = df.dropna(subset=["title"])
    df = df.sample(min(SAMPLE, len(df)), random_state=42)
    df.to_csv(OUT / f"items_{name}.csv", index=False)
    print(f"{name}: {len(df)} items, {df['image_url'].notna().sum()} with image")


if __name__ == "__main__":
    save(amazon_items(RAW / "amazon_beauty" / "meta.jsonl.gz"), "beauty")
    save(amazon_items(RAW / "amazon_video_games" / "meta.jsonl.gz"), "video_games")
    save(yelp_items(RAW / "yelp" / "business.json"), "yelp")