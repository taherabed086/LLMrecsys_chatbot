"""Collects product data for one Mahally store.

Product ids come from the store's own Salla sitemap, then each product page
on Mahally is fetched and its JSON-LD block is saved as one line of JSON.
The output goes to data/raw/, which is not committed.

    python scrape_store.py --limit 5    # quick test
    python scrape_store.py              # everything (safe to re-run, it resumes)
"""

import argparse
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

STORE_ID = "1894370252"
STORE_SITEMAP = "https://alnahalaljwal.com.sa/sitemap.xml"
PRODUCT_URL = "https://www.mahally.com/ar/products/{store}/{product}/"
OUT_FILE = Path("data/raw/alnahal/products.jsonl")
DELAY_SECONDS = 1.0

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; LLMRecsysResearch/0.1)"}
SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
JSONLD_RE = re.compile(r'<script[^>]*id="productJsonLd"[^>]*>(.*?)</script>', re.S)
PRODUCT_ID_RE = re.compile(r"/p(\d+)/?$")


def fetch(session: requests.Session, url: str):
    """GET with a few polite retries. Returns the response or None."""
    for attempt in range(3):
        try:
            response = session.get(url, headers=HEADERS, timeout=20)
            if response.status_code == 200:
                return response
            print(f"  {response.status_code} on {url}")
        except requests.RequestException as err:
            print(f"  {err.__class__.__name__} on {url}")
        time.sleep(3 * (attempt + 1))
    return None


def sitemap_locs(session: requests.Session, url: str) -> list[str]:
    response = fetch(session, url)
    if response is None:
        return []
    root = ET.fromstring(response.content)
    return [el.text.strip() for el in root.findall(".//sm:loc", SITEMAP_NS)]


def product_ids(session: requests.Session) -> list[str]:
    """Walks the sitemap index and keeps only product links (/p<digits>)."""
    ids = []
    for child in sitemap_locs(session, STORE_SITEMAP):
        if "blog" in child:
            continue
        for loc in sitemap_locs(session, child):
            match = PRODUCT_ID_RE.search(loc)
            if match:
                ids.append(match.group(1))
    return list(dict.fromkeys(ids))  # drop duplicates, keep order


def extract_product(html: str):
    """Returns the schema.org Product node from the page, or None."""
    match = JSONLD_RE.search(html)
    if not match:
        return None
    data = json.loads(match.group(1))
    for node in data.get("@graph", [data]):
        kind = node.get("@type")
        if kind == "Product" or (isinstance(kind, list) and "Product" in kind):
            return node
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="only fetch this many (0 = all)")
    args = parser.parse_args()

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT_FILE.exists():
        with OUT_FILE.open(encoding="utf-8") as f:
            done = {json.loads(line)["item_id"] for line in f if line.strip()}

    session = requests.Session()
    ids = product_ids(session)
    print(f"{len(ids)} product ids in the sitemap, {len(done)} already saved")

    todo = [pid for pid in ids if pid not in done][: args.limit or None]
    saved = missing = 0
    with OUT_FILE.open("a", encoding="utf-8") as out:
        for n, pid in enumerate(todo, 1):
            url = PRODUCT_URL.format(store=STORE_ID, product=pid)
            response = fetch(session, url)
            product = extract_product(response.text) if response else None

            if product is None:
                missing += 1
                print(f"[{n}/{len(todo)}] {pid}: no product data")
            else:
                row = {"item_id": pid, "url": url, "product": product}
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()  # keep what we have if the run stops halfway
                saved += 1
                print(f"[{n}/{len(todo)}] {pid}: {str(product.get('name', ''))[:50]}")
            time.sleep(DELAY_SECONDS)

    print(f"Done. Saved {saved}, missing {missing}. File: {OUT_FILE}")


if __name__ == "__main__":
    main()