#!/usr/bin/env python3
"""
wikimedia_ingest.py

Modes:
  1) search   - ingest by keyword search topics (TOPICS list)
  2) category - ingest by crawling Wikimedia Commons categories (Category:...),
                including subcategories, collecting File:... titles.

This script:
- Uses a browser-like requests.Session() + warmup to reduce 403 blocks
- Pulls raster thumbnails via iiurlwidth so SVGs become PNG/JPG
- Filters by license (Public Domain / CC0 / CC BY / CC BY-SA)
- Resizes and stores local images:
    ./k12_images/full
    ./k12_images/thumb
- Inserts image embeddings + metadata into ChromaDB:
    ./image_db   collection: k12_education_images
- Sanitizes metadata to Chroma-compatible primitives

Usage:
  python wikimedia_ingest.py --mode search

  python wikimedia_ingest.py --mode category --start "Category:Mathematics" --max 200 --delay 3.0 --subject Math
"""

from __future__ import annotations

import argparse
import hashlib
import re
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import requests
from PIL import Image, UnidentifiedImageError

import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

try:
    from chromadb.utils.data_loaders import ImageLoader
except Exception:
    ImageLoader = None


# ---------------- CONFIG ----------------

DB_PATH = "./image_db"
COLLECTION_NAME = "k12_education_images"

SAVE_ROOT = Path("./k12_images")
FULL_DIR = SAVE_ROOT / "full"
THUMB_DIR = SAVE_ROOT / "thumb"

FULL_MAX_SIDE = 1200
THUMB_MAX_SIDE = 256

# Default delay (tune up if you see blocks)
DEFAULT_DELAY_SEC = 3.0

# License allow-list by substring
ALLOWED_LICENSE_SUBSTRINGS = [
    "public domain",
    "cc0",
    "cc by",
    "cc-by",
    "cc by-sa",
    "cc-by-sa",
]

# Search topics (MODE=search)
TOPICS = [
    {"query": "plant cell diagram", "subject": "Biology", "grade_min": 6, "grade_max": 8, "limit": 10},
    {"query": "water cycle diagram", "subject": "Science", "grade_min": 4, "grade_max": 6, "limit": 10},
    {"query": "fractions pie chart", "subject": "Math", "grade_min": 3, "grade_max": 5, "limit": 10},
    {"query": "blank map united states", "subject": "Geography", "grade_min": 3, "grade_max": 8, "limit": 10},
]

# Wikimedia Commons API
API_URL = "https://commons.wikimedia.org/w/api.php"

# Thumbnail width request. This produces thumburl and rasterizes SVGs.
IIURL_WIDTH = FULL_MAX_SIDE

# Browser-like headers (important to reduce 403)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://commons.wikimedia.org/",
    "Connection": "keep-alive",
    # Optional (good etiquette):
    # "From": "your_email@example.com",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

_TAG_RE = re.compile(r"<[^>]+>")


# ---------------- Utilities ----------------

def ensure_dirs():
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)


def warmup_session():
    try:
        SESSION.get("https://commons.wikimedia.org/wiki/Main_Page", timeout=30)
        time.sleep(0.8)
    except Exception:
        pass


def license_allowed(name: Optional[str]) -> bool:
    if not name:
        return False
    low = str(name).lower()
    return any(s in low for s in ALLOWED_LICENSE_SUBSTRINGS)


def _to_safe_str(v: Any) -> str:
    if v is None:
        return ""
    s = str(v)
    s = _TAG_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > 500:
        s = s[:500] + "…"
    return s


def safe_meta(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chroma requires metadata values be primitives: str/int/float/bool.
    """
    out: Dict[str, Any] = {}
    for k, v in (d or {}).items():
        if v is None:
            continue
        if isinstance(v, (int, float, bool)):
            out[k] = v
        else:
            s = _to_safe_str(v)
            if s:
                out[k] = s
    return out


def resize_and_save(src: Path, dest: Path, max_side: int):
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = min(max_side / max(w, h), 1.0)
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        im.save(dest, format="JPEG", quality=90, optimize=True)


def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    embedding_function = OpenCLIPEmbeddingFunction()
    kwargs = dict(name=COLLECTION_NAME, embedding_function=embedding_function)
    if ImageLoader:
        kwargs["data_loader"] = ImageLoader()
    return client.get_or_create_collection(**kwargs)


def already_in_db(collection, _id: str) -> bool:
    try:
        got = collection.get(ids=[_id])
        return bool(got.get("ids"))
    except Exception:
        return False


def download_to(dest: Path, url: str) -> Tuple[int, str]:
    """
    Returns (status_code, content_type)
    """
    try:
        with SESSION.get(url, stream=True, timeout=60, allow_redirects=True) as r:
            status = r.status_code
            ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()

            if status != 200:
                print("DOWNLOAD FAIL", status, ctype, "for", url)
                try:
                    print("Preview:", r.text[:160].replace("\n", " "))
                except Exception:
                    pass
                return status, ctype

            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 128):
                    if chunk:
                        f.write(chunk)

        return 200, ctype
    except Exception as e:
        print("DOWNLOAD EXCEPTION", type(e).__name__, str(e)[:160], "for", url)
        return 0, ""


# ---------------- API: Search Mode ----------------

def search_commons(query: str, limit: int):
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": limit,
        "gsrnamespace": 6,  # File namespace only
        "prop": "imageinfo",
        "iiprop": "url|mime|extmetadata",
        "iiurlwidth": IIURL_WIDTH,  # thumburl provided
    }
    r = SESSION.get(API_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    return list(data.get("query", {}).get("pages", {}).values())


# ---------------- API: Category Mode ----------------

def category_members(category: str, cmtype: str = "file|subcat", cmcontinue: Optional[str] = None, limit: int = 500):
    """
    Iterate through categorymembers results. cmtype can include file, subcat.
    """
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": category,
        "cmtype": cmtype,
        "cmlimit": limit,
    }
    if cmcontinue:
        params["cmcontinue"] = cmcontinue

    while True:
        r = SESSION.get(API_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        for item in data.get("query", {}).get("categorymembers", []):
            yield item

        cont = data.get("continue", {})
        if not cont or "cmcontinue" not in cont:
            break
        params["cmcontinue"] = cont["cmcontinue"]


def imageinfo_for_titles(titles: List[str]):
    """
    Batch fetch imageinfo for File:... titles.
    """
    if not titles:
        return []
    params = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "titles": "|".join(titles),
        "iiprop": "url|mime|extmetadata",
        "iiurlwidth": IIURL_WIDTH,
    }
    r = SESSION.get(API_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    pages = data.get("query", {}).get("pages", {})
    return list(pages.values())


def crawl_categories_for_files(start_categories: List[str], max_files: int) -> Iterable[str]:
    """
    BFS crawl categories and yield File:... titles.
    Stops after yielding max_files file titles.
    """
    q = deque(start_categories)
    seen_cats: Set[str] = set()
    yielded = 0

    while q and yielded < max_files:
        cat = q.popleft()
        if cat in seen_cats:
            continue
        seen_cats.add(cat)
        print("Exploring category:", cat)

        try:
            for member in category_members(cat, cmtype="file|subcat", limit=500):
                ns = member.get("ns")
                title = member.get("title")
                if not title:
                    continue

                # File namespace
                if ns == 6 and title.startswith("File:"):
                    yield title
                    yielded += 1
                    if yielded >= max_files:
                        break

                # Category namespace
                if ns == 14 and title.startswith("Category:") and title not in seen_cats:
                    q.append(title)

        except Exception as e:
            print("Category crawl error:", cat, "-", str(e)[:160])
            continue


# ---------------- Core ingest routine (shared) ----------------

def process_page_into_db(
    collection,
    page: Dict[str, Any],
    subject: str,
    topic: str,
    grade_min: int,
    grade_max: int,
    delay_sec: float,
    skipped: Dict[str, int],
) -> bool:
    """
    Process a single page dict (from search_commons or imageinfo_for_titles) into DB.
    Returns True if added, else False.
    """
    info = (page.get("imageinfo") or [{}])[0]
    meta = info.get("extmetadata", {}) or {}

    license_name = (meta.get("LicenseShortName", {}) or {}).get("value")
    if not license_allowed(license_name):
        skipped["license"] += 1
        return False

    thumburl = info.get("thumburl")
    if not thumburl:
        skipped["no_thumburl"] += 1
        return False

    _id = hashlib.sha256(thumburl.encode("utf-8")).hexdigest()[:16]
    if already_in_db(collection, _id):
        return False

    tmp = SAVE_ROOT / f"tmp_{_id}"
    status, _ctype = download_to(tmp, thumburl)
    if status != 200:
        skipped["download"] += 1
        tmp.unlink(missing_ok=True)
        time.sleep(max(delay_sec, 6.0))
        return False

    full_path = FULL_DIR / f"{_id}.jpg"
    thumb_path = THUMB_DIR / f"{_id}.jpg"

    try:
        resize_and_save(tmp, full_path, FULL_MAX_SIDE)
        resize_and_save(tmp, thumb_path, THUMB_MAX_SIDE)
    except UnidentifiedImageError:
        skipped["process"] += 1
        tmp.unlink(missing_ok=True)
        return False
    except Exception as e:
        print("PROCESS FAIL", str(e)[:160])
        skipped["process"] += 1
        tmp.unlink(missing_ok=True)
        return False
    finally:
        tmp.unlink(missing_ok=True)

    raw_metadata = {
        "subject": subject,
        "topic": topic,
        "grade_min": int(grade_min),
        "grade_max": int(grade_max),
        "license": license_name,
        "license_url": (meta.get("LicenseUrl", {}) or {}).get("value"),
        "artist": (meta.get("Artist", {}) or {}).get("value"),
        "credit": (meta.get("Credit", {}) or {}).get("value"),
        "source_url": info.get("url"),
        "thumb_url": thumburl,
        "source_page": f"https://commons.wikimedia.org/wiki/{page.get('title','').replace(' ', '_')}",
        "review_status": "unreviewed",
    }
    metadata = safe_meta(raw_metadata)

    try:
        collection.add(ids=[_id], uris=[str(full_path)], metadatas=[metadata])
        time.sleep(delay_sec)
        return True
    except Exception as e:
        print("DB ADD FAIL", str(e)[:160])
        skipped["db"] += 1
        time.sleep(delay_sec)
        return False


# ---------------- Runners ----------------

def run_search_mode(delay_sec: float):
    ensure_dirs()
    warmup_session()
    collection = get_collection()

    print("DB:", DB_PATH, "| Collection:", COLLECTION_NAME, "| Current:", collection.count())
    added_total = 0

    for topic in TOPICS:
        query = topic["query"]
        limit = int(topic.get("limit", 10))
        subject = topic["subject"]
        grade_min = int(topic["grade_min"])
        grade_max = int(topic["grade_max"])

        print("\nSearching:", query)
        pages = search_commons(query, limit)

        skipped = {"no_thumburl": 0, "license": 0, "download": 0, "process": 0, "db": 0}
        added_topic = 0

        for page in pages:
            ok = process_page_into_db(
                collection=collection,
                page=page,
                subject=subject,
                topic=query,
                grade_min=grade_min,
                grade_max=grade_max,
                delay_sec=delay_sec,
                skipped=skipped,
            )
            if ok:
                added_topic += 1
                added_total += 1
                print("Added:", subject, "|", query)

        print(f"Added for topic: {added_topic}")
        print("Skipped:", skipped)

    print("\nDone! Total added:", added_total)
    print("Total images in DB:", collection.count())


def run_category_mode(
    start_categories: List[str],
    subject: str,
    topic_label: str,
    grade_min: int,
    grade_max: int,
    max_files: int,
    delay_sec: float,
):
    ensure_dirs()
    warmup_session()
    collection = get_collection()

    print("DB:", DB_PATH, "| Collection:", COLLECTION_NAME, "| Current:", collection.count())
    print("Mode: category")
    print("Start categories:", start_categories)
    print("Max files:", max_files, "| Delay:", delay_sec)

    skipped = {"no_thumburl": 0, "license": 0, "download": 0, "process": 0, "db": 0}
    added_total = 0

    # Crawl file titles
    file_titles = list(crawl_categories_for_files(start_categories, max_files=max_files))
    print("Collected file titles:", len(file_titles))

    # Process in batches to fetch imageinfo + thumburl
    BATCH = 20
    for i in range(0, len(file_titles), BATCH):
        batch = file_titles[i : i + BATCH]
        pages = imageinfo_for_titles(batch)
        for page in pages:
            ok = process_page_into_db(
                collection=collection,
                page=page,
                subject=subject,
                topic=topic_label,
                grade_min=grade_min,
                grade_max=grade_max,
                delay_sec=delay_sec,
                skipped=skipped,
            )
            if ok:
                added_total += 1
                if added_total % 10 == 0:
                    print(f"Added so far: {added_total}/{max_files}")

            if added_total >= max_files:
                break
        if added_total >= max_files:
            break

    print("\nCategory ingest complete.")
    print("Total added:", added_total)
    print("Skipped:", skipped)
    print("Total images in DB:", collection.count())


# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["search", "category"], default="search", help="Ingest mode")
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY_SEC, help="Seconds between downloads")

    # Category mode args
    ap.add_argument(
        "--start",
        nargs="+",
        default=["Category:Mathematics"],
        help='Start categories (e.g. "Category:Mathematics")',
    )
    ap.add_argument("--max", type=int, default=200, help="Max files to add (category mode)")
    ap.add_argument("--subject", type=str, default="Math", help="Metadata subject label (category mode)")
    ap.add_argument("--topic-label", type=str, default="category ingest", help="Metadata topic label (category mode)")
    ap.add_argument("--grade-min", type=int, default=0, help="Grade min metadata (category mode)")
    ap.add_argument("--grade-max", type=int, default=12, help="Grade max metadata (category mode)")

    args = ap.parse_args()

    if args.mode == "search":
        run_search_mode(delay_sec=args.delay)
    else:
        run_category_mode(
            start_categories=args.start,
            subject=args.subject,
            topic_label=args.topic_label,
            grade_min=args.grade_min,
            grade_max=args.grade_max,
            max_files=args.max,
            delay_sec=args.delay,
        )


if __name__ == "__main__":
    main()
