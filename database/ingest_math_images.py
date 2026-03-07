#!/usr/bin/env python3
"""
ingest_math_images.py

Math-only ingestion pipeline for a local K-12 image library using:
- Wikimedia Commons (MediaWiki API)
- ChromaDB PersistentClient + OpenCLIP embeddings
- Local storage of images (full + thumb)
- License allowlist and media-type filtering
- Rate limit handling (Retry-After + exponential backoff)
- Stable IDs (based on canonical file URL or title)

Example:
  python3 ingest_math_images.py --mode search --delay 3 --max 50
  python3 ingest_math_images.py --mode category --start "Category:Mathematics" --max 200 --delay 3

Requires:
  pip install requests pillow chromadb
  (and your OpenCLIP deps if needed by your chromadb embedding function)
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import random
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

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("ingest_math_images")

# ---------------- Config ----------------
DB_PATH = "./image_db"
COLLECTION_NAME = "k12_education_images"

SAVE_ROOT = Path("./k12_images")
FULL_DIR = SAVE_ROOT / "full"
THUMB_DIR = SAVE_ROOT / "thumb"

FULL_MAX_SIDE = 1200
THUMB_MAX_SIDE = 256

API_URL = "https://commons.wikimedia.org/w/api.php"

HEADERS = {
    "User-Agent": "k12-math-image-ingestor/1.0 (educational; contact: add-your-email-here)",
    "Accept": "application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://commons.wikimedia.org/",
    "Connection": "keep-alive",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

_TAG_RE = re.compile(r"<[^>]+>")

# Only allow real images (not video/audio). PDFs are disabled by default because PIL often can't handle them cleanly.
ALLOW_PDF = False

ALLOWED_LICENSE_SUBSTRINGS = [
    "public domain",
    "cc0",
    "cc by",
    "cc-by",
    "cc by-sa",
    "cc-by-sa",
]

# Math-focused starter topics (edit freely)
MATH_TOPICS = [
    {"query": "fractions pie chart", "grade_min": 3, "grade_max": 5, "limit": 15},
    {"query": "number line integers", "grade_min": 4, "grade_max": 7, "limit": 15},
    {"query": "coordinate plane blank grid", "grade_min": 5, "grade_max": 8, "limit": 15},
    {"query": "area of triangle formula diagram", "grade_min": 6, "grade_max": 9, "limit": 15},
    {"query": "circle radius diameter diagram", "grade_min": 4, "grade_max": 8, "limit": 15},
]

DEFAULT_CATEGORY_THUMB_WIDTH = 800  # polite size for category mode batches

# ---------------- Utilities ----------------
def ensure_dirs() -> None:
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)


def warmup_session() -> None:
    try:
        SESSION.get("https://commons.wikimedia.org/wiki/Main_Page", timeout=30)
        time.sleep(0.6)
    except Exception:
        pass


def license_allowed(name: Optional[str]) -> bool:
    if not name:
        return False
    low = str(name).lower()
    return any(s in low for s in ALLOWED_LICENSE_SUBSTRINGS)


def is_allowed_mime(mime: Optional[str]) -> bool:
    if not mime:
        return False
    m = mime.lower().strip()
    if m.startswith("image/"):
        return True
    if ALLOW_PDF and m == "application/pdf":
        return True
    return False


def clean_html(s: Any) -> str:
    s = "" if s is None else str(s)
    s = _TAG_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def safe_meta(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (d or {}).items():
        if v is None:
            continue
        if isinstance(v, (int, float, bool)):
            out[k] = v
        else:
            s = clean_html(v)
            if s:
                out[k] = s[:800]  # keep metadata compact
    return out


def resize_and_save(src: Path, dest: Path, max_side: int) -> None:
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = min(max_side / max(w, h), 1.0)
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        im.save(dest, format="JPEG", quality=90, optimize=True)


def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    embedder = OpenCLIPEmbeddingFunction()
    kwargs = dict(name=COLLECTION_NAME, embedding_function=embedder)
    if ImageLoader is not None:
        kwargs["data_loader"] = ImageLoader()
    return client.get_or_create_collection(**kwargs)


# ---------------- Backoff Helpers ----------------
def _sleep_with_jitter(seconds: float) -> None:
    time.sleep(max(0.0, seconds + random.uniform(0.0, 0.8)))


def download_with_backoff(dest: Path, url: str, base_delay: float, max_retries: int = 8) -> Tuple[int, str]:
    """
    Download with Retry-After + exponential backoff on 429/503/504.
    Returns (status_code, content_type).
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            with SESSION.get(url, stream=True, timeout=60, allow_redirects=True) as r:
                status = r.status_code
                ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()

                if status == 200:
                    with open(dest, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 128):
                            if chunk:
                                f.write(chunk)
                    return 200, ctype

                if status in (429, 503, 504) and attempt <= max_retries:
                    ra = r.headers.get("Retry-After")
                    if ra and ra.isdigit():
                        sleep_s = float(int(ra))
                    else:
                        sleep_s = min(120.0, (2 ** (attempt - 1)) * max(base_delay, 2.0))
                    logger.warning("DOWNLOAD RETRY %d/%d status=%s sleep=%.1fs url=%s",
                                   attempt, max_retries, status, sleep_s, url)
                    dest.unlink(missing_ok=True)
                    _sleep_with_jitter(sleep_s)
                    continue

                logger.warning("DOWNLOAD FAIL status=%s ctype=%s url=%s", status, ctype, url)
                dest.unlink(missing_ok=True)
                return status, ctype

        except Exception as e:
            if attempt <= max_retries:
                sleep_s = min(120.0, (2 ** (attempt - 1)) * max(base_delay, 2.0))
                logger.warning("DOWNLOAD EXCEPTION retry %d/%d sleep=%.1fs %s: %s",
                               attempt, max_retries, sleep_s, type(e).__name__, str(e)[:160])
                dest.unlink(missing_ok=True)
                _sleep_with_jitter(sleep_s)
                continue
            logger.error("DOWNLOAD EXCEPTION %s: %s", type(e).__name__, str(e)[:200])
            dest.unlink(missing_ok=True)
            return 0, ""


def api_get(params: Dict[str, Any], base_delay: float, maxlag: int = 5, timeout: int = 60, max_retries: int = 8) -> Dict[str, Any]:
    """
    MediaWiki API GET with retries for:
    - HTTP 429/503/504
    - JSON 'maxlag' errors
    """
    params = dict(params)
    params["maxlag"] = maxlag

    attempt = 0
    while True:
        attempt += 1
        try:
            r = SESSION.get(API_URL, params=params, timeout=timeout)
            status = r.status_code

            if status == 200:
                data = r.json()
                err = data.get("error") if isinstance(data, dict) else None
                if err and str(err.get("code", "")).lower() == "maxlag" and attempt <= max_retries:
                    sleep_s = min(120.0, (2 ** (attempt - 1)) * max(base_delay, 2.0))
                    logger.warning("API maxlag retry %d/%d sleep=%.1fs", attempt, max_retries, sleep_s)
                    _sleep_with_jitter(sleep_s)
                    continue
                return data

            if status in (429, 503, 504) and attempt <= max_retries:
                ra = r.headers.get("Retry-After")
                if ra and ra.isdigit():
                    sleep_s = float(int(ra))
                else:
                    sleep_s = min(120.0, (2 ** (attempt - 1)) * max(base_delay, 2.0))
                logger.warning("API RETRY %d/%d status=%s sleep=%.1fs", attempt, max_retries, status, sleep_s)
                _sleep_with_jitter(sleep_s)
                continue

            r.raise_for_status()
            return {}  # unreachable

        except Exception as e:
            if attempt <= max_retries:
                sleep_s = min(120.0, (2 ** (attempt - 1)) * max(base_delay, 2.0))
                logger.warning("API EXCEPTION retry %d/%d sleep=%.1fs %s: %s",
                               attempt, max_retries, sleep_s, type(e).__name__, str(e)[:160])
                _sleep_with_jitter(sleep_s)
                continue
            raise


# ---------------- Commons API (search + category) ----------------
def search_commons(query: str, limit: int, iiurlwidth: int, delay_sec: float) -> List[Dict[str, Any]]:
    data = api_get(
        {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": limit,
            "gsrnamespace": 6,  # File:
            "prop": "imageinfo",
            "iiprop": "url|mime|extmetadata",
            "iiurlwidth": iiurlwidth,
        },
        base_delay=delay_sec,
    )
    return list(data.get("query", {}).get("pages", {}).values())


def category_members(category: str, cmtype: str, limit: int, delay_sec: float) -> Iterable[Dict[str, Any]]:
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": category,
        "cmtype": cmtype,
        "cmlimit": limit,
    }
    while True:
        data = api_get(params, base_delay=delay_sec)
        for item in data.get("query", {}).get("categorymembers", []):
            yield item
        cont = data.get("continue", {})
        if not cont or "cmcontinue" not in cont:
            break
        params["cmcontinue"] = cont["cmcontinue"]


def imageinfo_for_titles(titles: List[str], thumb_width: int, delay_sec: float) -> List[Dict[str, Any]]:
    if not titles:
        return []
    data = api_get(
        {
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "titles": "|".join(titles),
            "iiprop": "url|mime|extmetadata",
            "iiurlwidth": thumb_width,
        },
        base_delay=delay_sec,
    )
    pages = data.get("query", {}).get("pages", {})
    return list(pages.values())


def crawl_categories_for_files(start_categories: List[str], max_files: int, delay_sec: float, max_depth: int = 3) -> Iterable[Tuple[str, int]]:
    """
    BFS crawl categories up to max_depth. Yields (file_title, depth).
    """
    q = deque([(c, 0) for c in start_categories])
    seen_cats: Set[str] = set()
    yielded = 0

    while q and yielded < max_files:
        cat, depth = q.popleft()
        if cat in seen_cats:
            continue
        seen_cats.add(cat)

        logger.info("Exploring category: %s (depth=%d)", cat, depth)

        for member in category_members(cat, cmtype="file|subcat", limit=200, delay_sec=delay_sec):
            ns = member.get("ns")
            title = member.get("title")
            if not title:
                continue

            if ns == 6 and title.startswith("File:"):
                yield title, depth
                yielded += 1
                if yielded >= max_files:
                    break

            if ns == 14 and title.startswith("Category:") and depth < max_depth and title not in seen_cats:
                q.append((title, depth + 1))


# ---------------- Core ingest ----------------
def stable_id(page: Dict[str, Any], info: Dict[str, Any], fallback: str) -> str:
    # stable across thumb widths / modes
    key = info.get("url") or page.get("title") or fallback
    return hashlib.sha256(str(key).encode("utf-8")).hexdigest()[:16]


def process_page(
    collection,
    page: Dict[str, Any],
    *,
    subject: str,
    topic: str,
    grade_min: int,
    grade_max: int,
    delay_sec: float,
    skipped: Dict[str, int],
) -> bool:
    info = (page.get("imageinfo") or [{}])[0]
    meta = info.get("extmetadata", {}) or {}

    mime = info.get("mime")
    if not is_allowed_mime(mime):
        skipped["mime"] += 1
        return False

    license_name = (meta.get("LicenseShortName", {}) or {}).get("value")
    if not license_allowed(license_name):
        skipped["license"] += 1
        return False

    thumburl = info.get("thumburl")
    if not thumburl:
        skipped["no_thumburl"] += 1
        return False

    _id = stable_id(page, info, thumburl)

    # Download temp
    tmp = SAVE_ROOT / f"tmp_{_id}"
    status, ctype = download_with_backoff(tmp, thumburl, base_delay=delay_sec, max_retries=8)
    if status != 200 or not (ctype.startswith("image/")):
        skipped["download"] += 1
        tmp.unlink(missing_ok=True)
        _sleep_with_jitter(max(delay_sec, 3.5))
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
        logger.error("PROCESS FAIL %s: %s", type(e).__name__, str(e)[:160])
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
        "mime": mime,
        "source_title": page.get("title"),
        "source_page": f"https://commons.wikimedia.org/wiki/{str(page.get('title','')).replace(' ', '_')}",
        "review_status": "unreviewed",
        # extra safety for search scripts:
        "file_path": str(full_path),
        "thumb_path": str(thumb_path),
    }
    metadata = safe_meta(raw_metadata)

    try:
        # Store the actual file path in uris for Chroma image loader usage (when supported)
        collection.add(ids=[_id], uris=[str(full_path)], metadatas=[metadata])
        _sleep_with_jitter(delay_sec)
        return True
    except Exception as e:
        logger.error("DB ADD FAIL %s: %s", type(e).__name__, str(e)[:160])
        skipped["db"] += 1
        _sleep_with_jitter(delay_sec)
        return False


# ---------------- Runners ----------------
def run_search_mode(max_add: int, delay_sec: float) -> None:
    ensure_dirs()
    warmup_session()
    col = get_collection()

    logger.info("DB ready | path=%s collection=%s current=%d", DB_PATH, COLLECTION_NAME, col.count())
    print("DB:", DB_PATH, "| Collection:", COLLECTION_NAME, "| Current:", col.count())

    added_total = 0
    skipped = {"no_thumburl": 0, "license": 0, "download": 0, "process": 0, "db": 0, "mime": 0}

    for t in MATH_TOPICS:
        if added_total >= max_add:
            break

        query = t["query"]
        gmin = int(t["grade_min"])
        gmax = int(t["grade_max"])
        limit = int(t.get("limit", 10))

        logger.info("SEARCH topic=%s limit=%d", query, limit)
        print("\nSearching:", query)

        pages = search_commons(query, limit=limit, iiurlwidth=FULL_MAX_SIDE, delay_sec=delay_sec)

        for page in pages:
            if added_total >= max_add:
                break
            ok = process_page(
                col,
                page,
                subject="Math",
                topic=query,
                grade_min=gmin,
                grade_max=gmax,
                delay_sec=delay_sec,
                skipped=skipped,
            )
            if ok:
                added_total += 1
                print("Added:", query, f"({added_total}/{max_add})")

    print("\nDone.")
    print("Added total:", added_total)
    print("Skipped:", skipped)
    print("Total in DB:", col.count())


def run_category_mode(start_categories: List[str], max_add: int, delay_sec: float, thumb_width: int, max_depth: int) -> None:
    ensure_dirs()
    warmup_session()
    col = get_collection()

    logger.info("DB ready | path=%s collection=%s current=%d", DB_PATH, COLLECTION_NAME, col.count())
    print("DB:", DB_PATH, "| Collection:", COLLECTION_NAME, "| Current:", col.count())
    print("Mode: category")
    print("Start categories:", start_categories)
    print("Max add:", max_add, "| Delay:", delay_sec, "| Thumb width:", thumb_width, "| Max depth:", max_depth)

    added_total = 0
    skipped = {"no_thumburl": 0, "license": 0, "download": 0, "process": 0, "db": 0, "mime": 0}

    # Gather file titles (bounded)
    file_titles: List[str] = []
    for title, _depth in crawl_categories_for_files(start_categories, max_files=max_add * 4, delay_sec=delay_sec, max_depth=max_depth):
        file_titles.append(title)
        if len(file_titles) >= max_add * 4:
            break

    print("Collected file titles:", len(file_titles))

    BATCH = 20
    for i in range(0, len(file_titles), BATCH):
        if added_total >= max_add:
            break
        batch = file_titles[i : i + BATCH]
        pages = imageinfo_for_titles(batch, thumb_width=thumb_width, delay_sec=delay_sec)

        for page in pages:
            if added_total >= max_add:
                break
            ok = process_page(
                col,
                page,
                subject="Math",
                topic="category:" + ",".join(start_categories),
                grade_min=0,
                grade_max=12,
                delay_sec=delay_sec,
                skipped=skipped,
            )
            if ok:
                added_total += 1
                if added_total % 10 == 0 or added_total == max_add:
                    print(f"Added so far: {added_total}/{max_add}")

    print("\nCategory ingest complete.")
    print("Added total:", added_total)
    print("Skipped:", skipped)
    print("Total in DB:", col.count())


# ---------------- CLI ----------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Math-only Wikimedia Commons ingestor -> local ChromaDB image library")
    ap.add_argument("--mode", choices=["search", "category"], default="search")
    ap.add_argument("--delay", type=float, default=3.0, help="Seconds between downloads/API pacing")
    ap.add_argument("--max", type=int, default=50, help="Max images to add in this run")

    ap.add_argument("--start", nargs="+", default=["Category:Mathematics"], help="Start categories (category mode)")
    ap.add_argument("--thumb-width", type=int, default=DEFAULT_CATEGORY_THUMB_WIDTH, help="Category mode thumb width")
    ap.add_argument("--max-depth", type=int, default=3, help="Category crawl depth limit")

    args = ap.parse_args()

    if args.mode == "search":
        run_search_mode(max_add=args.max, delay_sec=args.delay)
    else:
        run_category_mode(
            start_categories=args.start,
            max_add=args.max,
            delay_sec=args.delay,
            thumb_width=args.thumb_width,
            max_depth=args.max_depth,
        )


if __name__ == "__main__":
    main()