"""
wikimedia_ingest.py (UPDATED - session + warmup + thumbs + metadata sanitizing)

Fixes Chroma error:
  argument 'metadatas': Cannot convert Python object to MetadataValue

Cause:
- Wikimedia extmetadata fields can contain HTML or non-primitive values.
- Chroma only allows str/int/float/bool in metadatas.

This version:
- casts metadata values to safe strings
- removes None
- limits long strings
"""

import hashlib
import re
import time
from pathlib import Path

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

SLEEP_BETWEEN_DOWNLOADS_SEC = 3.0

ALLOWED_LICENSE_SUBSTRINGS = [
    "public domain",
    "cc0",
    "cc by",
    "cc-by",
    "cc by-sa",
    "cc-by-sa",
]

TOPICS = [
    {"query": "plant cell diagram", "subject": "Biology", "grade_min": 6, "grade_max": 8, "limit": 10},
    {"query": "water cycle diagram", "subject": "Science", "grade_min": 4, "grade_max": 6, "limit": 10},
    {"query": "fractions pie chart", "subject": "Math", "grade_min": 3, "grade_max": 5, "limit": 10},
    {"query": "blank map united states", "subject": "Geography", "grade_min": 3, "grade_max": 8, "limit": 10},
]

API_URL = "https://commons.wikimedia.org/w/api.php"

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

# ----------------------------------------


def ensure_dirs():
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)


def warmup_session():
    try:
        SESSION.get("https://commons.wikimedia.org/wiki/Main_Page", timeout=30)
        time.sleep(1.0)
    except Exception:
        pass


def license_allowed(name: str | None) -> bool:
    if not name:
        return False
    low = str(name).lower()
    return any(s in low for s in ALLOWED_LICENSE_SUBSTRINGS)


_TAG_RE = re.compile(r"<[^>]+>")


def _to_safe_str(v) -> str:
    """
    Convert any value to a safe short string:
    - cast to str
    - strip HTML tags (basic)
    - collapse whitespace
    - limit length
    """
    if v is None:
        return ""
    s = str(v)
    s = _TAG_RE.sub(" ", s)  # remove html tags
    s = re.sub(r"\s+", " ", s).strip()
    # Chroma metadata values should be reasonably small
    if len(s) > 500:
        s = s[:500] + "…"
    return s


def safe_meta(d: dict) -> dict:
    """
    Ensure metadatas only contain primitives Chroma accepts:
    - str/int/float/bool (no None, no dict/list)
    """
    out = {}
    for k, v in (d or {}).items():
        if v is None:
            continue
        # allow numeric/bool as-is
        if isinstance(v, (int, float, bool)):
            out[k] = v
        else:
            s = _to_safe_str(v)
            if s != "":
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


def search_commons(query: str, limit: int):
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": limit,
        "gsrnamespace": 6,
        "prop": "imageinfo",
        "iiprop": "url|mime|extmetadata",
        "iiurlwidth": FULL_MAX_SIDE,  # makes thumburl appear
    }
    r = SESSION.get(API_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    return list(data.get("query", {}).get("pages", {}).values())


def download(url: str, dest: Path) -> tuple[int, str]:
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


def already_in_db(collection, _id: str) -> bool:
    try:
        got = collection.get(ids=[_id])
        return bool(got.get("ids"))
    except Exception:
        return False


def main():
    ensure_dirs()
    warmup_session()

    collection = get_collection()
    print("DB:", DB_PATH, "| Collection:", COLLECTION_NAME, "| Current:", collection.count())

    added_total = 0

    for topic in TOPICS:
        query = topic["query"]
        limit = int(topic.get("limit", 10))
        print("\nSearching:", query)

        results = search_commons(query, limit)

        added_topic = 0
        skipped = {"no_thumburl": 0, "license": 0, "download": 0, "process": 0, "db": 0}

        for page in results:
            info = page.get("imageinfo", [{}])[0]
            meta = info.get("extmetadata", {}) or {}

            license_name = (meta.get("LicenseShortName", {}) or {}).get("value")
            if not license_allowed(license_name):
                skipped["license"] += 1
                continue

            thumburl = info.get("thumburl")
            if not thumburl:
                skipped["no_thumburl"] += 1
                continue

            _id = hashlib.sha256(thumburl.encode("utf-8")).hexdigest()[:16]
            if already_in_db(collection, _id):
                continue

            tmp = SAVE_ROOT / f"tmp_{_id}"
            status, _ctype = download(thumburl, tmp)
            if status != 200:
                skipped["download"] += 1
                tmp.unlink(missing_ok=True)
                time.sleep(max(SLEEP_BETWEEN_DOWNLOADS_SEC, 6.0))
                continue

            full_path = FULL_DIR / f"{_id}.jpg"
            thumb_path = THUMB_DIR / f"{_id}.jpg"

            try:
                resize_and_save(tmp, full_path, FULL_MAX_SIDE)
                resize_and_save(tmp, thumb_path, THUMB_MAX_SIDE)
            except UnidentifiedImageError:
                skipped["process"] += 1
                tmp.unlink(missing_ok=True)
                continue
            except Exception as e:
                print("PROCESS FAIL", str(e)[:160])
                skipped["process"] += 1
                tmp.unlink(missing_ok=True)
                continue
            finally:
                tmp.unlink(missing_ok=True)

            raw_metadata = {
                "subject": topic["subject"],
                "topic": topic.get("topic", query),
                "grade_min": int(topic["grade_min"]),
                "grade_max": int(topic["grade_max"]),
                "license": license_name,
                "license_url": (meta.get("LicenseUrl", {}) or {}).get("value"),
                "artist": (meta.get("Artist", {}) or {}).get("value"),
                "credit": (meta.get("Credit", {}) or {}).get("value"),
                "source_url": info.get("url"),
                "thumb_url": thumburl,
                "source_page": f"https://commons.wikimedia.org/wiki/{page['title'].replace(' ', '_')}",
                "review_status": "unreviewed",
            }
            metadata = safe_meta(raw_metadata)

            try:
                collection.add(ids=[_id], uris=[str(full_path)], metadatas=[metadata])
                added_topic += 1
                added_total += 1
                print("Added:", full_path.name, "|", metadata.get("license", ""))
            except Exception as e:
                print("DB ADD FAIL", str(e)[:160])
                skipped["db"] += 1

            time.sleep(SLEEP_BETWEEN_DOWNLOADS_SEC)

        print(f"Added for topic: {added_topic}")
        print("Skipped:", skipped)

    print("\nDone! Total added:", added_total)
    print("Total images in DB:", collection.count())


if __name__ == "__main__":
    main()