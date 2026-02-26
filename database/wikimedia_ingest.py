"""
wikimedia_ingest.py (fixed)
Downloads K-12-friendly RASTER images from Wikimedia Commons and inserts them into ChromaDB.

Fixes:
- Verifies download is an actual image via Content-Type + status
- Skips SVG/PDF and other non-raster formats (these often trigger PIL errors)
- Uses streaming download + safe temp filenames
- Adds basic error handling so one bad file won’t stop the run
"""

import hashlib
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

# Accept licenses by substring match
ALLOWED_LICENSES = ["Public domain", "CC0", "CC BY", "CC-BY"]

# Only accept raster content-types (skip svg/pdf)
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}

TOPICS = [
    {"query": "plant cell diagram", "subject": "Biology", "grade_min": 6, "grade_max": 8, "limit": 25},
    {"query": "water cycle diagram", "subject": "Science", "grade_min": 4, "grade_max": 6, "limit": 25},
    {"query": "fractions pie chart", "subject": "Math", "grade_min": 3, "grade_max": 5, "limit": 25},
    {"query": "blank map united states", "subject": "Geography", "grade_min": 3, "grade_max": 8, "limit": 25},
]

API_URL = "https://commons.wikimedia.org/w/api.php"

# IMPORTANT: set a real UA; Wikimedia may throttle generic ones
HEADERS = {
    "User-Agent": "K12ImageIndexer/0.1 (local prototype; contact: you@example.com)"
}

# ----------------------------------------


def ensure_dirs():
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    THUMB_DIR.mkdir(parents=True, exist_ok=True)


def license_allowed(name: str | None) -> bool:
    if not name:
        return False
    low = name.lower()
    return any(x.lower() in low for x in ALLOWED_LICENSES)


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


def search_commons(query: str, limit: int = 20):
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": limit,
        "gsrnamespace": 6,  # File namespace
        "prop": "imageinfo",
        "iiprop": "url|mime|extmetadata",
    }
    r = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    pages = data.get("query", {}).get("pages", {})
    return list(pages.values())


def download_raster_image(url: str, dest: Path) -> str | None:
    """
    Downloads the URL to dest if it is a raster image.
    Returns the Content-Type if OK, else None.
    """
    with requests.get(url, headers=HEADERS, stream=True, timeout=60, allow_redirects=True) as r:
        if r.status_code != 200:
            return None
        ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype not in ALLOWED_CONTENT_TYPES:
            return None

        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 128):
                if chunk:
                    f.write(chunk)
        return ctype


def main():
    ensure_dirs()
    collection = get_collection()

    for topic in TOPICS:
        query = topic["query"]
        limit = int(topic.get("limit", 20))
        print("\nSearching:", query)

        results = search_commons(query, limit=limit)

        for page in results:
            info = page.get("imageinfo", [{}])[0]
            meta = info.get("extmetadata", {}) or {}

            license_name = (meta.get("LicenseShortName", {}) or {}).get("value")
            if not license_allowed(license_name):
                continue

            url = info.get("url")
            mime = info.get("mime")  # e.g. image/svg+xml sometimes
            if not url:
                continue

            # Skip obvious non-raster types early
            if mime and ("svg" in mime or "pdf" in mime):
                continue

            file_id = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]

            tmp_path = SAVE_ROOT / f"temp_{file_id}"
            ctype = download_raster_image(url, tmp_path)
            if not ctype:
                # Not a raster image or download failed
                tmp_path.unlink(missing_ok=True)
                continue

            full_path = FULL_DIR / f"{file_id}.jpg"
            thumb_path = THUMB_DIR / f"{file_id}.jpg"

            try:
                resize_and_save(tmp_path, full_path, FULL_MAX_SIDE)
                resize_and_save(tmp_path, thumb_path, THUMB_MAX_SIDE)
            except UnidentifiedImageError:
                tmp_path.unlink(missing_ok=True)
                continue
            except Exception as e:
                print("Image processing failed:", url, "error:", e)
                tmp_path.unlink(missing_ok=True)
                continue
            finally:
                tmp_path.unlink(missing_ok=True)

            metadata = {
                "subject": topic["subject"],
                "grade_min": topic["grade_min"],
                "grade_max": topic["grade_max"],
                "license": license_name,
                "source_url": url,
                "source_page": f"https://commons.wikimedia.org/wiki/{page['title'].replace(' ', '_')}",
                "review_status": "unreviewed",
            }

            try:
                collection.add(ids=[file_id], uris=[str(full_path)], metadatas=[metadata])
                print("Added:", full_path.name, "|", license_name)
            except Exception as e:
                print("Chroma add failed:", full_path, "error:", e)

            time.sleep(0.5)

    print("\nDone! Total images in DB:", collection.count())


if __name__ == "__main__":
    main()