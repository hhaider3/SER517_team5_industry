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

import requests
from PIL import Iamge, UnidentifiedImageError

import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

try:
    from chromadb.utils.data_loaders import ImageLoader
except Exception:
    ImageLoader = None

try:
    import cairosvg
except Exception:
    cairosvg = None 
# ---------------- CONFIG ----------------

DB_PATH = "./image_db"
COLLECTION_NAME = "k12_education_images"

SAVE_ROOT = Path("./k12_images")
FULL_DIR = SAVE_ROOT / "full"
THUMB_DIR = SAVE_ROOT / "thumb"

FULL_MAX_SIDE = 1200
THUMB_MAX_SIDE = 256

# Accept licenses by substring match
ALLOWED_LICENSE_SUBSTRINGS = [
    "public domain",
    "cc0",
    "cc by",
    "cc-by",
    "cc by-sa",
    "cc-by-sa",
]

# Only accept raster content-types (skip svg/pdf)
ALLOWED_RASTER_CTYPES = {
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
HEADERS = {
    "User-Agent": "K12ImageIndexer/0.1 (local prototype; contact: you@example.com)"
}

def ensure_dirs():
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)


def license_allowed(name: str | None) -> bool:
    if not name:
        return False
    low = name.lower()
    return any(s in low for s in ALLOWED_LICENSE_SUBSTRINGS)


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
        "gsrnamespace": 6,  
        "prop": "imageinfo",
        "iiprop": "url|mime|extmetadata",
    }
    r = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    return list(data.get("query", {}).get("pages", {}).values())


def download(url: str, dest: Path) -> tuple[int,str]:
    """
    Downloads the URL to dest if it is a raster image.
    Returns the Content-Type if OK, else None.
    """
    with requests.get(url, headers=HEADERS, stream=True, timeout=60, allow_redirects=True) as r:
        status = r.status_code
        ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if status != 200:
            return status, ctype
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 128):
                if chunk:
                    f.write(chunk)
        return status, ctype

def svg_to_pnd(svg_path: Path, png_path: Path):
    if cairosvg is None:
        raise RuntimeError("cairosvg not installed. Run: pip install cairosvg")
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path))

def already_in_db(collection, _id: str) -> bool:
    try:
        got = collection.get(ids=[_id])
        return bool(got.get("ids"))
    except Exception:
        return False


def main():
    ensure_dirs()
    collection = get_collection()
    print("DB:", DB_PATH, "| Collection:", COLLECTION_NAME, "| Current:", collection.count())
    if cairosvg is None:
        print("NOTE: cairosvg not installed. SVG diagrams will be skipped (likely many results).")
    
    added_total = 0

    for topic in TOPICS:
        query = topic["query"]
        limit = int(topic.get("limit", 20))
        print("\nSearching:", query)

        results = search_commons(query, limit)
        added_topic = 0
        skipped = {"no_url": 0, "license": 0, "ctype": 0, "svg_no_cairosvg": 0, "process": 0, "db": 0}


        for page in results:
            info = page.get("imageinfo", [{}])[0]
            meta = info.get("extmetadata", {}) or {}

            url = info.get("url")
            if not url:
                skipped["no_url"] += 1
                continue

            license_name = (meta.get("LicenseShortName", {}) or {}).get("value")
            if not license_allowed(license_name):
                skipped["license"] += 1
                continue

            file_id = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
            if already_in_db(collection, file_id):
                continue

            tmp = SAVE_ROOT / f"tmp_{file_id}"
            status,ctype = download(url, tmp)
            if status != 200:
                skipped["ctype"] += 1
                tmp.unlink(missing_ok=True)
                continue

            work_input = tmp
            converted_png = None

            if ctype in ("image/svg+xml",) or url.lower().endswith(".svg"):
                if cairosvg is None:
                    skipped["svg_no_cairosvg"] += 1
                    tmp.unlink(missing_ok=True)
                    continue
                converted_png = SAVE_ROOT / f"tmp_{file_id}.png"
                try:
                    svg_to_png(tmp, converted_png)
                    work_input = converted_png
                except Exception:
                    skipped["process"] += 1
                    tmp.unlink(missing_ok=True)
                    converted_png.unlink(missing_ok=True)
                    continue

            elif ctype not in ALLOWED_RASTER_CTYPES:
                skipped["ctype"] += 1
                tmp.unlink(missing_ok=True)
                if converted_png:
                    converted_png.unlink(missing_ok=True)
                continue

            full_path = FULL_DIR / f"{file_id}.jpg"
            thumb_path = THUMB_DIR / f"{file_id}.jpg"


            try:
                resize_and_save(tmp_path, full_path, FULL_MAX_SIDE)
                resize_and_save(tmp_path, thumb_path, THUMB_MAX_SIDE)
            except UnidentifiedImageError:
                skipped["process"] += 1
                tmp_path.unlink(missing_ok=True)
                if converted_png:
                    converted_png.unlink(missing_ok=True)
                continue
            except Exception:
                skipped["process"] += 1 
                tmp.unlink(missing_ok=True)
                if converted_png:
                    converted_png.unlink(missing_ok=True)
                continue
            finally:
                tmp_path.unlink(missing_ok=True)
                if converted_png:
                    converted_png.unlink(missing_ok=True)

            metadata = {
                "subject": topic["subject"],
                "topic": topic.get("topic", query),
                "grade_min": topic["grade_min"],
                "grade_max": topic["grade_max"],
                "license": license_name,
                "license_url": (meta.get("LicenseUrl", {}) or {}).get("value"),
                "artist": (meta.get("Artist", {}) or {}).get("value"),
                "credit": (meta.get("Credit", {}) or {}).get("value"),
                "source_url": url,
                "source_page": f"https://commons.wikimedia.org/wiki/{page['title'].replace(' ', '_')}",
                "review_status": "unreviewed",
            }

            try:
                collection.add(ids=[file_id], uris=[str(full_path)], metadatas=[metadata])
                added_topic += 1
                added_total += 1
            except Exception:
                skipped["db"] += 1

            time.sleep(0.35)

        print(f"Added for topic: {added_topic}")
        print("Skipped:", skipped)

    print("\nDone! Total added:", added_total)
    print("Total images in DB:", collection.count())


if __name__ == "__main__":
    main()