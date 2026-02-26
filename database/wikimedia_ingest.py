"""
wikimedia_ingest.py
Downloads K-12 images from Wikimedia Commons and inserts them into ChromaDB.
"""

import hashlib
import json
import time
from pathlib import Path
import requests
from PIL import Image

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

ALLOWED_LICENSES = ["Public domain", "CC0", "CC BY", "CC-BY"]

TOPICS = [
    {"query": "plant cell diagram", "subject": "Biology", "grade_min": 6, "grade_max": 8},
    {"query": "water cycle diagram", "subject": "Science", "grade_min": 4, "grade_max": 6},
    {"query": "fractions pie chart", "subject": "Math", "grade_min": 3, "grade_max": 5},
    {"query": "blank map united states", "subject": "Geography", "grade_min": 3, "grade_max": 8},
]

API_URL = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "K12ImageIndexer/0.1"}

# ----------------------------------------

def ensure_dirs():
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    THUMB_DIR.mkdir(parents=True, exist_ok=True)


def license_allowed(name):
    if not name:
        return False
    return any(l.lower() in name.lower() for l in ALLOWED_LICENSES)


def resize_and_save(src, dest, max_side):
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = min(max_side / max(w, h), 1.0)
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)))
        im.save(dest, format="JPEG", quality=90)


def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    embedding_function = OpenCLIPEmbeddingFunction()

    kwargs = dict(name=COLLECTION_NAME, embedding_function=embedding_function)
    if ImageLoader:
        kwargs["data_loader"] = ImageLoader()

    return client.get_or_create_collection(**kwargs)


def search_commons(query, limit=20):
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": limit,
        "gsrnamespace": 6,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata"
    }

    r = requests.get(API_URL, params=params, headers=HEADERS)
    data = r.json()

    if "query" not in data:
        return []

    return data["query"]["pages"].values()


def main():
    ensure_dirs()
    collection = get_collection()

    for topic in TOPICS:
        print("\nSearching:", topic["query"])
        results = search_commons(topic["query"])

        for page in results:
            info = page["imageinfo"][0]
            meta = info.get("extmetadata", {})

            license_name = meta.get("LicenseShortName", {}).get("value")
            if not license_allowed(license_name):
                continue

            url = info["url"]
            file_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
            temp_path = SAVE_ROOT / "temp.jpg"

            # Download
            img_data = requests.get(url).content
            with open(temp_path, "wb") as f:
                f.write(img_data)

            full_path = FULL_DIR / f"{file_hash}.jpg"
            thumb_path = THUMB_DIR / f"{file_hash}.jpg"

            resize_and_save(temp_path, full_path, FULL_MAX_SIDE)
            resize_and_save(temp_path, thumb_path, THUMB_MAX_SIDE)

            metadata = {
                "subject": topic["subject"],
                "grade_min": topic["grade_min"],
                "grade_max": topic["grade_max"],
                "license": license_name,
                "source_url": url,
                "source_page": f"https://commons.wikimedia.org/wiki/{page['title']}",
            }

            collection.add(
                ids=[file_hash],
                uris=[str(full_path)],
                metadatas=[metadata],
            )

            print("Added:", full_path.name)

            temp_path.unlink(missing_ok=True)
            time.sleep(0.5)

    print("\nDone!")
    print("Total images in DB:", collection.count())


if _name_ == "_main_":
    main()