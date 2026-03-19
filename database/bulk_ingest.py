import asyncio
import aiohttp
import aiofiles
import hashlib
import logging
import urllib.parse
import threading
import queue as thread_queue
from pathlib import Path
from PIL import Image, UnidentifiedImageError

import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
try:
    from chromadb.utils.data_loaders import ImageLoader
except ImportError:
    ImageLoader = None

# --- Configuration ---
INPUT_FILE = "math_filenames.txt"
CONCURRENT_DOWNLOADS = 15
MAX_RETRIES = 3

DB_PATH = "./image_db"
COLLECTION_NAME = "k12_education_images"

SAVE_ROOT = Path("./k12_images")
FULL_DIR = SAVE_ROOT / "full"
THUMB_DIR = SAVE_ROOT / "thumb"

FULL_MAX_SIDE = 1200
THUMB_MAX_SIDE = 256

# --- Threading / Queue Globals ---
batch_queue = thread_queue.Queue()
shutdown_event = threading.Event()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# --- Database Initialization ---
def get_collection():
    """Initializes and returns the ChromaDB collection."""
    client = chromadb.PersistentClient(path=DB_PATH)
    embedding_function = OpenCLIPEmbeddingFunction()
    kwargs = dict(name=COLLECTION_NAME, embedding_function=embedding_function)
    if ImageLoader:
        kwargs["data_loader"] = ImageLoader()
    return client.get_or_create_collection(**kwargs)

# Initialize globally so threads can access it
collection = get_collection()


# --- Helper Functions ---
def get_wikimedia_url(filename: str) -> str:
    """Reconstructs the Wikimedia Commons URL using MD5 hashing."""
    safe_filename = filename.replace(" ", "_")
    md5_hash = hashlib.md5(safe_filename.encode('utf-8')).hexdigest()
    encoded_filename = urllib.parse.quote(safe_filename)
    return f"https://upload.wikimedia.org/wikipedia/commons/{md5_hash[0]}/{md5_hash[:2]}/{safe_filename}"

def resize_image(src: Path, dest: Path, max_side: int):
    """Resizes an image using PIL."""
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = min(max_side / max(w, h), 1.0)
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        im.save(dest, format="JPEG", quality=90, optimize=True)

def db_batch_worker():
    """Reads from batch_queue and inserts into ChromaDB in batches to prevent locking."""
    batch_ids, batch_uris, batch_metas, to_unlink = [], [], [], []
    
    def flush():
        if not batch_ids: return
        try:
            collection.add(ids=batch_ids, uris=batch_uris, metadatas=batch_metas)
            logging.info(f"BATCH INSERTED {len(batch_ids)} images.")
        except Exception as e:
            logging.error(f"BATCH ADD FAIL: {e}")
            for u in batch_uris:
                Path(u).unlink(missing_ok=True)
                thumb_p = Path(u.replace('/full/', '/thumb/'))
                thumb_p.unlink(missing_ok=True)
        finally:
            for p in to_unlink: p.unlink(missing_ok=True)
            batch_ids.clear()
            batch_uris.clear()
            batch_metas.clear()
            to_unlink.clear()

    while not shutdown_event.is_set() or not batch_queue.empty():
        try:
            item = batch_queue.get(timeout=1.0)
            if item is None: continue
            _id, full_path, metadata, tmp_path = item
            batch_ids.append(_id)
            batch_uris.append(str(full_path))
            batch_metas.append(metadata)
            to_unlink.append(tmp_path)
            if len(batch_ids) >= 50: flush()
        except thread_queue.Empty: continue
    flush()

def resize_and_enqueue_sync(filename: str, tmp_path: Path):
    """Resizes the image and pushes it to the batch queue (no direct DB inserts)."""
    safe_filename = filename.replace(" ", "_")
    _id = hashlib.sha256(safe_filename.encode("utf-8")).hexdigest()[:16]

    try:
        if collection.get(ids=[_id]).get("ids"):
            tmp_path.unlink(missing_ok=True)
            return
    except Exception: pass

    full_path = FULL_DIR / f"{_id}.jpg"
    thumb_path = THUMB_DIR / f"{_id}.jpg"

    try:
        resize_image(tmp_path, full_path, FULL_MAX_SIDE)
        resize_image(tmp_path, thumb_path, THUMB_MAX_SIDE)
        metadata = {
            "subject": "Math",
            "topic": "Bulk DB Extract",
            "source_page": f"https://commons.wikimedia.org/wiki/File:{safe_filename}",
            "review_status": "unreviewed"
        }
        batch_queue.put((_id, full_path, metadata, tmp_path))
    except UnidentifiedImageError:
        logging.warning(f"CORRUPT IMAGE OR SVG SKIPPED: {filename}")
        tmp_path.unlink(missing_ok=True)
    except Exception as e:
        logging.error(f"RESIZE FAIL for {filename}: {str(e)[:100]}")
        tmp_path.unlink(missing_ok=True)


# --- Async Download Logic ---
async def download_and_process(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, filename: str):
    """Downloads the file, then hands it off to a background thread for embedding."""
    url = get_wikimedia_url(filename)
    tmp_path = SAVE_ROOT / f"tmp_{filename.replace(' ', '_')}"
    
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        async with aiofiles.open(tmp_path, 'wb') as f:
                            await f.write(await response.read())
                        
                        # Once downloaded, push the heavy embedding work to a thread
                        await asyncio.to_thread(process_and_embed_sync, filename, tmp_path)
                        return
                        
                    elif response.status == 429:
                        logging.warning(f"RATE LIMITED: Backing off for {filename}")
                        await asyncio.sleep(5 * (attempt + 1))
                    else:
                        logging.error(f"FAIL {response.status}: {filename}")
                        return
            except Exception as e:
                logging.error(f"NETWORK ERROR on {filename}: {str(e)}")
                await asyncio.sleep(2)
                
        logging.error(f"GAVE UP: {filename} after {MAX_RETRIES} attempts.")

async def main():
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            filenames = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logging.error(f"Could not find {INPUT_FILE}.")
        return

    logging.info(f"Loaded {len(filenames)} filenames. Starting bulk pipeline...")
    logging.info(f"Current DB size: {collection.count()} images")

    semaphore = asyncio.Semaphore(CONCURRENT_DOWNLOADS)
    headers = {"User-Agent": "K12-Education-Bot/1.0"}

    db_thread = threading.Thread(target=db_batch_worker, daemon=True)
    db_thread.start()
    
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [download_and_process(session, semaphore, filename) for filename in filenames]
        await asyncio.gather(*tasks)

    shutdown_event.set()
    db_thread.join()

    logging.info("Bulk pipeline complete.")
    logging.info(f"New DB size: {collection.count()} images")

if __name__ == "__main__":
    asyncio.run(main())