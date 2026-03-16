import asyncio
import aiohttp
import aiofiles
import hashlib
import logging
from pathlib import Path

INPUT_FILE = "math_filenames.txt"
SAVE_DIR = Path("./math_images_bulk")
CONCURRENT_DOWNLOADS = 15  
MAX_RETRIES = 3

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def get_wikimedia_url(filename: str) -> str:
    """Reconstructs the Wikimedia Commons URL using MD5 hashing."""
    safe_filename = filename.replace(" ", "_")
    md5_hash = hashlib.md5(safe_filename.encode('utf-8')).hexdigest()
    return f"https://upload.wikimedia.org/wikipedia/commons/{md5_hash[0]}/{md5_hash[:2]}/{safe_filename}"

async def download_image(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, filename: str):
    """Downloads a single image asynchronously with retry logic."""
    url = get_wikimedia_url(filename)
    dest_path = SAVE_DIR / filename
    
    if dest_path.exists():
        return  

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        async with aiofiles.open(dest_path, 'wb') as f:
                            await f.write(await response.read())
                        logging.info(f"SUCCESS: {filename}")
                        return
                    elif response.status == 429:
                        logging.warning(f"RATE LIMITED: Backing off for {filename}")
                        await asyncio.sleep(5 * (attempt + 1))
                    else:
                        logging.error(f"FAIL {response.status}: {filename}")
                        return
            except Exception as e:
                logging.error(f"ERROR on {filename}: {str(e)}")
                await asyncio.sleep(2)
                
        logging.error(f"GAVE UP: {filename} after {MAX_RETRIES} attempts.")

async def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            filenames = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logging.error(f"Could not find {INPUT_FILE}. Please run the SQL query first.")
        return
    logging.info(f"Loaded {len(filenames)} filenames. Starting bulk download...")

    semaphore = asyncio.Semaphore(CONCURRENT_DOWNLOADS)
    headers = {"User-Agent": "K12-Education-Bot/1.0 (Contact: your_email@example.com)"}
    
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [download_image(session, semaphore, filename) for filename in filenames]
        await asyncio.gather(*tasks)
    logging.info("Bulk download complete.")

if __name__ == "__main__":
    asyncio.run(main())