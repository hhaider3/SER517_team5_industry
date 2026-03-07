# SER517_team5_industry
Team project for SER 517 by Team 5(industry)

# K-12 Image Vector Database 

This repository builds and maintains a local, searchable K–12 educational image database using:

Wikimedia Commons API (for metadata and thumbnails)

ChromaDB (persistent vector store)

OpenCLIP (image/text embeddings)

Local image storage (./k12_images/full and ./k12_images/thumb)

Included scripts:

wikimedia_ingest.py — ingestion pipeline using Wikimedia keyword search (predefined TOPICS list). Downloads images, creates full + thumbnail JPEGs, and inserts URIs + metadata into ChromaDB.

search_k12_db_optimized.py — CLI semantic search over the local ChromaDB (metadata filtering + optional strict grade overlap + soft grade-aware reranking).

init_db.py (optional) — create/open Chroma collection (if you have one).

k12_images/ — local image storage created by the ingest script.

image_db/ — Chroma persistent database folder created automatically.

Quick start
1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate
2. Install Python dependencies
pip install -r requirements.txt

If you don’t have requirements.txt, install the main packages:

pip install chromadb requests pillow

Optional but recommended (reduces rate-limit warnings for OpenCLIP downloads):

export HF_TOKEN="your_huggingface_token_here"
3. Prepare directories

The ingest script will create directories automatically, but you can inspect or create them manually:

./k12_images/full
./k12_images/thumb
./image_db
How the system works (short)

wikimedia_ingest.py fetches file metadata from Wikimedia (either by keyword search — search mode — or by crawling categories — category mode), downloads thumbnails (rasterized for SVGs), resizes images, sanitizes metadata, and inserts image file URI + metadata into ChromaDB using OpenCLIP embeddings.

search_k12_db.py computes an embedding for your text query, queries ChromaDB, then re-ranks results by grade/subject soft signals and prints file paths and attribution metadata.

Using wikimedia_ingest.py

wikimedia_ingest.py supports two modes:

Mode A — search (default)

Use this to ingest a handful of sample topics (the file contains TOPICS with example queries).

python wikimedia_ingest.py --mode search

You can also tune delay:

python wikimedia_ingest.py --mode search --delay 4.0
Mode B — category (crawl categories like Category:Mathematics)

This is the mode for collecting math-related images. Start small and increase gradually.

Basic example:

python wikimedia_ingest.py --mode category \
  --start "Category:Mathematics" \
  --max 200 \
  --delay 5.0 \
  --thumb-width 640 \
  --subject Math \
  --topic-label "category ingest" \
  --grade-min 0 \
  --grade-max 12

Parameters (category mode)

--start — one or more start categories, e.g. "Category:Mathematics" "Category:Mathematical diagrams".

--max — maximum number of files to add this run (recommended small at first: 200–1000).

--delay — base seconds to sleep between downloads. Increase to 5–10s if you see 429 errors.

--thumb-width — width (px) requested from Wikimedia for thumbnails (smaller reduces CDN load).

--subject / --topic-label / --grade-min / --grade-max — metadata values to apply to ingested images.

Important behavior notes

The script honors Wikimedia politely: warmup page visit, maxlag on API calls, and exponential backoff + Retry-After handling if the CDN returns 429/503.

The script filters out non-image MIME types (video, audio), and will optionally accept PDFs if enabled in the code.

Metadata is sanitized so it is compatible with ChromaDB (strings / numbers only).

The script avoids re-adding files that are already in the DB.

Using search_k12_db.py

Basic usage:

python search_k12_db.py "water cycle diagram" --n 6

With grade and subject filters (grade used as soft re-rank):

python search_k12_db.py "fractions pie chart" --grade 4 --subject Math --n 8

Output includes: file path, subject, topic, grade range, license, cleaned artist/credit, and source page.

Recommended workflow for building a large math collection

Start with a curated seed list of math categories:

Category:Mathematics

Category:Mathematical diagrams

Category:Geometry

Category:Algebra

Category:Mathematical notation

Run the category mode with conservative parameters:

python wikimedia_ingest.py --mode category --start "Category:Mathematics" --max 500 --delay 5.0 --thumb-width 640

Monitor output; if you start seeing many DOWNLOAD RETRY or DOWNLOAD FAIL 429, increase --delay to 8–12s and/or reduce --thumb-width.

After you have a local cache of commonly-used images, use search_k12_db.py to validate search quality and create an approval workflow (review review_status in metadata).

For very large-scale indexing (10k+ metadata rows), switch to a catalog-first approach (collect metadata only in SQLite/Postgres, then download images on demand or via controlled workers). The current scripts are intended for initial/catalog+cache workflows.

Storage & sizing

Images saved as JPEG after resizing. Full images: up to 1200px longest side (configurable). Thumbnails: 256px.

Typical storage:

~0.3–1 MB per full image (varies).

Embeddings are small (a few KB each).

Rough estimate:

10k images ≈ 3–10 GB total (images + embeddings)

100k images ≈ 30–100 GB total
Adjust based on desired resolution and thumbnail sizes.

Licensing & attribution

The ingest pipeline filters images to only ingest those whose license contains one of the allowed substrings: public domain, cc0, cc by, cc by-sa.

The script stores license and attribution metadata (artist, credit, source page). You are responsible for showing proper attribution when redistributing images.

If you want to expand or restrict licenses, edit ALLOWED_LICENSE_SUBSTRINGS in wikimedia_ingest.py.

Handling rate limits and blocks

If Wikimedia responds with 429 (Too Many Requests) or other transient errors:

Increase --delay and/or --thumb-width (smaller thumbs).

Re-run with a smaller --max and let the script complete; it will skip items already in DB.

If you expect to run very large crawls, consider:

Running several small jobs over time (cron / overnight).

Adding a resume checkpoint (I can add a --resume-file flag to save progress).

Using multiple IPs only if you are authorized (do not circumvent rate limits).

Advanced: scaling tips

Move metadata-only ingestion into a separate catalog (SQLite/Postgres) to store titles, thumb URLs, and metadata without downloading images. This allows large-scale coverage without heavy disk use.

Implement on-demand caching: when a user searches, fetch and embed top N images at query time and store them locally.

Consider a distributed worker queue with per-worker polite delays for high-throughput ingestion, plus a single shared catalog.

Troubleshooting

PIL.UnidentifiedImageError on resize — caused by downloading non-image blobs or incomplete downloads. Increase delay, or check the preview printed in the log. The updated script uses backoff and mime filtering to reduce these cases.

Many DB ADD FAIL messages — inspect the error printed; often caused by incorrect metadata types (the updated script sanitizes metadata).

HuggingFace rate warnings — set HF_TOKEN in your environment to avoid these warnings and increase embed weight download limits:

export HF_TOKEN="your_token_here"
