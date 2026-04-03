*Some files in this repo serve as a copy of the files we've changes in the main chatbot repo. They can be accessed here https://github.com/johnleddoMETY/chatbot/tree/image-integration

# SER517_team5_industry - K-12 Image Vector Database

A project for SER 517 by Team 5 (industry). 

This repository builds and maintains a local, searchable K–12 educational image database using the Wikimedia Commons API, ChromaDB, OpenCLIP, and local image storage.

---

## 🏗️ System Architecture: How It Works

The system is broken down into four main phases:

1. **Data Ingestion (`wikimedia_ingest.py`):** Fetches file metadata from Wikimedia via keyword search or by crawling categories. It downloads images, rasterizes SVGs, resizes them, sanitizes the metadata, and filters out non-image MIME types.
2. **Bulk Ingestion Suite (`database/bulk_ingest.py` & `database/query.txt`):** Takes recursive SQL category dumps (`query.txt`), handles concurrent downloads via `aiohttp`, and uses a queued, thread-safe pipeline to bypass corrupt vector files (like SVGs) and batch-embed hundreds of images into ChromaDB rapidly.
3. **Vector Storage (`init_db.py` & ChromaDB):** Uses `OpenCLIPEmbeddingFunction` to convert images into mathematical vectors. It inserts the image file URIs and metadata into a persistent ChromaDB collection (`k12_education_images`), while saving actual images locally to `./k12_images/full` and `./k12_images/thumb`.
4. **Semantic Search (`search_k12_db_optimized.py`):** Takes a natural language text query, computes its embedding, and queries ChromaDB. It then re-ranks the closest visual matches using soft signals like grade and subject metadata.

---

## 🚀 Quick Start

### 1. Environment Setup
Create and activate your virtual environment:
**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
Windows:

PowerShell
python -m venv venv
.\venv\Scripts\activate
2. Install Dependencies
Install the required packages using the requirements file:

Bash
pip install -r requirements.txt
(Optional but recommended): Export your Hugging Face token to reduce rate-limit warnings for OpenCLIP downloads:

Bash
export HF_TOKEN="your_huggingface_token_here"
📥 Ingesting Data
The ingestion pipeline respects Wikimedia politely by using maxlag, warmup page visits, and exponential backoff for HTTP 429/503 errors.

Mode A: Keyword Search (Default)
Use this to ingest a small batch of predefined sample topics.

Bash
python database/wikimedia_ingest.py --mode search --delay 4.0
Mode B: Category Crawling (Building the Math Database)
This is the recommended workflow for collecting specific subject images, like K-12 math diagrams. Start small and increase gradually.

For Mac/Linux (Bash):

Bash
python database/wikimedia_ingest.py --mode category \
  --start "Category:Mathematics" \
  --max 500 \
  --delay 5.0 \
  --thumb-width 640 \
  --subject Math \
  --topic-label "category ingest" \
  --grade-min 0 \
  --grade-max 12
For Windows (PowerShell):
Note: PowerShell uses backticks (`) instead of backslashes () for multiline commands.

PowerShell
python database/wikimedia_ingest.py --mode category `
  --start "Category:Mathematics" `
  --max 500 `
  --delay 5.0 `
  --thumb-width 640 `
  --subject Math `
  --topic-label "category ingest" `
  --grade-min 0 `
  --grade-max 12
Recommended Math Seed Categories:
Once you build your base, try swapping the --start parameter with:

Category:Mathematical diagrams

Category:Geometry

Category:Algebra

Category:Mathematical notation

### Mode C: Bulk SQL-Driven Ingestion
For massive-scale scraping without relying entirely on API endpoints, you can utilize the `database/` suite:
1. Run the recursive CTE SQL query inside `database/query.txt` against a Wikimedia database replica.
2. Export the result list to `math_filenames.txt`.
3. Run the concurrent queuing pipeline:
```bash
python database/bulk_ingest.py
```
This mode utilizes a custom Producer/Consumer threading architecture to process and batch-upload images into ChromaDB rapidly and safely.

🔍 Searching the Database
Once you have images downloaded and embedded, you can query your local database. Output includes file paths, subjects, grade ranges, licenses, and attributions.

Basic Search:

Bash
python database/search_k12_db.py "water cycle diagram" --n 6
Filtered Search (Subject & Grade soft re-rank):

Bash
python database/search_k12_db.py "fractions pie chart" --grade 4 --subject Math --n 8
⚠️ Common Troubleshooting
Terminal Error: Missing expression after unary operator '--' or Unexpected token:

Cause: You are running a Bash-formatted multiline command (using \) inside Windows PowerShell.

Fix: Either write the command on a single continuous line, or replace all trailing backslashes (\) with backticks (`).

Terminal Error: DOWNLOAD RETRY or DOWNLOAD FAIL 429:

Cause: Wikimedia is rate-limiting your connection.

Fix: Increase the --delay parameter to 8.0 or 12.0 seconds and reduce the --thumb-width.

Terminal Error: PIL.UnidentifiedImageError on resize:

Cause: Caused by downloading non-image blobs or incomplete downloads.

Fix: The script handles this safely and skips the image, but increasing the --delay can prevent incomplete downloads in the future.

Terminal Error: DB ADD FAIL messages:

Cause: Often caused by incorrect metadata types being passed to ChromaDB.

Fix: The current script automatically sanitizes metadata to avoid this, but check the error log output to see if an unexpected string format slipped through.

HuggingFace rate warnings:

Cause: Downloading the OpenCLIP model without authentication.

Fix: Set HF_TOKEN in your environment variables.

⚖️ Licensing & Storage
Licensing: The pipeline strictly filters images to ensure the license contains allowed substrings: public domain, cc0, cc by, or cc by-sa. You are responsible for showing proper attribution when redistributing.

Storage: Full images are up to 1200px; thumbnails are 256px. Rough sizing estimates: 10k images ≈ 3–10 GB total; 100k images ≈ 30–100 GB total.


### Example Search Output

When running a basic search for `"water cycle diagram"`, the console will output the closest semantic matches along with their metadata:

```text
Query: "water cycle diagram"
Retrieving top 3 matches...

1. File: ./k12_images/full/Water_Cycle_Diagram.jpg
   Score: 0.89
   Subject: Science
   Grade Range: 3-6
   License: CC BY-SA 4.0
   Attribution: Wikimedia Commons

2. File: ./k12_images/full/Hydrologic_Cycle.png
   Score: 0.85
   Subject: Science
   Grade Range: 4-8
   License: Public Domain
   Attribution: USGS / Wikimedia Commons
