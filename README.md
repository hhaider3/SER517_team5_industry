***Some files in this repo serve as a copy of the files we've changes in the main chatbot repo. They can be accessed here https://github.com/johnleddoMETY/chatbot/tree/image-integration***

# SER517_team5_industry - K-12 Image Vector Database

This repository builds and maintains a local, searchable K–12 educational image database using the Wikimedia Commons API, ChromaDB, OpenCLIP, and local image storage.

---

## 🏗️ System Architecture: How It Works

The system is broken down into four main phases:

1. **Data Ingestion (`wikimedia_ingest.py`):** Fetches file metadata from Wikimedia via keyword search or by crawling categories. It downloads images, rasterizes SVGs, resizes them, sanitizes the metadata, and filters out non-image MIME types.
2. **Bulk Ingestion Suite (`database/bulk_ingest.py` & `database/query.txt`):** Takes recursive SQL category dumps (`query.txt`), performs high-throughput asynchronous downloads using `aiohttp`, and utilizes a Producer–Consumer queue with thread offloading for CPU-bound tasks (PIL processing, OpenCLIP embeddings). This design prevents event loop blocking and enables efficient batch ingestion into ChromaDB at scale.
3. **Vector Storage (`init_db.py` & ChromaDB):** Uses `OpenCLIPEmbeddingFunction` to convert images into mathematical vectors. It inserts the image file URIs and metadata into a persistent ChromaDB collection (`k12_education_images`), while saving actual images locally to `./k12_images/full` and `./k12_images/thumb`.
4. **Semantic Search (`search_k12_db_optimized.py`):** Takes a natural language text query, computes its embedding, and queries ChromaDB. It then re-ranks the closest visual matches using soft signals like grade and subject metadata.

---

## ⚡ Pipeline Design & Performance

**Asynchronous Ingestion**
- Uses `aiohttp` and `asyncio` for non-blocking I/O operations  
- Supports concurrent image downloads for improved throughput  

**Multithreaded Processing**
- CPU-intensive tasks (image preprocessing, embedding generation) are executed in background threads  
- Prevents blocking of the async event loop  

**Offline SQL-Driven Pipeline**
- Uses Wikimedia SQL dumps for large-scale ingestion  
- Recursive CTE queries traverse category hierarchies  
- MD5-based reconstruction generates direct image URLs  
- Reduces dependency on API rate limits during bulk ingestion  

**Throughput Characteristics**
- Designed to handle large datasets (10k–100k+ images)  
- Batch embedding and storage into ChromaDB  
- Stable ingestion with retry and backoff strategies

---

## 🧠 Key Engineering Decisions

- Use of vector embeddings (OpenCLIP) for semantic search instead of keyword matching  
- Adoption of asynchronous I/O for scalable data ingestion  
- Hybrid async + multithreading model to balance I/O and CPU workloads  
- Use of offline SQL dumps to enable large-scale ingestion without API bottlenecks  
- Metadata-aware re-ranking for improved educational relevance

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

python database/bulk_ingest.py
```
This mode utilizes a hybrid async + Producer/Consumer threading architecture to process downloads concurrently while offloading embedding and image processing tasks, enabling efficient and stable batch ingestion into ChromaDB.

🔍 Searching the Database
Once you have images downloaded and embedded, you can query your local database. Output includes file paths, subjects, grade ranges, licenses, and attributions.

Basic Search:
```bash
python database/search_k12_db.py "water cycle diagram" --n 6
```

Filtered Search (Subject & Grade soft re-rank):
```bash
python database/search_k12_db.py "fractions pie chart" --grade 4 --subject Math --n 8
```
---

## ⚖️ Licensing & Storage

### Licensing
Allowed licenses:
- Public Domain  
- CC0  
- CC BY  
- CC BY-SA

You are responsible for providing proper attribution when redistributing images.

### Storage
- Full images: up to **1200px**  
- Thumbnails: **256px**

Estimated storage requirements:
- 10k images → ~3–10 GB  
- 100k images → ~30–100 GB

---

## ⚠️ Common Troubleshooting

### ❌ PowerShell Syntax Errors
**Error:** `Missing expression after unary operator '--'` or `Unexpected token`

- **Cause:** Running Bash-style multiline commands (`\`) in Windows PowerShell  
- **Fix:**  
  - Use a single-line command, OR  
  - Replace `\` with backticks `` ` `` for multiline commands in PowerShell  

### ❌ Download Failures / HTTP 429
**Error:** `DOWNLOAD RETRY` or `DOWNLOAD FAIL 429`

- **Cause:** Wikimedia rate limiting  
- **Fix:**  
  - Increase `--delay` to `8.0` or `12.0`  
  - Reduce `--thumb-width`  

### ❌ Image Processing Errors
**Error:** `PIL.UnidentifiedImageError`

- **Cause:** Corrupted downloads or non-image files  
- **Fix:**  
  - Automatically skipped by the pipeline  
  - Increasing `--delay` can reduce occurrence  

### ❌ ChromaDB Insert Failures
**Error:** `DB ADD FAIL`

- **Cause:** Invalid or improperly formatted metadata  
- **Fix:**  
  - Metadata is sanitized automatically  
  - Check logs for unexpected formats if errors persist  

### ❌ HuggingFace Rate Warnings
- **Cause:** OpenCLIP model downloads without authentication  
- **Fix:**  
  ```bash
  export HF_TOKEN="your_huggingface_token_here"

---

## Example Search Output

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
