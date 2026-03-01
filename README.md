# SER517_team5_industry
Team project for SER 517 by Team 5(industry)

# K-12 Image Vector Database
A local, searchable K-12 education image database built using:

Wikimedia Commons API

ChromaDB (vector database)

OpenCLIP embeddings

Python ingestion and search scripts

This system allows you to:

Download licensed educational images

Store them locally

Index them with vector embeddings

Perform semantic search with grade-aware re-ranking


# Overview
This project consists of three main scripts:

init_db.py – Initializes the Chroma vector database

wikimedia_ingest.py – Downloads and indexes images from Wikimedia Commons

search_k12_db.py – Performs semantic search over the local database

The database is stored locally and persists between runs.

# Project Structure
database/
│
├── image_db/                # Chroma persistent vector database
├── k12_images/
│   ├── full/                # Resized indexed images
│   └── thumb/               # Small thumbnails
│
├── init_db.py
├── wikimedia_ingest.py
├── search_k12_db.py
└── README.md
Installation

Create a virtual environment (recommended):

python3 -m venv venv
source venv/bin/activate

Install dependencies:

pip install chromadb requests pillow

Optional (reduces HuggingFace rate warnings for embeddings):

export HF_TOKEN="your_token_here"

# Steps to run
Step 1: Initialize the Database

Run:

python init_db.py

This creates the persistent Chroma collection in:

./image_db/

The script is safe to run multiple times.

Step 2: Ingest Images from Wikimedia

Edit the TOPICS list inside wikimedia_ingest.py if you want to change what gets downloaded.

Run:

python wikimedia_ingest.py

The ingestion script will:

Search Wikimedia Commons

Filter by allowed licenses

Download raster thumbnails

Resize images

Store files locally

Insert them into ChromaDB with metadata

Images are stored in:

./k12_images/full/
./k12_images/thumb/
x
Step 3: Search the Database

Example searches:

python search_k12_db.py "water cycle diagram"
python search_k12_db.py "fractions pie chart" --grade 4
python search_k12_db.py "blank map united states" --subject Geography

Example output:

#1  score=1.351  dist=0.649
File: k12_images/full/0775599a32381e8e.jpg
Subject: Science
Topic: water cycle diagram
Grades: 4 - 6
License: CC BY-SA 3.0
Artist: NASA Scientific Visualization Studio
Source page: https://commons.wikimedia.org/wiki/File:...
Metadata Schema

Each image entry contains:

Field	Description
subject	Subject category
topic	Topic bucket used during ingestion
grade_min	Minimum recommended grade
grade_max	Maximum recommended grade
license	Wikimedia license
license_url	Link to license
artist	Cleaned attribution text
credit	Credit text
source_url	Original image URL
thumb_url	Thumbnail URL used
source_page	Wikimedia file page
review_status	Default is "unreviewed"

All metadata is sanitized to ensure compatibility with ChromaDB.

Storage Considerations

Images are resized to:

Full images: max 1200px

Thumbnails: 256px

Typical storage estimates:

300 KB – 1 MB per image

~3 KB per embedding

Approximate disk usage:

10,000 images ≈ 3–10 GB total

100,000 images ≈ 30–100 GB total

The vector database itself is small compared to the image files.

Licensing

The ingestion script only downloads images with licenses that contain:

Public domain

CC0

CC-BY

CC-BY-SA

License and attribution information is stored in metadata.

You are responsible for displaying proper attribution if redistributing images.

Wikimedia Robot Policy Notes

Wikimedia may block aggressive automated downloads.

Mitigations implemented:

Browser-like headers

Session with cookies

Warm-up request

3-second delay between downloads

If downloads fail with 403:

Increase the delay between downloads

Reduce batch size

Try a different network

Avoid large-scale bulk ingestion

Architecture Summary
User Query
    ↓
OpenCLIP Embedding
    ↓
Chroma Vector Search
    ↓
Grade + Subject Re-ranking
    ↓
Top Results
    ↓
Image Path + Metadata
Project Purpose

This project implements a local, semantic, grade-aware K-12 educational image retrieval system.

It is suitable for:

AI tutoring systems

Classroom tools

Offline education environments

Research prototypes

Foundations for EdTech products