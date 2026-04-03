"""
Quick script to seed the ChromaDB image database with test images for verification.
Run from the SER517_team5_industry/database/ directory.
"""
import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
try:
    from chromadb.utils.data_loaders import ImageLoader
except ImportError:
    ImageLoader = None

DB_PATH = "./image_db"
COLLECTION_NAME = "k12_education_images"
FULL_DIR = Path("./k12_images/full")

def main():
    print("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=DB_PATH)
    embedder = OpenCLIPEmbeddingFunction()
    
    kwargs = dict(name=COLLECTION_NAME, embedding_function=embedder)
    if ImageLoader:
        kwargs["data_loader"] = ImageLoader()
    
    collection = client.get_or_create_collection(**kwargs)
    print(f"Collection '{COLLECTION_NAME}' has {collection.count()} images")

    # Find all images in the full directory
    images = list(FULL_DIR.glob("*.jpg")) + list(FULL_DIR.glob("*.png"))
    
    if not images:
        print("❌ No images found in k12_images/full/")
        return
    
    for img_path in images:
        img_id = img_path.stem
        
        # Skip if already exists
        existing = collection.get(ids=[img_id])
        if existing and existing.get("ids"):
            print(f"⏭️  {img_id} already in DB, skipping")
            continue
        
        uri = str(img_path.resolve())
        metadata = {
            "subject": "Math",
            "topic": "Fractions",
            "description": "Understanding fractions on a number line with fraction bars showing 1/2, 1/3, and 1/4 comparisons",
            "grade_min": 3,
            "grade_max": 6,
            "review_status": "approved",
        }
        
        print(f"📥 Adding {img_id} to collection...")
        collection.add(
            ids=[img_id],
            uris=[uri],
            metadatas=[metadata],
        )
        print(f"✅ Added {img_id}")

    print(f"\nDone! Collection now has {collection.count()} images")

if __name__ == "__main__":
    main()