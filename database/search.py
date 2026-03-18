import argparse
import logging
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

# Setup logging to keep the terminal output clean
logging.basicConfig(level=logging.ERROR, format="%(levelname)s | %(message)s")

# Configuration
DB_PATH = "./image_db"
COLLECTION_NAME = "k12_education_images"

def search_database(query_text: str, n_results: int = 5):
    """Queries the ChromaDB collection using OpenCLIP embeddings."""
    
    print(f"\n🔍 Searching for: '{query_text}'...")
    
    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=DB_PATH)
    embedder = OpenCLIPEmbeddingFunction()
    
    try:
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedder)
    except ValueError:
        print(f"❌ Error: Collection '{COLLECTION_NAME}' not found. Make sure you've run the ingestion script first.")
        return

    # Execute the semantic search
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        include=["metadatas", "distances", "uris"]
    )

    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    uris = results.get("uris", [[]])[0]

    if not ids:
        print("No results found.")
        return

    print(f"\n✅ Found top {len(ids)} results:\n" + "="*50)

    # Format and print the results
    for i in range(len(ids)):
        dist = distances[i]
        uri = uris[i]
        meta = metadatas[i] or {}
        
        # Calculate a rough similarity score (lower distance = higher similarity)
        similarity = max(0.0, 1.0 - dist)

        print(f"Result #{i + 1} | Match Score: {similarity:.1%}")
        print(f"📁 Local File : {uri}")
        print(f"🔗 Source Page: {meta.get('source_page', 'Unknown')}")
        
        # Only print subject/topic if they exist in the metadata
        if "subject" in meta:
            print(f"🏷️  Subject    : {meta.get('subject')}")
        if "topic" in meta:
            print(f"📂 Topic      : {meta.get('topic')}")
        
        print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search the local K-12 Image Database.")
    parser.add_argument("query", type=str, help="The text description to search for (e.g., 'right triangle').")
    parser.add_argument("--n", type=int, default=5, help="Number of results to return (default: 5).")
    
    args = parser.parse_args()
    search_database(args.query, args.n)