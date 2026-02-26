"""
init_db.py
Creates (or opens) a persistent ChromaDB collection for your K–12 image library.
"""

import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

try:
    from chromadb.utils.data_loaders import ImageLoader
except Exception:
    ImageLoader = None

DB_PATH = "./image_db"
COLLECTION_NAME = "k12_education_images"


def main():
    client = chromadb.PersistentClient(path=DB_PATH)
    embedding_function = OpenCLIPEmbeddingFunction()

    kwargs = dict(name=COLLECTION_NAME, embedding_function=embedding_function)
    if ImageLoader is not None:
        kwargs["data_loader"] = ImageLoader()

    collection = client.get_or_create_collection(**kwargs)

    print("✅ Database Ready")
    print("Path:", DB_PATH)
    print("Collection:", collection.name)
    print("Current image count:", collection.count())


if __name__ == "__main__":
    main()