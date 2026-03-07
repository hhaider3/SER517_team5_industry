"""
init_db.py
Creates (or opens) a persistent ChromaDB collection for your K–12 image library.
"""

import logging
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

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

    logger.info("Database initialized successfully")
    logger.info("DB path: %s", DB_PATH)
    logger.info("Collection: %s", collection.name)
    logger.info("Current image count: %d", collection.count())

    print("✅ Database Ready")
    print("Path:", DB_PATH)
    print("Collection:", collection.name)
    print("Current image count:", collection.count())


if __name__ == "__main__":
    main()