"""
init_db.py
Initializes and verifies the persistent ChromaDB collection for the K-12 image library.
"""

import logging
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
try:
    from chromadb.utils.data_loaders import ImageLoader
except Exception:
    ImageLoader = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

DB_PATH = "./image_db"
COLLECTION_NAME = "k12_education_images"


def main():
    try:
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
    
    except Exception as e:
        logger.error("Failed to initialize database: %s", e)
        raise
    
if __name__ == "__main__":
    main()