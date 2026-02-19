import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

# 1. Initialize ChromaDB (This creates a local folder called 'image_db' to store your vectors)
client = chromadb.PersistentClient(path="./image_db")

# 2. Load the CLIP model (This understands both images and text)
embedding_function = OpenCLIPEmbeddingFunction()

# 3. Create a collection (like a table in SQL)
collection = client.get_or_create_collection(
    name="k12_education_images",
    embedding_function=embedding_function
)

# 4. Add your test images and their K-12 metadata
collection.add(
    ids=["img_1", "img_2", "img_3"],
    uris=[ # The local paths to your images
        "./k12_images/plant_cell.jpg", 
        "./k12_images/us_map.jpg",
        "./k12_images/fractions.png"
    ],
    metadatas=[
        {"grade_level": 6, "subject": "Biology", "type": "Diagram"},
        {"grade_level": 4, "subject": "Geography", "type": "Map"},
        {"grade_level": 3, "subject": "Math", "type": "Chart"}
    ]
)

print("Images successfully added to the vector database!")
