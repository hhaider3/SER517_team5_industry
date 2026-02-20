import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

client = chromadb.PersistentClient(path="./image_db")
embedding_function = OpenCLIPEmbeddingFunction()
image_loader = ImageLoader()

collection = client.get_or_create_collection(
    name="k12_education_images",
    embedding_function=embedding_function,
    data_loader=image_loader,
)

collection.add(
    ids=["img_1", "img_2", "img_3"],
    uris=[
        "./k12_images/plant_cell.jpg",
        "./k12_images/us_map.jpg",
        "./k12_images/fractions.png"
    ],
    metadatas=[
        {"grade_level": 6, "subject": "Biology", "type": "Diagram"},
        {"grade_level": 4, "subject": "Geography", "type": "Map"},
        {"grade_level": 3, "subject": "Math", "type": "Chart"}
    ],
)

print("Images successfully added!")
