import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader


client = chromadb.PersistentClient(path="./image_db")

embedding_function = OpenCLIPEmbeddingFunction()
image_loader = ImageLoader()

collection = client.get_collection(
    name="k12_education_images",
    embedding_function=embedding_function,
    data_loader=image_loader,
)

def update_metadata(image_id, new_metadata):
    collection.update(
        ids=[image_id],
        metadatas=[new_metadata]
    )
    print(f"Metadata updated for {image_id}")


def update_image(image_id, new_uri):
    collection.update(
        ids=[image_id],
        uris=[new_uri]
    )
    print(f"Image URI updated for {image_id}")


def delete_image(image_id):
    collection.delete(ids=[image_id])
    print(f"{image_id} deleted successfully")


def add_new_image(image_id, image_uri, metadata):
    collection.add(
        ids=[image_id],
        uris=[image_uri],
        metadatas=[metadata]
    )
    print(f"{image_id} added successfully")



