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

def find_image_for_student(student_query, grade_filter):
    results = collection.query(
        query_texts=[student_query],
        n_results=1,
        where={"grade_level": grade_filter},
        include=["uris","metadatas","distances","ids"]
    )

    uris = results.get("uris",[[]])
    if uris and uris[0]:
        return uris[0][0]
    return None
    

