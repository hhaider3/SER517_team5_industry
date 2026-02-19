import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

# 1. Connect to the existing local database
client = chromadb.PersistentClient(path="./image_db")
embedding_function = OpenCLIPEmbeddingFunction()
collection = client.get_collection(
    name="k12_education_images", 
    embedding_function=embedding_function
)

def find_image_for_student(student_query, grade_filter):
    # 2. Search the database using the student's natural language text
    results = collection.query(
        query_texts=[student_query],
        n_results=1,
        where={"grade_level": grade_filter} # Apply the strict metadata filter
    )
    
    # 3. Return the result
    if results['uris'][0]:
        return results['uris'][0][0] # Returns the file path to the best matching image
    return None

# --- Simulating the Chatbot ---
student_message = "Show me the parts of a cell"
student_grade = 6

image_path = find_image_for_student(student_message, student_grade)

if image_path:
    print(f"Chatbot replies: 'Here is what I found!' -> Load image from {image_path}")
else:
    print("Chatbot replies: 'I couldn't find an image for that.'")