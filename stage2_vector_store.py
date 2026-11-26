import os
import chromadb
from google import genai

# 1. Initialize Gemini client & local ChromaDB
ai_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
chroma_client = chromadb.PersistentClient(path="./my_vector_db")
collection = chroma_client.get_or_create_collection(name="company_handbook")

# Dynamically find the active embedding model on your API key
embed_models = [
    m.name.replace("models/", "")
    for m in ai_client.models.list()
    if "embedContent" in (m.supported_actions or [])
]
print(f"Available embedding models: {embed_models}")
EMBED_MODEL = embed_models[0]  # Pick the first active embedding model
print(f"Using embedding model: {EMBED_MODEL}\n")

# 2. Sample raw document
raw_document = """
Remote Work Policy: Employees are allowed to work from home up to 3 days per week. 
Core collaboration hours are 10:00 AM to 3:00 PM EAT. Managers must approve full-time remote requests.

Expense Reimbursement: Meals during business travel are covered up to $40 per day. 
All receipts must be uploaded to the finance portal within 14 days of returning from a trip.

Tech Stack Standards: All new backend microservices must be built using Python 3.11+ and FastAPI. 
PostgreSQL is our primary relational database, and Docker is required for deployment.
"""

# 3. Splitting (Chunking)
def chunk_text(text: str, chunk_size: int = 220, overlap: int = 40) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return chunks

chunks = chunk_text(raw_document)
print(f"Split document into {len(chunks)} chunks.\n")

# 4. Helper function to turn text into a vector
def get_embedding(text: str) -> list[float]:
    result = ai_client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
    )
    return result.embeddings[0].values

# 5. Embed and store every chunk in ChromaDB
for idx, chunk in enumerate(chunks):
    vector = get_embedding(chunk)
    collection.upsert(
        ids=[f"doc_chunk_{idx}"],
        embeddings=[vector],
        documents=[chunk],
        metadatas=[{"source": "employee_handbook.pdf", "chunk_index": idx}],
    )
print("Successfully stored all vectors in ChromaDB!\n")

# 6. Semantic Search
user_query = "Can I do my job from my house?"
print(f"Searching for: '{user_query}'")

query_vector = get_embedding(user_query)
results = collection.query(
    query_embeddings=[query_vector],
    n_results=2,
)

print("\n--- Top Matching Chunks ---")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"[Source: {meta['source']} | Chunk {meta['chunk_index']}]")
    print(f"-> {doc}\n")