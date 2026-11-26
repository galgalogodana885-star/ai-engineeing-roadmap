import os
import chromadb
from google import genai
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Initialize Gemini & a fresh ChromaDB collection
ai_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
chroma_client = chromadb.PersistentClient(path="./my_vector_db")
collection = chroma_client.get_or_create_collection(name="pdf_handbook")

EMBED_MODEL = "gemini-embedding-001"

def get_embedding(text: str) -> list[float]:
    result = ai_client.models.embed_content(model=EMBED_MODEL, contents=text)
    return result.embeddings[0].values

# 2. Extract text page-by-page from the real PDF
pdf_path = "addis_tech_handbook.pdf"
reader = PdfReader(pdf_path)
print(f"Loaded '{pdf_path}' ({len(reader.pages)} pages).\n")

# 3. Configure smart recursive splitter (splits by paragraph -> line -> sentence)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=40,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# 4. Split each page cleanly and store with page-number metadata
total_chunks = 0
for page_num, page in enumerate(reader.pages, start=1):
    page_text = page.extract_text() or ""
    chunks = splitter.split_text(page_text)

    for chunk_idx, chunk in enumerate(chunks):
        vector = get_embedding(chunk)
        doc_id = f"{pdf_path}_p{page_num}_c{chunk_idx}"

        collection.upsert(
            ids=[doc_id],
            embeddings=[vector],
            documents=[chunk],
            metadatas=[{"source": pdf_path, "page": page_num, "chunk": chunk_idx}],
        )
        total_chunks += 1
        print(f"Indexed Page {page_num}, Chunk {chunk_idx}: {chunk[:60]}...")

print(f"\nDone! Stored {total_chunks} clean chunks in ChromaDB.\n")

# 5. Test semantic search across pages
query = "What framework and tools do we use to deploy backend services?"
print(f"Query: '{query}'")
results = collection.query(
    query_embeddings=[get_embedding(query)],
    n_results=2,
)

print("\n--- Top Retrieved Chunks ---")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"[File: {meta['source']} | Page {meta['page']}]")
    print(f"{doc}\n")