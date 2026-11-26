import os
import chromadb
from google import genai
from google.genai.errors import ServerError
from pydantic import BaseModel, Field

# 1. Connect to Gemini & your existing Stage 2 ChromaDB collection
ai_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
chroma_client = chromadb.PersistentClient(path="./my_vector_db")
collection = chroma_client.get_collection(name="pdf_handbook")

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.1-flash-lite"

# 2. Structured RAG Output: Forces the AI to cite exact page numbers!
class RAGResponse(BaseModel):
    answer: str = Field(description="Answer grounded strictly in the provided context")
    citations: list[str] = Field(description="List of sources/pages used, e.g. ['addis_tech_handbook.pdf - Page 2']")
    found_in_docs: bool = Field(description="True if the context contained the answer, False otherwise")

SYSTEM_PROMPT = """
You are the Addis Tech HR & Engineering Assistant.
Answer the user's question using ONLY the provided Context Chunks below.
- If the answer is not in the context, set found_in_docs to False and say: "I couldn't find that in the company handbook."
- Never guess or use outside knowledge.
- Always cite the exact file name and page number from the context.
"""

# 3. Conversation Memory (stores recent turns)
chat_history: list[str] = []

def retrieve_context(query: str, top_k: int = 2) -> str:
    query_vec = ai_client.models.embed_content(
        model=EMBED_MODEL,
        contents=query,
    ).embeddings[0].values
    results = collection.query(query_embeddings=[query_vec], n_results=top_k)

    formatted_chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        formatted_chunks.append(f"[Source: {meta['source']}, Page {meta['page']}]\n{doc}")
    return "\n\n---\n\n".join(formatted_chunks)

def ask_handbook(user_question: str) -> RAGResponse:
    # Step A: Rewrite/combine with recent history so follow-up questions retrieve well
    search_query = f"{' '.join(chat_history[-2:])} {user_question}".strip()

    # Step B: Retrieve relevant chunks from ChromaDB
    context = retrieve_context(search_query, top_k=2)

    # Step C: Augment prompt with history + retrieved context, then Generate
    full_prompt = f"""
Recent Conversation History:
{chr(10).join(chat_history[-4:]) or "None"}

Context Chunks:
{context}

User Question: {user_question}
"""
    response = ai_client.models.generate_content(
        model=CHAT_MODEL,
        contents=full_prompt,
        config={
            "system_instruction": SYSTEM_PROMPT,
            "response_mime_type": "application/json",
            "response_schema": RAGResponse,
            "temperature": 0.1,
        },
    )
    parsed = RAGResponse.model_validate_json(response.text)

    # Save turn to memory
    chat_history.append(f"User: {user_question}")
    chat_history.append(f"Assistant: {parsed.answer}")
    return parsed

# 4. Interactive terminal chat loop (tests grounding, follow-up memory, and anti-hallucination!)
if __name__ == "__main__":
    print("Addis Tech Handbook RAG Bot ready! Type 'exit' to quit.\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        try:
            result = ask_handbook(q)
        except ServerError as error:
            if error.code != 503:
                raise
            print(
                "\nGemini is temporarily unavailable (503). "
                "Your question wasn't processed; please try again in a moment.\n"
            )
            continue
        print(f"\nBot: {result.answer}")
        print(f"Found in docs: {result.found_in_docs} | Citations: {result.citations}\n")