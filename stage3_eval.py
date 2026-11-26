import os
from google import genai
from google.genai.errors import ServerError
from pydantic import BaseModel, Field
from stage3_rag_chat import ask_handbook, retrieve_context, chat_history

ai_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
JUDGE_MODEL = "gemini-3.8-flash"  # Use a strong model as the impartial judge

# 1. Structured Scorecard returned by the LLM Judge
class RAGEvalScore(BaseModel):
    context_relevance: int = Field(ge=1, le=5, description="1-5: Did the retrieved context contain the answer?")
    faithfulness: int = Field(ge=1, le=5, description="1-5: Is the answer 100% supported by the context with zero hallucinations?")
    answer_relevance: int = Field(ge=1, le=5, description="1-5: Did the answer directly address the user's question?")
    critique: str = Field(description="1-sentence explanation of any deduction")

JUDGE_PROMPT = """
You are an impartial AI Quality Auditor grading a RAG system.
Compare the User Question, Retrieved Context, and Generated Answer.
- Deduct points from Faithfulness if the answer includes ANY fact not explicitly written in the Retrieved Context.
- If the context did NOT have the answer and the bot honestly said it couldn't find it, give Faithfulness = 5 and Answer Relevance = 5.
"""

def evaluate_rag(test_questions: list[str]):
    for q in test_questions:
        chat_history.clear()  # Reset memory so each test case is independent

        # Run your actual Stage 3 RAG pipeline
        retrieved_docs = retrieve_context(q, top_k=2)
        rag_output = ask_handbook(q)

        # Ask the Judge LLM to grade the pipeline
        eval_input = f"""
User Question: {q}

Retrieved Context:
{retrieved_docs}

Generated Answer:
{rag_output.answer}
"""
        judge_resp = ai_client.models.generate_content(
            model=JUDGE_MODEL,
            contents=eval_input,
            config={
                "system_instruction": JUDGE_PROMPT,
                "response_mime_type": "application/json",
                "response_schema": RAGEvalScore,
                "temperature": 0.0,
            },
        )
        score = RAGEvalScore.model_validate_json(judge_resp.text)

        print(f"Q: {q}")
        print(f"A: {rag_output.answer}")
        print(f"Scores -> Context: {score.context_relevance}/5 | Faithfulness: {score.faithfulness}/5 | Answer: {score.answer_relevance}/5")
        print(f"Judge Note: {score.critique}\n" + "-" * 60)

if __name__ == "__main__":
    golden_test_set = [
        "Which days of the week do Addis Ababa engineers work in the Bole office?",
        "How many days do I have to upload travel receipts?",
        "What is the company budget for buying a new laptop?",  # Trap: Not in the PDF!
    ]
    try:
        evaluate_rag(golden_test_set)
    except ServerError as error:
        if error.code != 503:
            raise
        raise SystemExit(
            "Evaluation incomplete: Gemini is temporarily unavailable (503). "
            "No score was produced for this run; try again later."
        ) from None