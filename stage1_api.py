import os
import time
from fastapi import FastAPI, HTTPException
from google import genai
from google.genai.errors import APIError
from pydantic import BaseModel, Field

app = FastAPI(title="Stage 1: AI Job Extractor API")
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Request schema (incoming data from user)
class JobRequest(BaseModel):
    raw_text: str = Field(..., min_length=10, description="Unstructured job description text")

# Response schema (strict JSON guaranteed from LLM)
class JobAnalysis(BaseModel):
    role_title: str
    seniority_level: str = Field(description="Junior, Mid, Senior, Lead, or Unspecified")
    required_skills: list[str]
    is_remote: bool
    salary_range: str = Field(description="Extracted salary or 'Not mentioned'")
    summary: str = Field(description="A 1-sentence summary of the role")

# Few-shot system instruction: teaching the model with an example
SYSTEM_PROMPT = """
You are an expert technical recruiter. Extract accurate structured data from job posts.
Never guess skills that aren't mentioned.

Example Input:
"Hiring a Junior React dev in Addis Ababa. Hybrid work. Must know TypeScript and Tailwind."

Example Output:
{
  "role_title": "React Developer",
  "seniority_level": "Junior",
  "required_skills": ["React", "TypeScript", "Tailwind"],
  "is_remote": false,
  "salary_range": "Not mentioned",
  "summary": "A junior frontend role in Addis Ababa focused on building apps with React and TypeScript."
}
"""

MODELS = ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]

@app.post("/analyze-job", response_model=JobAnalysis)
def analyze_job(request: JobRequest):
    for model_name in MODELS:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=f"Extract details from this job post:\n\n{request.raw_text}",
                    config={
                        "system_instruction": SYSTEM_PROMPT,
                        "response_mime_type": "application/json",
                        "response_schema": JobAnalysis,
                        "temperature": 0.1,
                    },
                )
                return JobAnalysis.model_validate_json(response.text)
            except APIError as e:
                if e.code == 503 and attempt == 0:
                    time.sleep(2)
                else:
                    break
    raise HTTPException(status_code=503, detail="All AI models are currently busy. Try again shortly.")