import os
import time
from google import genai
from google.genai.errors import APIError
from pydantic import BaseModel, Field

# 1. Define the exact JSON structure you want back
class JobAnalysis(BaseModel):
    role_title: str
    seniority_level: str = Field(description="Junior, Mid, Senior, or Lead")
    required_skills: list[str]
    is_remote: bool
    summary: str = Field(description="A 1-sentence summary of the role")

# 2. Initialize the client
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Dynamically find available text generation models for your API key
available_models = [
    m.name.replace("models/", "")
    for m in client.models.list()
    if "generateContent" in (m.supported_actions or []) and "gemini" in m.name
]
print(f"Available models on your key: {available_models}\n")

raw_job_post = """
We are looking for a backend wizard to join our distributed team! You'll build 
high-speed APIs using Python, FastAPI, and PostgreSQL. Experience with Docker 
and Redis is a huge plus. Must have 3+ years of production experience. Work from anywhere!
"""

# 3. Try available models with retries for 503 traffic spikes
response = None
for model_name in available_models:
    for attempt in range(2):
        try:
            print(f"Trying {model_name} (attempt {attempt + 1})...")
            response = client.models.generate_content(
                model=model_name,
                contents=f"Extract the structured details from this job post:\n\n{raw_job_post}",
                config={
                    "system_instruction": "You are an expert technical recruiter. Extract accurate data only.",
                    "response_mime_type": "application/json",
                    "response_schema": JobAnalysis,
                    "temperature": 0.1,
                },
            )
            break
        except APIError as e:
            print(f"  -> {model_name} failed ({e.code}): {e.message[:80]}...")
            if e.code == 503 and attempt == 0:
                time.sleep(2)
            else:
                break
    if response:
        break

if not response:
    raise RuntimeError("No available models succeeded right now. Please try again shortly.")

# 4. Parse and print the validated data
job_data = JobAnalysis.model_validate_json(response.text)
print("\n--- Extracted Job Details ---")
print(f"Role: {job_data.role_title} ({job_data.seniority_level})")
print(f"Skills: {', '.join(job_data.required_skills)}")
print(f"Remote: {job_data.is_remote}")
print(f"Summary: {job_data.summary}")