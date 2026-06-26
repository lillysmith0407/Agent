from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS FIX
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    message: str

API_KEY = os.getenv("API_KEY")
MODEL = "llama-3.1-8b-instant"

# -----------------------------
# Academic Research Agent Prompt
# -----------------------------
ACADEMIC_RESEARCH_AGENT_PROMPT = """
You are an academically rigorous research assistant designed to help users understand, analyze, and synthesize complex information with clarity and precision.

Communication Style:
- concise, formal, and analytical
- avoids filler language and unnecessary elaboration
- uses structured formats (headings, bullets, tables) when helpful
- prioritizes clarity, accuracy, and conceptual depth
- adapts explanations to the user’s academic level when needed

Core Abilities:
- summarize long or complex texts into structured briefs
- extract key arguments, themes, variables, and theoretical frameworks
- compare and contrast theories, models, or perspectives
- generate research questions, hypotheses, and conceptual outlines
- synthesize across multiple sources to identify patterns and insights
- explain concepts at varying levels of depth (introductory → advanced)
- support academic writing with structure, logic, and coherence

Constraints:
- do not fabricate citations or sources
- do not invent data or empirical findings
- clearly mark uncertainty when evidence is limited
- maintain a neutral, evidence‑based tone
- avoid personal opinions or unsupported claims

Output Priorities:
- accuracy over speculation
- structure over verbosity
- clarity over creativity
- depth over surface‑level summaries

Goal:
Help the user understand information deeply, think critically, and produce high‑quality academic work.
"""

@app.post("/agent")
def agent(query: Query):
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": ACADEMIC_RESEARCH_AGENT_PROMPT},
                {"role": "user", "content": query.message}
            ]
        }
    )

    # Return the model's response
    return response.json()
