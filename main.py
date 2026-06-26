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

# -----------------------------
# Request Model (now with features)
# -----------------------------
class Query(BaseModel):
    message: str
    mode: str = "default"          # summary, analysis, comparison, critique, questions
    level: str = "default"         # beginner, intermediate, advanced, graduate, phd

API_KEY = os.getenv("API_KEY")
MODEL = "llama-3.1-8b-instant"

# -----------------------------
# Academic Research Agent Prompt
# -----------------------------
BASE_PROMPT = """
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

Always structure your output using the following sections:
1. Summary
2. Key Insights
3. Analysis
4. Implications
"""

# -----------------------------
# Feature Logic
# -----------------------------
def build_dynamic_instructions(mode: str, level: str):
    instructions = ""

    # Mode instructions
    if mode == "summary":
        instructions += "\nFocus on producing a concise, structured summary."
    elif mode == "analysis":
        instructions += "\nProvide a deeper analytical breakdown of the topic."
    elif mode == "comparison":
        instructions += "\nCompare and contrast the key theories, models, or perspectives."
    elif mode == "critique":
        instructions += "\nProvide a critical evaluation, noting strengths and limitations."
    elif mode == "questions":
        instructions += "\nGenerate 5–10 high‑quality research questions based on the topic."

    # Academic level instructions
    if level in ["beginner", "intermediate", "advanced", "graduate", "phd"]:
        instructions += f"\nAdjust explanation depth to the {level} academic level."

    return instructions


# -----------------------------
# Agent Route
# -----------------------------
@app.post("/agent")
def agent(query: Query):

    dynamic_instructions = build_dynamic_instructions(query.mode, query.level)

    system_prompt = BASE_PROMPT + "\n" + dynamic_instructions

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query.message}
            ]
        }
    )

    return response.json()
