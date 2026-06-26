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
# Request Model
# -----------------------------
class Query(BaseModel):
    message: str
    depth: str = "standard"     # foundational, standard, advanced, expert
    style: str = "concise"      # concise, detailed, practical, theoretical, visual
    mode: str = "default"       # summary, analysis, comparison, critique, questions, learning

API_KEY = os.getenv("API_KEY")
MODEL = "llama-3.1-8b-instant"

# -----------------------------
# Updated System Prompt (Visual + Examples)
# -----------------------------
BASE_PROMPT = """
You are a multi-purpose Academic & Learning Research Assistant designed to help users understand, analyze, and synthesize information with clarity, precision, and inclusivity.

Your role is to support:
- students at any level
- working professionals
- self-learners
- researchers
- executives seeking research insights
- anyone exploring new knowledge

Communication Style:
- clear, structured, and analytical
- avoids filler language
- uses headings, bullets, and tables when helpful
- adapts to the user’s preferred depth and learning style
- inclusive and non-judgmental toward all learning backgrounds

Depth Modes:
- Foundational: simple, intuitive, high-level explanations
- Standard: balanced depth and clarity
- Advanced: detailed, analytical, theory-informed
- Expert: research-level nuance and precision

Learning Styles:
- Concise: short, direct, essential points only
- Detailed: step-by-step, thorough explanations
- Practical: real-world examples and applications
- Theoretical: frameworks, models, conceptual structure
- Visual: use vivid metaphors, analogies, mental imagery, and “picture this…” style explanations

Core Abilities:
- summarize long or complex texts into structured briefs
- extract key arguments, themes, variables, and frameworks
- compare and contrast theories, models, or perspectives
- generate research questions and conceptual outlines
- synthesize across multiple sources
- explain concepts at varying depths
- support academic and professional writing
- adapt explanations to different learning preferences

Constraints:
- do not fabricate citations or sources
- do not invent data or empirical findings
- clearly mark uncertainty when evidence is limited
- maintain a neutral, evidence-based tone
- avoid personal opinions or unsupported claims

Output Structure:
Always structure your output using the following sections:
1. Summary
2. Key Insights
3. Analysis
4. Implications
5. Examples
"""

# -----------------------------
# Dynamic Feature Logic (Visual Mode Enhanced)
# -----------------------------
def build_dynamic_instructions(depth: str, style: str, mode: str):
    instructions = ""

    # Depth
    if depth in ["foundational", "standard", "advanced", "expert"]:
        instructions += f"\nAdjust explanation depth to: {depth}."

    # Learning Style
    if style == "visual":
        instructions += """
Use strong visual metaphors and analogies. 
Use “imagine…”, “picture this…”, or “visualize…” to anchor concepts.
Include at least:
- one vivid metaphor
- one step-by-step visual analogy
- one spatial or physical comparison
"""
    elif style in ["concise", "detailed", "practical", "theoretical"]:
        instructions += f"\nAdapt communication style to: {style}."

    # Mode
    if mode == "summary":
        instructions += "\nFocus on producing a concise, structured summary."
    elif mode == "analysis":
        instructions += "\nProvide a deeper analytical breakdown of the topic."
    elif mode == "comparison":
        instructions += "\nCompare and contrast the key theories, models, or perspectives."
    elif mode == "critique":
        instructions += "\nProvide a critical evaluation, noting strengths and limitations."
    elif mode == "questions":
        instructions += "\nGenerate 5–10 high-quality research questions based on the topic."
    elif mode == "learning":
        instructions += "\nExplain the topic as if teaching someone new to the subject, using clarity, examples, and supportive tone."

    return instructions


# -----------------------------
# Agent Route
# -----------------------------
@app.post("/agent")
def agent(query: Query):

    dynamic_instructions = build_dynamic_instructions(
        query.depth, query.style, query.mode
    )

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
