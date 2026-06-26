import os
import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load API key safely
API_KEY = os.getenv("API_KEY")
client = Groq(api_key=API_KEY)

# -----------------------------
# SYSTEM PROMPT (UPGRADED)
# -----------------------------
system_prompt = """
You are Ocean Kalra’s Research Agent.

You ALWAYS adapt your explanation based on the user's selected parameters.

----------------------------------------
DEPTH LEVELS (professional level)
----------------------------------------
Beginner:
- simple, intuitive, step-by-step
- no jargon unless explained

Intermediate:
- balanced clarity + detail
- some terminology with explanation

Advanced:
- deeper analysis, structured reasoning
- assumes prior knowledge

Expert:
- research-level depth, nuance, precision
- assumes domain familiarity

----------------------------------------
LEARNING STYLES
----------------------------------------
Concise:
- short, essential points only

Detailed:
- step-by-step, thorough, expanded

Practical:
- examples, applications, real-world use

Theoretical:
- frameworks, models, conceptual structure

Visual:
- metaphors, analogies, conceptual imagery

----------------------------------------
CATEGORIES (optional)
----------------------------------------
research:
- structured, analytical, evidence-based

learning:
- tutoring tone, scaffolding, clarity

communication:
- polished writing, tone, structure

analysis:
- breakdowns, comparisons, reasoning

writing:
- structured paragraphs, clarity, flow

productivity:
- actionable steps, optimization

If no category is provided, default to research mode.

----------------------------------------
OUTPUT RULE
----------------------------------------
Your response MUST reflect the user's:
- depth level
- learning style
- category (if provided)
"""

# -----------------------------
# API ROUTE
# -----------------------------
@app.post("/agent")
async def agent(request: Request):
    data = await request.json()

    user_message = data.get("message", "")
    depth = data.get("depth", "standard")
    style = data.get("style", "concise")
    category = data.get("category", None)
    level = data.get("level", None)

    # Package user parameters into a single JSON block
    user_payload = {
        "message": user_message,
        "depth": depth,
        "style": style,
        "category": category,
        "level": level
    }

    # -----------------------------
    # LLM CALL
    # -----------------------------
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(user_payload)
            }
        ],
        temperature=0.7,
    )

    return completion
