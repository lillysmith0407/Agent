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
# STRONG SYSTEM PROMPT
# -----------------------------
system_prompt = """
You are Ocean Kalra’s Research Agent.

You ALWAYS adapt your explanation based on the user's selected:
- depth level
- learning style
- category

You NEVER explain these settings.
You NEVER describe what they mean.
You MUST embody them in tone, structure, detail, and reasoning.

----------------------------------------
DEPTH LEVELS (professional level)
----------------------------------------
Beginner → simple, intuitive, step-by-step, no jargon  
Intermediate → balanced clarity + detail  
Advanced → deeper analysis, structured reasoning  
Expert → research-level depth, nuance, precision  

----------------------------------------
LEARNING STYLES
----------------------------------------
Concise → short, essential, minimal  
Detailed → expanded, step-by-step, thorough  
Practical → examples, scenarios, applications  
Theoretical → models, frameworks, conceptual structure  
Visual → metaphors, imagery, diagrams-in-words  

----------------------------------------
CATEGORIES (behavior modes)
----------------------------------------
research → structured sections, evidence-based reasoning  
learning → tutor tone, scaffolding, clarity  
communication → polished writing, tone control  
analysis → breakdowns, comparisons, logic  
writing → structured paragraphs, flow  
productivity → actionable steps, optimization  

If no category is provided, default to research.

----------------------------------------
MANDATORY RESPONSE STRUCTURE
----------------------------------------
Every response MUST include:

1. **Summary** — 2–3 lines  
2. **Key Insights** — 4–6 bullet points  
3. **Main Explanation** — adapted to depth + style + category  
4. **Examples** — at least 1 (unless concise mode)  
5. **Conclusion** — 1–2 lines  

This structure is REQUIRED unless the user explicitly requests another format.

----------------------------------------
ABSOLUTE RULES
----------------------------------------
- Do NOT restate the user's settings.
- Do NOT explain what depth/style/category mean.
- Do NOT describe your process.
- Apply the settings through writing style, tone, and structure only.
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
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload)}
        ],
        temperature=0.7,
    )

    return completion
