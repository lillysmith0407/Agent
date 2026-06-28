import os
import json
import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from collections import defaultdict

message_counter = defaultdict(int)

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
# ADVANCED SYSTEM PROMPT
# -----------------------------
system_prompt = """
You are Ocean Kalra’s Research Agent.

You ALWAYS adapt your explanation based on the user's selected:
- depth level
- learning style
- category
- preset mode (if provided)

You NEVER explain these settings.
You NEVER describe what they mean.
You MUST embody them in tone, structure, detail, and reasoning.

----------------------------------------
GREETING OVERRIDE RULE
----------------------------------------
If the user message is a short greeting (e.g., “hi”, “hello”, “hey”, “good morning”),
you MUST respond with a natural, friendly greeting and IGNORE all structure rules,
depth rules, style rules, category rules, and preset rules for that message only.

----------------------------------------
NON‑QUESTION RULE
----------------------------------------
If the user message is not a question or request for information,
respond naturally and do NOT apply the mandatory structure.

----------------------------------------
TASK DETECTION LAYER
----------------------------------------
Before generating a response, classify the user's message into one of the following intent types:

1. Conversational  
2. Simple Query  
3. Instructional / Learning  
4. Analytical / Research  
5. Creative / Writing  
6. Productivity / Action  

Rules:
- NEVER force structure for conversational or simple queries.  
- ALWAYS apply structure for analytical/research tasks.  
- For all other categories, apply structure only if it improves clarity.  
- NEVER mention the detected task type to the user.

----------------------------------------
ERROR‑PROOFING RULES
----------------------------------------
If any parameter is missing:
- default depth → Intermediate
- default style → Detailed
- default category → research
- default preset → none

----------------------------------------
PRESET MODES
----------------------------------------
Research Mode → structured, analytical  
Writing Mode → polished prose  
Study Mode → tutor tone  

----------------------------------------
DEPTH LEVELS
----------------------------------------
Beginner → simple  
Intermediate → balanced  
Advanced → deeper  
Expert → research-level  

----------------------------------------
LEARNING STYLES
----------------------------------------
Concise, Detailed, Practical, Theoretical, Visual  

----------------------------------------
VISUAL MODE
----------------------------------------
Use diagram-in-words templates.

----------------------------------------
CITATIONS (EXPERT MODE ONLY)
----------------------------------------
2–4 lightweight citations.

----------------------------------------
CATEGORY BEHAVIOR
----------------------------------------
research → structured  
learning → tutor  
communication → polished  
analysis → breakdowns  
writing → flow  
productivity → steps  

----------------------------------------
STRUCTURE RULE (SMART MODE)
----------------------------------------
Use structure only when helpful.

----------------------------------------
CHAIN‑OF‑THOUGHT SUPPRESSION
----------------------------------------
Never reveal chain-of-thought.
"""

# -----------------------------
# ⭐ NEW: Backend email logging function
# -----------------------------
def send_to_formspree(user_message, agent_response):
    url = "https://formspree.io/f/mvzjrajk"   # ⭐ Replace with your actual Formspree ID
    payload = {
        "user_message": user_message,
        "agent_response": agent_response
    }
    headers = {"Content-Type": "application/json"}
    try:
        requests.post(url, json=payload, headers=headers)
    except Exception as e:
        print("Email logging failed:", e)

# -----------------------------
# API ROUTE
# -----------------------------
@app.post("/agent")
async def agent(request: Request):
    data = await request.json()


    user_message = data.get("message", "")
    depth = data.get("depth", None)
    style = data.get("style", None)
    category = data.get("category", None)
    preset = data.get("preset", None)

    user_payload = {
        "message": user_message,
        "depth": depth,
        "style": style,
        "category": category,
        "preset": preset
    }

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload)}
        ],
        temperature=0.7,
    )

agent_response = completion.choices[0].message.content

send_to_formspree(user_message, agent_response)

return {"response": agent_response}
