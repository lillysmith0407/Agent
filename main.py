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
# ADVANCED SYSTEM PROMPT
# -----------------------------
system_prompt = """
(… your full system prompt …)
"""

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

    # ⭐ THIS WAS MISSING — THE ROOT CAUSE ⭐
return {"response": completion.choices[0].message.content}
