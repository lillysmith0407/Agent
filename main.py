import os
import json
import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from collections import defaultdict

message_counter = defaultdict(int)

app = FastAPI()

@app.get("/health")
async def health():
    return {"ok": True}

# Warm‑up model on startup to prevent slow first response
@app.on_event("startup")
async def warm_model():
    client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "system", "content": "warmup"}],
        temperature=0
    )

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
# CREATIVE QUOTIENT AGENT PROMPT
# -----------------------------
creative_system_prompt = """
You are Ocean Kalra’s Creative Quotient Agent.

Your purpose is to expand ideas, generate creative outputs, unlock imagination,
and help users think divergently, visually, metaphorically, and conceptually.

You ALWAYS adapt your output based on the user's selected:
- creative mode
- creative style

You NEVER explain these settings.
You MUST embody them in tone, structure, rhythm, and imagination.

----------------------------------------
CREATIVE MODES
----------------------------------------
Divergent → many ideas, high variety  
Conceptual → metaphors, analogies, frameworks  
Narrative → stories, characters, scenes  
Visual → imagery, aesthetics, sensory detail  
Synthesis → combine ideas into one cohesive concept  

----------------------------------------
CREATIVE STYLES
----------------------------------------
Soft & Imaginative  
Bold & Experimental  
Minimal & Abstract  
Warm & Playful  
Analytical but Creative  

----------------------------------------
GREETING RULE
----------------------------------------
If the user sends a greeting (“hi”, “hello”), respond naturally and ignore all creative rules.

----------------------------------------
NON‑QUESTION RULE
----------------------------------------
If the user message is conversational, respond naturally.

----------------------------------------
CHAIN‑OF‑THOUGHT SUPPRESSION
----------------------------------------
Never reveal chain-of-thought.
"""

# -----------------------------
# MOREIN V.1 — Productivity + Thinking Partner Prompt
# -----------------------------
morein_system_prompt = """
You are Morein V.1, Ocean Kalra's Productivity and Thinking Partner.

PURPOSE
Help the user think clearly, plan effectively, and act with intention.

CORE BEHAVIOR
- Reduce cognitive load: simplify, don't add.
- Turn vague ideas into concrete, ordered next steps.
- Surface decisions and trade-offs, then help the user choose.
- Prioritize by the user's stated goals, deadlines, and effort vs. impact.
- Keep a calm, strategic, supportive tone.

BOUNDARIES
- Don't bury the user in detail. Give what's needed for the next move.
- Don't do deep academic or research analysis. Offer to hand it to the Research Agent.
- Don't brainstorm expansively or write imaginative content. Offer to hand it to the Creative Agent.
- Stay in your identity as Morein. Don't refer to other AI brands or models.

CHOOSE A MODE BASED ON THE USER'S STATE
- Stuck between options -> Decision-first: name the options, the trade-offs, and your recommendation.
- Unclear or overwhelmed -> Reflection-first: ask one focused question, then structure their answer.
- Ready to move -> Action-first: give the next 1-3 steps, with a time estimate if useful.

RESPONSE FORMAT
- Default to short and structured: a one-line takeaway, then steps or options.
- Use lists only when they help. Use plain sentences for simple answers.
- If key information is missing, make a reasonable assumption and state it, or ask at most one clarifying question. Don't interrogate.
- End with a clear next action when the user is planning or deciding.

CONVERSATIONAL MESSAGES
If the user sends a greeting, small talk, or a non-task message, reply naturally and briefly. Skip all structure.

REASONING
Share conclusions and a short rationale when it helps the user trust the advice. Don't expose raw internal reasoning or step-by-step scratch work.
"""
TASK OUTPUT
If asked to write a document, write it in full Markdown with a # title and ## sections. Task requests override length and brevity preferences.
If asked for a JSON object in a json fence, reply with only that JSON.

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

    # DEMO FIREWALL
    user_ip = request.client.host
    message_counter[user_ip] += 1

    if message_counter[user_ip] > 20:
        return {
            "response": (
                "⚠️ Demo limit reached (5 messages).\n\n"
                "Full version coming soon."
            )
        }

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
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload)}
        ],
        temperature=0.7,
    )

    agent_response = completion.choices[0].message.content

    send_to_formspree(user_message, agent_response)

    return {"response": agent_response}

    
# -----------------------------
# CREATIVE QUOTIENT AGENT ROUTE
# -----------------------------
@app.post("/creative-agent")
async def creative_agent(request: Request):
    data = await request.json()

    user_message = data.get("message", "")
    mode = data.get("mode", None)
    style = data.get("style", None)

    user_payload = {
        "message": user_message,
        "mode": mode,
        "style": style
    }

    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": creative_system_prompt},
            {"role": "user", "content": json.dumps(user_payload)}
        ],
        temperature=0.9,
    )

    agent_response = completion.choices[0].message.content

    send_to_formspree(user_message, agent_response)

    return {"response": agent_response}

# -----------------------------
# MOREIN V.1 — Productivity + Thinking Partner Agent Route
# -----------------------------
import logging
from typing import Literal, Optional

from fastapi import BackgroundTasks, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

logger = logging.getLogger("morein")

MODEL = "openai/gpt-oss-120b"
TEMPERATURE = 0.4  # stable, calm, structured
REQUEST_TIMEOUT = 30  # seconds

MODE_HINTS = {
    "decision": "The user is stuck between options. Lead with options, trade-offs, and a recommendation.",
    "reflection": "The user is unclear or overwhelmed. Ask one focused question, then structure their answer.",
    "action": "The user is ready to move. Lead with the next 1-3 concrete steps.",
}

STYLE_HINTS = {
    "concise": "Keep the response especially brief.",
    "detailed": "Give a bit more depth, but stay structured.",
}


class MoreinRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    mode: Optional[Literal["decision", "reflection", "action"]] = None
    style: Optional[Literal["concise", "detailed"]] = None


def build_system_prompt(mode: Optional[str], style: Optional[str]) -> str:
    """Append optional mode/style guidance to the base prompt."""
    extras = [MODE_HINTS[mode]] if mode else []
    if style:
        extras.append(STYLE_HINTS[style])
    if not extras:
        return morein_system_prompt
    return morein_system_prompt + "\n\nSESSION PREFERENCES\n" + "\n".join(extras)


@app.post("/morein-agent")
async def morein_agent(payload: MoreinRequest, background_tasks: BackgroundTasks):
    user_message = payload.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        # The client call is synchronous, so run it off the event loop.
        completion = await run_in_threadpool(
            client.chat.completions.create,
            model=MODEL,
            messages=[
                {"role": "system", "content": build_system_prompt(payload.mode, payload.style)},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            timeout=REQUEST_TIMEOUT,
        )
        agent_response = (completion.choices[0].message.content or "").strip()
    except Exception:
        logger.exception("Morein model call failed")
        raise HTTPException(status_code=502, detail="Morein is unavailable right now. Please try again.")

    if not agent_response:
        raise HTTPException(status_code=502, detail="Morein returned an empty response.")

    # Log after responding so Formspree latency or failure never affects the user.
    background_tasks.add_task(_safe_send_to_formspree, user_message, agent_response)

    return {"response": agent_response}


def _safe_send_to_formspree(user_message: str, agent_response: str) -> None:
    try:
        send_to_formspree(user_message, agent_response)
    except Exception:
        logger.exception("Formspree logging failed")
