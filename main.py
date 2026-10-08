import os
import json
import time
import logging
from collections import defaultdict
from typing import Literal, Optional

import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel, Field

logger = logging.getLogger("agents")

# -----------------------------
# SETUP
# -----------------------------
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY environment variable is not set.")
client = Groq(api_key=API_KEY)

MODEL = "openai/gpt-oss-120b"
REQUEST_TIMEOUT = 30      # seconds
MAX_TOKENS = 4000         # reasoning tokens count toward this, so leave headroom
FORMSPREE_URL = "https://formspree.io/f/mvzjrajk"

# Set ALLOWED_ORIGINS on Render, e.g. "https://oceankalra.com,https://www.oceankalra.com"
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"ok": True}


# -----------------------------
# HELPERS
# -----------------------------
def client_ip(request: Request) -> str:
    """Behind Render's proxy, request.client.host is the proxy, not the visitor."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


_hits = defaultdict(list)


def rate_limit(request: Request, bucket: str, limit: int = 12, window: int = 60) -> None:
    key = (bucket, client_ip(request))
    now = time.time()
    _hits[key] = [t for t in _hits[key] if now - t < window]
    if len(_hits[key]) >= limit:
        raise HTTPException(status_code=429, detail="Too many requests. Wait a moment and try again.")
    _hits[key].append(now)


def send_to_formspree(user_message: str, agent_response: str) -> None:
    try:
        requests.post(
            FORMSPREE_URL,
            json={"user_message": user_message, "agent_response": agent_response},
            headers={"Content-Type": "application/json"},
            timeout=5,
        )
    except Exception:
        logger.exception("Formspree logging failed")


async def ask_model(system: str, user: str, temperature: float) -> str:
    try:
        # The Groq client is synchronous, so run it off the event loop.
        completion = await run_in_threadpool(
            client.chat.completions.create,
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=MAX_TOKENS,
            timeout=REQUEST_TIMEOUT,
        )
        text = (completion.choices[0].message.content or "").strip()
    except Exception:
        logger.exception("Model call failed")
        raise HTTPException(status_code=502, detail="The agent is unavailable right now. Please try again.")
    if not text:
        raise HTTPException(status_code=502, detail="The agent returned an empty response.")
    return text


# -----------------------------
# RESEARCH AGENT PROMPT
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

TASK OUTPUT
If asked to write a document, write it in full Markdown with a # title and ## sections. Task requests override length and brevity preferences.
If asked for a JSON object in a json fence, reply with only that JSON.

REASONING
Share conclusions and a short rationale when it helps the user trust the advice. Don't expose raw internal reasoning or step-by-step scratch work.
"""

MODE_HINTS = {
    "decision": "The user is stuck between options. Lead with options, trade-offs, and a recommendation.",
    "reflection": "The user is unclear or overwhelmed. Ask one focused question, then structure their answer.",
    "action": "The user is ready to move. Lead with the next 1-3 concrete steps.",
}

STYLE_HINTS = {
    "concise": "Keep the response especially brief.",
    "detailed": "Give a bit more depth, but stay structured.",
}


def build_morein_prompt(mode: Optional[str], style: Optional[str]) -> str:
    extras = [MODE_HINTS[mode]] if mode else []
    if style:
        extras.append(STYLE_HINTS[style])
    if not extras:
        return morein_system_prompt
    return morein_system_prompt + "\n\nSESSION PREFERENCES\n" + "\n".join(extras)


# -----------------------------
# REQUEST MODELS
# -----------------------------
class ResearchRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=6000)
    depth: Optional[str] = None
    style: Optional[str] = None
    category: Optional[str] = None
    preset: Optional[str] = None


class CreativeRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=6000)
    mode: Optional[str] = None
    style: Optional[str] = None


class MoreinRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=6000)
    mode: Optional[Literal["decision", "reflection", "action"]] = None
    style: Optional[Literal["concise", "detailed"]] = None


# -----------------------------
# RESEARCH AGENT ROUTE (demo)
# -----------------------------
DEMO_LIMIT = 20
message_counter = defaultdict(int)  # resets whenever the server restarts


@app.post("/agent")
async def agent(payload: ResearchRequest, request: Request, background_tasks: BackgroundTasks):
    rate_limit(request, "agent")

    ip = client_ip(request)
    message_counter[ip] += 1
    if message_counter[ip] > DEMO_LIMIT:
        return {
            "response": (
                f"⚠️ Demo limit reached ({DEMO_LIMIT} messages).\n\n"
                "Full version coming soon."
            )
        }

    user_message = payload.message.strip()
    user_payload = {
        "message": user_message,
        "depth": payload.depth,
        "style": payload.style,
        "category": payload.category,
        "preset": payload.preset,
    }
    agent_response = await ask_model(system_prompt, json.dumps(user_payload), 0.7)

    background_tasks.add_task(send_to_formspree, user_message, agent_response)
    return {"response": agent_response}


# -----------------------------
# CREATIVE QUOTIENT AGENT ROUTE
# -----------------------------
@app.post("/creative-agent")
async def creative_agent(payload: CreativeRequest, request: Request, background_tasks: BackgroundTasks):
    rate_limit(request, "creative")

    user_message = payload.message.strip()
    user_payload = {"message": user_message, "mode": payload.mode, "style": payload.style}
    agent_response = await ask_model(creative_system_prompt, json.dumps(user_payload), 0.9)

    background_tasks.add_task(send_to_formspree, user_message, agent_response)
    return {"response": agent_response}


# -----------------------------
# MOREIN V.1 — Productivity + Thinking Partner Route
# -----------------------------
@app.post("/morein-agent")
async def morein_agent(payload: MoreinRequest, request: Request, background_tasks: BackgroundTasks):
    rate_limit(request, "morein")

    user_message = payload.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    agent_response = await ask_model(build_morein_prompt(payload.mode, payload.style), user_message, 0.4)

    background_tasks.add_task(send_to_formspree, user_message, agent_response)
    return {"response": agent_response}
