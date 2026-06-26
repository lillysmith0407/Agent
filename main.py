from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os   # ← added

app = FastAPI()

class Query(BaseModel):
    message: str

# ← your API key now comes from Render environment variables
API_KEY = os.getenv("API_KEY")

# ← use a real Groq model (Gemini is NOT on Groq)
MODEL = "llama3-8b-8192"

@app.post("/agent")
def agent(query: Query):
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": MODEL,
            "messages": [
                {"role": "user", "content": query.message}
            ]
        }
    )
    return response.json()
