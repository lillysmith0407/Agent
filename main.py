from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()

class Query(BaseModel):
    message: str

API_KEY = "YOUR_GROQ_API_KEY"
MODEL = "gemini-1.5-flash"

@app.post("/agent")
def agent(query: Query):
    response = requests.post(
        "https://api.gemini.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": MODEL, "messages": [{"role": "user", "content": query.message}]}
    )
    return response.json()
