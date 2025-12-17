from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import json
import uuid
import os
from datetime import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()
app = FastAPI(title="LLM Lab Backend")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://llmlab-frontend.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_FILE = "experiments.json"

if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w") as f:
        json.dump([], f, indent=2)
        
        
class ParamSet(BaseModel):
    temperature: float
    top_p: float

class ExperimentRequest(BaseModel):
    prompt: str
    params: List[ParamSet]


def compute_metrics(prompt: str, response: str):
    #length
    words = response.split()
    total_words = len(words)
    unique_words = len(set(words))

    #diversity
    diversity = unique_words / total_words if total_words > 0 else 0

    #Completeness
    expected_min_tokens = 60
    completeness = min(total_words / expected_min_tokens, 1)

    #Relevance
    import hashlib
    def fake_embed(text):
        h = hashlib.sha256(text.encode()).hexdigest()
        # convert hex → list of numbers
        return [int(h[i:i+2], 16) for i in range(0, 64, 2)]

    import numpy as np
    a = np.array(fake_embed(prompt))
    b = np.array(fake_embed(response))
    relevance = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    #Coherence
    sentences = response.split(".")
    if len(sentences) > 1:
        s1 = len(sentences[0].split())
        s2 = len(sentences[1].split())
        coherence = min(s1, s2) / max(s1, s2)
    else:
        coherence = 1

    #Avg score
    avg_score = (relevance + completeness + diversity + coherence) / 4

    return {
        "relevance": relevance,
        "completeness": completeness,
        "diversity": diversity,
        "coherence": coherence,
        "avg_score": avg_score
    }



# -----------------------------
# API: RUN EXPERIMENT
# -----------------------------
@app.post("/run")
async def run_experiment(req: ExperimentRequest):
    results = []

    for p in req.params:
        print(p)
        try:
            response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=req.prompt,
            config=types.GenerateContentConfig(
            temperature=p.temperature,
            top_p=p.top_p)
            )
            text = response.text or ""
        except Exception as e:
            text = f"[Error generating response: {str(e)}]"

        results.append({
            "temperature": p.temperature,
            "top_p": p.top_p,
            "response": text,
            "metrics": compute_metrics(req.prompt, text)
        })

    experiment = {
        "id": str(uuid.uuid4()),
        "prompt": req.prompt,
        "timestamp": datetime.utcnow().isoformat(),
        "results": results
    }

    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    data.append(experiment)

    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

    return {"success": True, "experiment": experiment}


#List current experiment
@app.get("/experiments")
async def get_experiments():
    with open(DATA_FILE, "r") as f:
        return json.load(f)


#Export experiment
@app.get("/experiment/{exp_id}")
async def get_experiment(exp_id: str):
    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    for exp in data:
        if exp["id"] == exp_id:
            return exp

    return {"error": "Not found"}


@app.get("/export/{exp_id}")
async def export_experiment(exp_id: str):
    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    for exp in data:
        if exp["id"] == exp_id:
            filename = f"experiment_{exp_id}.json"
            return exp

    return {"error": "Not found"}


@app.get("/")
async def home():
    return {"msg": "LLM Lab Backend Running!"}
