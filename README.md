# README — Backend (FastAPI + Gemini)

**Live URL:** [https://llm-backend-xez7.onrender.com](https://llm-backend-xez7.onrender.com)
**Repo:** [https://github.com/devsvarun/llm-backend](https://github.com/devsvarun/llm-backend)

## Overview

FastAPI backend that:

- Accepts generation experiments (`/run`)
- Calls Gemini (via `google-generativeai` client)
- Computes custom quality metrics for each response
- Persists experiments to `experiments.json`
- Exposes listing and single-experiment endpoints

Tech stack:

- Python 3.11+
- FastAPI
- google-generativeai
- Uvicorn (local)
- Deployed on Render

## Endpoints

- `POST /run` — run experiment
  Request body:

  ```json
  {
    "prompt": "string",
    "params": [{"temperature": 0.7, "top_p": 0.9}, ...]
  }
  ```

  Response:

  ```json
  {
    "experiment": {
      "id": "...",
      "prompt": "...",
      "timestamp": "...",
      "results":[
        {"temperature":0.7,"top_p":0.9,"response":"...","metrics":{...}}
      ]
    }
  }
  ```

- `GET /experiments` — list saved experiments

- `GET /experiment/{id}` — get single experiment

## Setup (Local)

1. Clone:

```bash
git clone https://github.com/devsvarun/llm-backend.git
cd llm-backend
```

2. Create virtual env and install:

```bash
python -m venv .venv
source .venv/bin/activate    # windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Add environment variable:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

4. Run:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Persistence

- `experiments.json` is used for simple persistence (append-only). Good for assignment. For production, migrate to a DB (SQLite/Postgres).

## Metrics implemented

- `coherence` — sentence-length/structure heuristic
- `length_score` — ideal length between 80–180 words
- `relevance` — embedding similarity (or placeholder if embedding unavailable)
- `completeness` — response length vs expected minimum
- `diversity` — unique-word ratio
- `avg_score` — averaged metric for ranking

(See quality-metrics doc for formulas.)

## CORS

CORS is enabled for development; `allow_origins` set to `*` for assignment. Restrict in production.

## Deployment notes

- Render: add `GEMINI_API_KEY` to Render environment settings.
- Ensure `experiments.json` is writable or use an external storage solution.

## Troubleshooting

- `AttributeError: generate_content`: ensure you're calling `client.models.generate_content(...)` (per SDK quickstart) and using matching client version.
- 405 on OPTIONS: ensure FastAPI includes `CORSMiddleware` with `allow_methods=["*"]`.

## Contact

Varun Sharma
