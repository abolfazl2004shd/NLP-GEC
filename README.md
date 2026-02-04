# Context-Aware GEC (RAG + CoT + Semantic Caching)

This repo contains a reference implementation of a Context-Aware Grammatical Error Correction (GEC) system using Retrieval-Augmented Generation (RAG), Chain-of-Thought (CoT), and semantic caching.

Quickstart

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set `OPENAI_API_KEY` in env or `.env` if you plan to use OpenAI.

3. Run the API:

```bash
uvicorn gec_service.api:app --reload --host 0.0.0.0 --port 8000
```

API: POST /correct with JSON {"input": "sentence to correct"}

See `gec_service` for implementation details.
# NLP-GEC