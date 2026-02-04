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

Dataset preparation & indexing

1. Convert datasets to JSONL (one JSON object per line) with fields: `input`, `reasoning`, `correction`, `error_type`.

	- CoNLL M2 -> JSONL using:

```bash
python scripts/prepare_datasets.py --m2 path/to/data.m2 --out support.jsonl
```

2. Precompute embeddings and build support index:

```bash
python precompute.py --in support.jsonl --out data/support_index.npz
```

3. Start the API and query `/correct`.

Metrics & tools

- Metrics endpoint: `GET /metrics` returns cache stats and support set size.
- Evaluation: use `gec_service/eval_m2.py` to call external M2 scorer (gold vs system outputs).
# NLP-GEC