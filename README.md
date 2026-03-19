# Fake News Detection using KG + RAG + Local Llama

Phase 1 prototype for claim verification using:

- FastAPI backend
- Streamlit frontend
- SentenceTransformers embeddings
- FAISS retrieval
- spaCy-based NER with fallback
- Basic Knowledge Graph triples
- Local Llama inference through Ollama

## Folder Structure

```text
project_root/
  backend/
  corpus/
    docs/
  dataset/
  evaluation/
  frontend/
  models/
  outputs/
    indexes/
    predictions/
  utils/
  requirements.txt
  README.md
```

## Dataset Placement

Create these files manually inside `dataset/`:

- `dataset/train_clean.csv`
- `dataset/valid_clean.csv`
- `dataset/test_clean.csv`
- `dataset/phase1_balanced_sample.csv`

Required columns:

- `statement`
- `label`

Allowed labels:

- `pants-fire`
- `false`
- `barely-true`
- `half-true`
- `mostly-true`
- `true`

## Setup

1. Create and activate a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install Python dependencies.

```bash
pip install -r requirements.txt
```

3. Download the spaCy English model.

```bash
python -m spacy download en_core_web_sm
```

4. Install Ollama and pull a local Llama model.

```bash
ollama pull llama3.1:8b
```

Optional:

```bash
set OLLAMA_MODEL=llama3.1:8b
```

## Run The Backend

```bash
uvicorn backend.main:app --reload
```

The API will start at `http://127.0.0.1:8000`.

## Run The Frontend

In a second terminal:

```bash
streamlit run frontend/app.py
```

## Verify A Claim

Use the UI or call the API directly:

```bash
curl -X POST http://127.0.0.1:8000/verify ^
  -H "Content-Type: application/json" ^
  -d "{\"claim\":\"Climate change is a hoax invented by scientists.\",\"top_k\":3}"
```

## Evaluation

Run a small subset first:

```bash
python evaluation/evaluate.py --dataset-file phase1_balanced_sample.csv --limit 20
```

This prints Macro-F1 and saves predictions to `outputs/predictions/phase1_eval_predictions.json`.

## Notes

- The FAISS index is built automatically on first backend start from files inside `corpus/docs/`.
- The included corpus is intentionally small and local for offline demo use.
- If Ollama is not running, the system falls back to a heuristic verdict so the pipeline remains testable.
- For best results, keep Ollama running before using `/verify`.
