# PersonaMem 32k → HyperBinder

Evaluation of HyperBinder on the **PersonaMem 32k** benchmark using per-persona namespace ingestion and slot-based retrieval.

---

## Overview

PersonaMem 32k tests long-term memory and persona-consistent retrieval across multiple user identities. Each persona has its own isolated namespace in HyperBinder, and questions are mapped to answers using **dual-slot weighted semantic search** — binding `user_question_or_message` (semantic) to `correct_answer` (exact).

---

## How It Works

### Schema Design

Each row is stored with the following field encodings:

| Field | Encoding |
|---|---|
| `user_question_or_message` | Semantic |
| `correct_answer` | Exact |
| `all_options` | Exact |
| `question_type` | Exact |
| `topic` | Exact |
| `persona_id` | Exact |
| `question_id` | Exact |
| `distance_to_ref_in_blocks` | Exact |
| `distance_to_ref_proportion_in_context` | Exact |

The **semantic slot** on the question drives retrieval at eval time. The **exact slot** on the answer is returned directly when a match is found — no LLM inference required at query time.

### Retrieval Strategy

At evaluation, each question is queried against its persona's namespace using a **pinned slot search** — combining a semantic match on the question text with an exact filter on `question_id`. This prevents collisions when multiple rows share similar or identical question text.

---

## Usage

### 1. Ingest

```bash
python personamem_ingest.py --wipe
```

- Downloads the `32k` split from HuggingFace (`bowen-upenn/PersonaMem`)
- Precomputes embeddings using `all-MiniLM-L6-v2`
- Creates one namespace per persona: `personamem_32k_persona_{id}`
- Saves namespace mapping to `personamem_namespaces.json`

### 2. Evaluate

```bash
python personamem_eval.py
```

- Loads ingested namespaces from `personamem_namespaces.json`
- Runs slot search for each question, pinned by `question_id`
- Scores results via exact match (answers are single letters: A/B/C/D)
- Saves full results to a timestamped JSON file: `personamem_32k_results_YYYYMMDD_HHMMSS.json`

---

## Configuration

Set the following environment variables (or use a `.env` file):

| Variable | Default | Description |
|---|---|---|
| `HB_SERVER_URL` | `http://localhost:8000` | HyperBinder server URL |
| `HB_API_KEY` | *(empty)* | API key for authentication |

---

## Output

Results are broken down by:

- **Per-persona accuracy** — correct / total per persona
- **Question type** — accuracy across different question categories
- **Distance bucket** — `near`, `mid`, or `far` based on the reference position in context

Wrong answers are logged with full debug info: expected vs. retrieved question ID, answer, and confidence score.

---

## Get Access

Request an API key at [questions@semantic-reach.io](mailto:questions@semantic-reach.io)
