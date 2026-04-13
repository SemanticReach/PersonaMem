"""
personamem_ingest.py
Ingest PersonaMem 32k benchmark into HyperBinder.

Strategy: bind user_question_or_message (semantic) -> correct_answer (exact)
in a per-persona namespace so slot search returns the correct answer directly.
"""

import json
import io
import sys
import os
import time

import pandas as pd
import requests
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
SERVER_URL = os.getenv("HB_SERVER_URL", "http://localhost:8000")
API_KEY    = os.getenv("HB_API_KEY", "")
DB_NAME    = "fractal_db"
SPLIT      = "32k"
NS_PREFIX  = "personamem_32k_persona_"
EMBED_DIM  = 384
VECTOR_COL = "precomputed_vectors"

# ── QA Schema ─────────────────────────────────────────────────────────────────
# question slot = semantic  →  drives the slot-search at eval time
# answer  slot = exact      →  returned directly when question matches
PERSONAMEM_QA_SCHEMA = json.dumps({
    "molecule": "Row",
    "primary_key": "question_id",
    "fields": {
        "user_question_or_message": {"encoding": "semantic"},
        "correct_answer":           {"encoding": "exact"},
        "all_options":              {"encoding": "exact"},
        "question_type":            {"encoding": "exact"},
        "topic":                    {"encoding": "exact"},
        "persona_id":               {"encoding": "exact"},
        "question_id":              {"encoding": "exact"},
        "distance_to_ref_in_blocks":          {"encoding": "exact"},
        "distance_to_ref_proportion_in_context": {"encoding": "exact"},
    },
    "field_order": [
        "question_id", "persona_id", "question_type", "topic",
        "user_question_or_message", "correct_answer", "all_options",
        "distance_to_ref_in_blocks", "distance_to_ref_proportion_in_context",
        VECTOR_COL,
    ],
})

ROW_FIELDS = [
    "question_id", "persona_id", "question_type", "topic",
    "user_question_or_message", "correct_answer", "all_options",
    "distance_to_ref_in_blocks", "distance_to_ref_proportion_in_context",
    VECTOR_COL,
]


# ── Namespace tracking ────────────────────────────────────────────────────────
NS_FILE = "personamem_namespaces.json"


def save_namespace(persona_id: int, namespace: str):
    data = {}
    if os.path.exists(NS_FILE):
        with open(NS_FILE, "r") as f:
            data = json.load(f)
    data[str(persona_id)] = namespace
    with open(NS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def wipe_namespaces():
    print("\n" + "=" * 60)
    print("WIPING existing PersonaMem namespaces")
    print("=" * 60)

    try:
        resp = requests.get(
            f"{SERVER_URL}/db/{DB_NAME}/namespaces",
            headers={"X-API-Key": API_KEY} if API_KEY else {},
        )
        if resp.ok:
            all_ns = resp.json().get("namespaces", [])
            target = [ns for ns in all_ns if ns.startswith(NS_PREFIX)]
            print(f"Found {len(target)} PersonaMem namespaces to delete")
            for ns in target:
                r = requests.delete(
                    f"{SERVER_URL}/db/{DB_NAME}/namespace/{ns}",
                    headers={"X-API-Key": API_KEY} if API_KEY else {},
                    timeout=30,
                )
                print(f"  {'✓' if r.ok else '✗'} {ns}")
        else:
            print(f"  Could not list namespaces: {resp.status_code}")
    except Exception as e:
        print(f"  Error: {e}")

    if os.path.exists(NS_FILE):
        os.remove(NS_FILE)
        print(f"  Deleted {NS_FILE}")


# ── Embeddings ────────────────────────────────────────────────────────────────
def precompute_embeddings(questions: list[str]) -> dict:
    from sentence_transformers import SentenceTransformer

    print("\nLoading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    unique_qs = list(set(questions))
    print(f"Embedding {len(unique_qs)} unique questions...")

    cache = {}
    batch_size = 128
    for i in range(0, len(unique_qs), batch_size):
        batch = unique_qs[i : i + batch_size]
        vecs = model.encode(batch, show_progress_bar=False)
        for text, vec in zip(batch, vecs):
            cache[text] = vec.tolist()
        print(f"  {min(i + batch_size, len(unique_qs))}/{len(unique_qs)}")

    return cache


# ── Ingest one persona ────────────────────────────────────────────────────────
def ingest_persona(persona_id: int, rows: list[dict], embed_cache: dict, timeout: int = 600) -> str | None:
    namespace = f"{NS_PREFIX}{persona_id}"

    records = []
    for row in rows:
        question = str(row.get("user_question_or_message", ""))
        vec = embed_cache.get(question, [0.0] * EMBED_DIM)

        records.append({
            "question_id":               str(row.get("question_id", "")),
            "persona_id":                str(persona_id),
            "question_type":             str(row.get("question_type", "")),
            "topic":                     str(row.get("topic", "")),
            "user_question_or_message":  question,
            "correct_answer":            str(row.get("correct_answer", "")),
            "all_options":               str(row.get("all_options", "")),
            "distance_to_ref_in_blocks": str(row.get("distance_to_ref_in_blocks", "")),
            "distance_to_ref_proportion_in_context": str(
                row.get("distance_to_ref_proportion_in_context", "")
            ),
            VECTOR_COL: json.dumps(vec),
        })

    df = pd.DataFrame(records, columns=ROW_FIELDS)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)

    try:
        resp = requests.post(
            f"{SERVER_URL}/build_ingest_data/",
            headers={"X-API-Key": API_KEY} if API_KEY else {},
            files={"file": (f"personamem_persona_{persona_id}.csv", buf, "text/csv")},
            data={
                "dim":             EMBED_DIM,
                "seed":            42,
                "depth":           3,
                "db_name":         DB_NAME,
                "namespace":       namespace,
                "template_schema": PERSONAMEM_QA_SCHEMA,
                "vector_col":      VECTOR_COL,
            },
            timeout=timeout,
        )

        if resp.ok:
            result = resp.json()
            print(
                f"    ✓ Persona {persona_id}: {result.get('rows_added', 0)} rows "
                f"→ namespace={namespace} (vector_source={result.get('vector_source', '?')})"
            )
            return namespace
        else:
            print(f"    ✗ Persona {persona_id}: {resp.status_code} - {resp.text[:200]}")
            return None

    except requests.exceptions.Timeout:
        print(f"    ✗ Persona {persona_id}: timed out after {timeout}s")
        return None
    except Exception as e:
        print(f"    ✗ Persona {persona_id}: {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    wipe = "--wipe" in sys.argv

    print(f"\n{'=' * 60}")
    print("PersonaMem 32k → HyperBinder Ingest")
    print(f"{'=' * 60}")
    print(f"Server:        {SERVER_URL}")
    print(f"DB:            {DB_NAME}")
    print(f"Split:         {SPLIT}")
    print(f"API Key:       {'Set' if API_KEY else 'Not set'}")
    print(f"Wipe existing: {wipe}")
    print(f"{'=' * 60}\n")

    if wipe:
        wipe_namespaces()
        print("\n✅ Wipe complete. Proceeding with fresh ingestion...\n")

    # 1. Load dataset
    print("Loading PersonaMem dataset from HuggingFace...")
    ds = load_dataset("bowen-upenn/PersonaMem")
    df_full = ds[SPLIT].to_pandas()
    print(f"Loaded {len(df_full)} rows for split '{SPLIT}'")

    # 2. Group by persona
    personas = df_full.groupby("persona_id")
    persona_ids = sorted(personas.groups.keys())
    print(f"Found {len(persona_ids)} unique personas\n")

    # 3. Precompute embeddings for all questions
    all_questions = df_full["user_question_or_message"].tolist()
    embed_cache = precompute_embeddings(all_questions)

    # 4. Ingest per persona
    print(f"\n{'=' * 60}")
    print("Ingesting per-persona namespaces")
    print(f"{'=' * 60}")

    successful = 0
    for persona_id in persona_ids:
        rows = personas.get_group(persona_id).to_dict("records")
        print(f"\nPersona {persona_id}  ({len(rows)} questions)")

        ns = ingest_persona(persona_id, rows, embed_cache)
        if ns:
            save_namespace(persona_id, ns)
            successful += 1
        
        time.sleep(0.2)

    print(f"\n{'=' * 60}")
    print(f"✅ PersonaMem ingest complete!")
    print(f"   Processed: {successful}/{len(persona_ids)} personas")
    print(f"   Namespaces saved to: {NS_FILE}")
    print(f"{'=' * 60}")