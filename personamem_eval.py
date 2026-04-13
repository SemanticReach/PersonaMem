# personamem_eval.py - PersonaMem 32k Evaluation using Slot Search
import json
import requests
import os
import time
from dotenv import load_dotenv
from typing import List, Dict

from datasets import load_dataset

load_dotenv()

SERVER_URL    = os.getenv("HB_SERVER_URL", "http://localhost:8000")
API_KEY       = os.getenv("HB_API_KEY", "")
DB_NAME       = "fractal_db"
NS_PREFIX     = "personamem_32k_persona_"
NS_FILE       = "personamem_namespaces.json"
SPLIT         = "32k"
REQUEST_DELAY = 0.2


# ── Namespace helpers ─────────────────────────────────────────────────────────

def load_namespaces() -> Dict[int, str]:
    if os.path.exists(NS_FILE):
        with open(NS_FILE, "r") as f:
            return {int(k): v for k, v in json.load(f).items()}
    return {}


# ── Slot search ───────────────────────────────────────────────────────────────

def search_slot_answer(question: str, persona_id: int, expected_qid: str) -> Dict:
    """
    Query the persona namespace and return the bound correct_answer.
    Pins retrieval to the exact row via question_id exact filter,
    preventing collisions when multiple rows share identical question text.
    """
    namespace = f"{NS_PREFIX}{persona_id}"

    try:
        resp = requests.post(
            f"{SERVER_URL}/compose/search_slots/{DB_NAME}/{namespace}",
            headers={"X-API-Key": API_KEY} if API_KEY else {},
            json={
                "slot_queries": {
                    "user_question_or_message": {
                        "query":    question,
                        "weight":   1.0,
                        "encoding": "semantic",
                    },
                    "question_id": {
                        "query":    expected_qid,
                        "weight":   1.0,
                        "mode":     "filter",
                        "encoding": "exact",
                    },
                },
                "top_k": 1,
            },
            timeout=30,
        )

        if resp.ok:
            results = resp.json().get("results", [])
            if results:
                data = results[0].get("data", {})
                return {
                    "correct_answer":        data.get("correct_answer", ""),
                    "question_type":         data.get("question_type", ""),
                    "topic":                 data.get("topic", ""),
                    "distance_prop":         data.get("distance_to_ref_proportion_in_context", ""),
                    "confidence":            results[0].get("_score", results[0].get("score", 0)),
                    "retrieved_question_id": data.get("question_id", ""),
                    "retrieved_question":    data.get("user_question_or_message", ""),
                }
        return {
            "correct_answer": "", "confidence": 0,
            "question_type": "", "topic": "", "distance_prop": "",
            "retrieved_question_id": "", "retrieved_question": "",
        }

    except Exception as e:
        print(f"      Slot search error: {e}")
        return {
            "correct_answer": "", "confidence": 0,
            "question_type": "", "topic": "", "distance_prop": "",
            "retrieved_question_id": "", "retrieved_question": "",
        }


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_answer(retrieved: str, ground_truth: str) -> bool:
    """Exact match — PersonaMem answers are single letters (A/B/C/D)."""
    return retrieved.strip().upper() == ground_truth.strip().upper()


# ── Per-persona eval ──────────────────────────────────────────────────────────

def evaluate_persona(persona_id: int, rows: List[Dict]) -> Dict:
    print(f"\n  Evaluating {len(rows)} questions via slot search...")

    results   = []
    correct   = 0
    not_found = 0

    for i, row in enumerate(rows, 1):
        question       = str(row.get("user_question_or_message", ""))
        ground_truth   = str(row.get("correct_answer", ""))
        q_type         = str(row.get("question_type", ""))
        topic          = str(row.get("topic", ""))
        dist_prop      = str(row.get("distance_to_ref_proportion_in_context", ""))
        expected_qid   = str(row.get("question_id", ""))

        print(
            f"\n    [{i:3}/{len(rows)}] {q_type} | {topic[:20]:20s} | "
            f"{question[:55]}...",
            end=" ", flush=True,
        )

        # Pass expected_qid to pin retrieval to exact row
        result             = search_slot_answer(question, persona_id, expected_qid)
        retrieved          = result.get("correct_answer", "")
        confidence         = result.get("confidence", 0)
        retrieved_qid      = result.get("retrieved_question_id", "")
        retrieved_question = result.get("retrieved_question", "")

        if not retrieved:
            not_found += 1
            retrieved = "?"

        is_correct = score_answer(retrieved, ground_truth)
        if is_correct:
            correct += 1

        marker = "✓" if is_correct else "✗"
        if retrieved == "?":
            print(f"→ NOT FOUND (GT: '{ground_truth}') ✗")
        else:
            print(f"→ '{retrieved}' (GT: '{ground_truth}') [conf={confidence:.3f}] {marker}")

        # ── Console debug for any remaining wrong answers ─────────────────
        if not is_correct and retrieved != "?":
            qid_match = retrieved_qid == expected_qid
            print(f"      ┌─ WRONG RETRIEVAL DEBUG ──────────────────────────────")
            print(f"      │ Expected  QID : {expected_qid}")
            print(f"      │ Retrieved QID : {retrieved_qid}  {'✓ same row' if qid_match else '✗ DIFFERENT ROW'}")
            print(f"      │ Expected  Q   : {question[:80]}")
            print(f"      │ Retrieved Q   : {retrieved_question[:80]}")
            print(f"      └──────────────────────────────────────────────────────")

        results.append({
            "question_id":           expected_qid,
            "question":              question,
            "ground_truth":          ground_truth,
            "retrieved":             retrieved,
            "retrieved_question_id": retrieved_qid,
            "retrieved_question":    retrieved_question,
            "same_row":              retrieved_qid == expected_qid,
            "question_type":         q_type,
            "topic":                 topic,
            "distance_prop":         dist_prop,
            "confidence":            confidence,
            "correct":               is_correct,
        })

        time.sleep(REQUEST_DELAY)

    accuracy = correct / len(rows) if rows else 0

    wrong          = [r for r in results if not r["correct"] and r["retrieved"] != "?"]
    wrong_same_row = sum(1 for r in wrong if r["same_row"])
    wrong_diff_row = sum(1 for r in wrong if not r["same_row"])

    # ── Break down by question_type ───────────────────────────────────────
    type_stats: Dict[str, Dict] = {}
    for r in results:
        qt = r["question_type"]
        if qt not in type_stats:
            type_stats[qt] = {"correct": 0, "total": 0}
        type_stats[qt]["total"] += 1
        if r["correct"]:
            type_stats[qt]["correct"] += 1

    # ── Break down by topic ───────────────────────────────────────────────
    topic_stats: Dict[str, Dict] = {}
    for r in results:
        t = r["topic"]
        if t not in topic_stats:
            topic_stats[t] = {"correct": 0, "total": 0}
        topic_stats[t]["total"] += 1
        if r["correct"]:
            topic_stats[t]["correct"] += 1

    # ── Break down by distance bucket (near / mid / far) ─────────────────
    def dist_bucket(prop_str: str) -> str:
        try:
            p = float(prop_str)
            if p < 0.33:
                return "near"
            elif p < 0.66:
                return "mid"
            else:
                return "far"
        except (ValueError, TypeError):
            return "unknown"

    dist_stats: Dict[str, Dict] = {}
    for r in results:
        bucket = dist_bucket(r["distance_prop"])
        if bucket not in dist_stats:
            dist_stats[bucket] = {"correct": 0, "total": 0}
        dist_stats[bucket]["total"] += 1
        if r["correct"]:
            dist_stats[bucket]["correct"] += 1

    return {
        "persona_id":       persona_id,
        "total_questions":  len(rows),
        "correct":          correct,
        "not_found":        not_found,
        "accuracy":         accuracy,
        "wrong_same_row":   wrong_same_row,
        "wrong_diff_row":   wrong_diff_row,
        "type_accuracy":    {qt: s["correct"] / s["total"] for qt, s in type_stats.items()},
        "topic_accuracy":   {t:  s["correct"] / s["total"] for t,  s in topic_stats.items()},
        "dist_accuracy":    {b:  s["correct"] / s["total"] for b,  s in dist_stats.items()},
        "details":          results,
        "wrong_answers": [
            {
                "persona_id":         persona_id,
                "expected_qid":       r["question_id"],
                "retrieved_qid":      r["retrieved_question_id"],
                "same_row":           r["same_row"],
                "ground_truth":       r["ground_truth"],
                "retrieved_answer":   r["retrieved"],
                "confidence":         r["confidence"],
                "question_type":      r["question_type"],
                "topic":              r["topic"],
                "expected_question":  r["question"],
                "retrieved_question": r["retrieved_question"],
            }
            for r in results if not r["correct"] and r["retrieved"] != "?"
        ],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'=' * 60}")
    print("PersonaMem 32k Evaluation (Slot Search)")
    print(f"{'=' * 60}")
    print(f"Server:    {SERVER_URL}")
    print(f"DB:        {DB_NAME}")
    print(f"Split:     {SPLIT}")
    print(f"NS Prefix: {NS_PREFIX}")
    print(f"{'=' * 60}\n")

    namespaces = load_namespaces()
    if not namespaces:
        print("ERROR: No ingested namespaces found!")
        print(f"Run personamem_ingest.py first — expected file: {NS_FILE}")
        return

    print(f"Found {len(namespaces)} ingested personas: {sorted(namespaces.keys())}")

    print("\nLoading PersonaMem dataset from HuggingFace...")
    ds = load_dataset("bowen-upenn/PersonaMem")
    df = ds[SPLIT].to_pandas()
    print(f"Loaded {len(df)} rows\n")

    personas    = df.groupby("persona_id")
    all_results = []

    for persona_id in sorted(namespaces.keys()):
        if persona_id not in personas.groups:
            print(f"Persona {persona_id} not found in dataset — skipping")
            continue

        rows = personas.get_group(persona_id).to_dict("records")

        print(f"\n{'=' * 50}")
        print(f"Persona {persona_id}  |  namespace: {NS_PREFIX}{persona_id}")
        print(f"Questions: {len(rows)}")
        print(f"{'=' * 50}")

        result = evaluate_persona(persona_id, rows)
        all_results.append(result)

        print(f"\n  ✓ Persona {persona_id} Results:")
        print(f"      Accuracy:         {result['correct']}/{result['total_questions']} = {result['accuracy']:.1%}")
        print(f"      Not found:        {result['not_found']}")
        print(f"      Wrong (same row): {result['wrong_same_row']}  ← correct row, answer mismatch")
        print(f"      Wrong (diff row): {result['wrong_diff_row']}  ← wrong row retrieved")
        print(f"      By type:")
        for qt, acc in result["type_accuracy"].items():
            print(f"        {qt}: {acc:.1%}")
        print(f"      By distance:")
        for bucket, acc in result["dist_accuracy"].items():
            print(f"        {bucket}: {acc:.1%}")

    if not all_results:
        print("No results to summarize.")
        return

    total_correct    = sum(r["correct"]         for r in all_results)
    total_questions  = sum(r["total_questions"]  for r in all_results)
    total_not_found  = sum(r["not_found"]        for r in all_results)
    total_wrong_same = sum(r["wrong_same_row"]   for r in all_results)
    total_wrong_diff = sum(r["wrong_diff_row"]   for r in all_results)
    overall_acc      = total_correct / total_questions if total_questions else 0

    all_wrong_answers = [wa for r in all_results for wa in r["wrong_answers"]]

    agg_type: Dict[str, Dict] = {}
    agg_dist: Dict[str, Dict] = {}

    def dist_bucket(prop_str):
        try:
            p = float(prop_str)
            return "near" if p < 0.33 else ("mid" if p < 0.66 else "far")
        except (ValueError, TypeError):
            return "unknown"

    for r in all_results:
        for detail in r["details"]:
            qt = detail["question_type"]
            if qt not in agg_type:
                agg_type[qt] = {"correct": 0, "total": 0}
            agg_type[qt]["total"] += 1
            if detail["correct"]:
                agg_type[qt]["correct"] += 1

            b = dist_bucket(detail["distance_prop"])
            if b not in agg_dist:
                agg_dist[b] = {"correct": 0, "total": 0}
            agg_dist[b]["total"] += 1
            if detail["correct"]:
                agg_dist[b]["correct"] += 1

    print(f"\n{'=' * 60}")
    print("PERSONAMEM 32k OVERALL RESULTS")
    print(f"{'=' * 60}")
    print(f"Personas evaluated:  {len(all_results)}")
    print(f"Total questions:     {total_questions}")
    print(f"Total correct:       {total_correct}")
    print(f"Total not found:     {total_not_found}")
    print(f"Wrong (same row):    {total_wrong_same}  ← correct row retrieved, answer mismatch")
    print(f"Wrong (diff row):    {total_wrong_diff}  ← wrong row retrieved entirely")
    print(f"OVERALL ACCURACY:    {total_correct}/{total_questions} = {overall_acc:.1%}")
    print(f"\nBy question_type:")
    for qt, s in sorted(agg_type.items()):
        print(f"  {qt:50s}: {s['correct']}/{s['total']} = {s['correct']/s['total']:.1%}")
    print(f"\nBy distance bucket (proportion in context):")
    for b in ["near", "mid", "far", "unknown"]:
        if b in agg_dist:
            s = agg_dist[b]
            print(f"  {b:8s}: {s['correct']}/{s['total']} = {s['correct']/s['total']:.1%}")
    print(f"{'=' * 60}")

    timestamp   = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"personamem_32k_results_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp":          timestamp,
                "method":             "slot_search_pinned",
                "split":              SPLIT,
                "server_url":         SERVER_URL,
                "db_name":            DB_NAME,
                "personas_evaluated": sorted(namespaces.keys()),
                "total_questions":    total_questions,
                "total_correct":      total_correct,
                "total_not_found":    total_not_found,
                "wrong_same_row":     total_wrong_same,
                "wrong_diff_row":     total_wrong_diff,
                "overall_accuracy":   overall_acc,
                "type_accuracy":      {qt: s["correct"] / s["total"] for qt, s in agg_type.items()},
                "dist_accuracy":      {b:  s["correct"] / s["total"] for b,  s in agg_dist.items()},
                "wrong_answers":      all_wrong_answers,
                "results":            all_results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    main()