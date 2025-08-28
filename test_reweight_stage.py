# test_reweight_stage.py
import os, json, time
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Load env (secrets.env first, then .env)
load_dotenv(find_dotenv(filename="secrets.env", usecwd=True) or find_dotenv(), override=False)

from probe_stage import run_probe_stage  # uses OpenRouter
from reweight_stage import compute_doc_weights

def _ts():
    return time.strftime("%Y%m%d_%H%M%S")

def main():
    question = "Where did the first known cases of MERS occur?"
    top_k = 5
    n_probes = 5

    # Run probe stage (retrieval + OpenRouter probes)
    stage = run_probe_stage(question, top_k=top_k, n_probes=n_probes)

    # Reweight
    ranked = compute_doc_weights(
        stage["pinecone_results"],
        stage["probes"],
        stage["mean_probe_similarity"],
    )

    # Pretty print summary
    print(f"[test] Question: {question}")
    print(f"[test] top_k={top_k} n_probes={n_probes}")
    print("\n--- PROBES ---")
    for p in stage["probes"]:
        print("-", p)
    print(f"\nMean probe similarity: {round(stage['mean_probe_similarity'], 3)}")

    print("\n--- TOP (by final_weight) ---")
    for r in ranked[:min(5, len(ranked))]:
        preview = (r["text"][:160] + "…") if len(r["text"]) > 160 else r["text"]
        print(f"- {r['id']} | weight={round(r['final_weight'],3)} | cites={r['citation_count']} | red={round(r['redundancy_penalty'],2)}")
        print(f"  {preview}\n")

    # Save JSON
    out = {
        "question": question,
        "top_k": top_k,
        "n_probes": n_probes,
        "probes": stage["probes"],
        "mean_probe_similarity": float(stage["mean_probe_similarity"]),
        "pinecone_results_ids": [m.get("id") for m in stage["pinecone_results"].get("matches", [])],
        "ranked": ranked,
    }
    Path("results").mkdir(exist_ok=True, parents=True)
    out_path = f"results/test_reweight_{_ts()}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[saved] → {out_path}")

if __name__ == "__main__":
    main()
