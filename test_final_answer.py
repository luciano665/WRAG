# test_final_answer.py
import os, json, time
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(filename="secrets.env", usecwd=True) or find_dotenv(), override=False)

from probe_stage import run_probe_stage
from reweight_stage import compute_doc_weights
from final_ans import generate_final_answer

def ts():
    return time.strftime("%Y%m%d_%H%M%S")

if __name__ == "__main__":
    q = "Where did the first known cases of MERS occur?"
    stage = run_probe_stage(q, top_k=5, n_probes=5)
    ranked = compute_doc_weights(stage["pinecone_results"], stage["probes"], stage["mean_probe_similarity"])
    ans = generate_final_answer(q, ranked)
    print(json.dumps(ans, ensure_ascii=False, indent=2))

    # save
    import os
    os.makedirs("results", exist_ok=True)
    path = f"results/test_final_answer_{ts()}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "question": q,
            "probes": stage["probes"],
            "top_ids": [m.get("id") for m in stage["pinecone_results"].get("matches", [])],
            "answer": ans
        }, f, ensure_ascii=False, indent=2)
    print(f"[saved] → {path}")
