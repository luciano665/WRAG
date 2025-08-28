# test_probe_stage.py
import os, json, time, argparse
import numpy as np

# Use the OpenRouter-powered probe stage you just installed
from probe_stage import run_probe_stage

def ts():
    return time.strftime("%Y%m%d_%H%M%S")

def to_serializable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x

def main():
    ap = argparse.ArgumentParser(description="Quick probe-stage smoke test (OpenRouter + Pinecone).")
    ap.add_argument("--question", type=str,
                    default="Where did the first known cases of MERS occur?",
                    help="Question to generate probes for.")
    ap.add_argument("--subset", type=str, default="covidqa",
                    help="Metadata filter for your index: filters={'config': {'$in': [subset]}}")
    ap.add_argument("--top-k", type=int, default=5, dest="top_k")
    ap.add_argument("--n-probes", type=int, default=5, dest="n_probes")
    ap.add_argument("--out", type=str, default=f"results/test_probe_{ts()}.json",
                    help="Path to save JSON (will be created).")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Same filter convention as maintest.py
    metadata_filter = {"config": {"$in": [args.subset]}} if args.subset else None

    print(f"[test] Question: {args.question}")
    print(f"[test] subset={args.subset} top_k={args.top_k} n_probes={args.n_probes}")

    try:
        out = run_probe_stage(
            question=args.question,
            top_k=args.top_k,
            n_probes=args.n_probes,
            metadata_filter=metadata_filter
        )
    except Exception as e:
        print(f"[error] run_probe_stage failed: {type(e).__name__}: {e}")
        return

    # Pretty print summary
    probes = out.get("probes", [])
    mean_sim = out.get("mean_probe_similarity")
    matches = (out.get("pinecone_results") or {}).get("matches", []) or []

    print("\n--- PROBES ---")
    for i, p in enumerate(probes, 1):
        print(f"{i:02d}. {p}")
    print(f"\nMean probe similarity: {mean_sim:.3f}" if mean_sim is not None else "\nMean probe similarity: n/a")

    print("\n--- RETRIEVED DOC IDS (up to 5) ---")
    for m in matches[:5]:
        mid = m.get("id")
        meta = m.get("metadata", {}) or {}
        preview = (meta.get("text") or meta.get("chunk") or meta.get("content") or "")[:140].replace("\n", " ")
        print(f"- {mid}: {preview}{'…' if len(preview)==140 else ''}")

    # Save JSON (convert numpy arrays)
    payload = {
        "question": args.question,
        "subset": args.subset,
        "top_k": args.top_k,
        "n_probes": args.n_probes,
        "result": {
            **out,
            "probe_similarity_matrix": to_serializable(out.get("probe_similarity_matrix")),
        },
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=to_serializable)

    print(f"\n[saved] → {args.out}")

if __name__ == "__main__":
    main()
