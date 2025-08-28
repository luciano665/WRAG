# main.py
import os
import json
import argparse
import numpy as np

# --- Load env BEFORE importing other modules (so os.getenv works) ---
try:
    from dotenv import load_dotenv, find_dotenv
    _ = load_dotenv(find_dotenv(filename="secrets.env", usecwd=True) or find_dotenv(), override=False)
except Exception:
    pass

# --- Imports: probe + reweigh ---
try:
    from probe_stage import run_probe_stage
except ImportError as e:
    raise SystemExit(
        "Could not import probe_stage. Ensure it's alongside main.py or on PYTHONPATH.\n"
        f"ImportError: {e}"
    )

# Support either module name: reweigh.py or reweight_stage.py
compute_doc_weights = None
try:
    from reweigh import compute_doc_weights
except Exception:
    try:
        from reweight_stage import compute_doc_weights
    except Exception as e:
        raise SystemExit(
            "Could not import compute_doc_weights from reweigh.py or reweight_stage.py.\n"
            f"ImportError: {e}"
        )

# --- Final answer generation (Gemini) ---
import time, random
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

# RAGBench quenstion helper
try:
    from ragbench_questions import get_ragbench_questions
except Exception as e:
    get_ragbench_questions = None

GEMINI_KEY   = os.getenv("GEMINI_API_KEY", "realkey")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
genai.configure(api_key=GEMINI_KEY)

def warn_missing_env():
    missing = []
    for k in ["GEMINI_API_KEY", "PINECONE_API_KEY", "INDEX_NAME", "NAMESPACE"]:
        if not os.getenv(k):
            missing.append(k)
    if missing:
        print(f"[warn] Missing env vars: {', '.join(missing)}")
        print("       Ensure they're set in secrets.env or your environment.")

def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def _truncate(s: str, n: int = 1200) -> str:
    if s is None:
        return ""
    return s if len(s) <= n else s[:n]

def _format_context(weighted_docs, top_m: int, char_limit: int):
    lines = []
    for i, d in enumerate(weighted_docs[:top_m], 1):
        lines.append(f"[Doc {i} | id={d['id']} | w={d['final_weight']:.3f}] {_truncate(d['text'], char_limit)}")
    return "\n\n".join(lines), [d["id"] for d in weighted_docs[:top_m]]

def generate_final_answer(question: str, weighted_docs: list,
                          model_name: str,
                          top_m_docs: int = 5,
                          doc_char_lim: int = 1200,
                          temperature: float = 0.2,
                          max_output_tokens: int = 256):
    """
    Ask Gemini to produce a JSON answer with fields:
      - answer (string)
      - period_years (number or null)
      - yearly_rate_percent (number or null)
      - citations (array of 1–3 doc IDs; prefer 2+ when available)
      - conflict (boolean)
      - conflict_note (string; brief, optional)
    """
    context_str, doc_ids = _format_context(weighted_docs, top_m_docs, doc_char_lim)

    prompt = (
        "You are a careful QA assistant. Use ONLY the provided passages.\n"
        "Answer the question concisely and cite the supporting Doc IDs.\n"
        "Return JSON with fields:\n"
        "  - answer (string)\n"
        "  - period_years (number or null)\n"
        "  - yearly_rate_percent (number or null)\n"
        "  - citations (array of 1–3 doc IDs; prefer at least 2 when available)\n"
        "  - conflict (boolean)\n"
        "  - conflict_note (string; brief, optional)\n\n"
        f"Question: {question}\n\n"
        f"Passages:\n{context_str}\n\n"
        "Rules:\n"
        " - Do not use outside knowledge.\n"
        " - If evidence indicates a fixed amortization period of N years, set period_years = N and yearly_rate_percent = 100 / N (straight-line).\n"
        " - If evidence indicates indefinite-lived trademarks (no amortization), set period_years = null and yearly_rate_percent = 0.\n"
        " - If the passages conflict (e.g., both '5 years' and 'indefinite'), set conflict = true and briefly explain in conflict_note.\n"
        " - Citations must be a subset of the given Doc IDs and reflect the specific claim.\n"
        "JSON:"
    )

    model = genai.GenerativeModel(model_name)
    for attempt in range(6):
        try:
            resp = model.generate_content(
                prompt,
                generation_config={"temperature": temperature, "max_output_tokens": max_output_tokens}
            )
            return resp.text
        except ResourceExhausted:
            wait = min(32, (2 ** attempt) + random.random())
            print(f"[Gemini 429] Backing off {wait:.1f}s (attempt {attempt+1}/6)")
            time.sleep(wait)
    raise RuntimeError("Gemini rate limit/quota reached repeatedly.")

def main():
    parser = argparse.ArgumentParser(
        description="WRAG: run probe stage, reweigh, and final answer generation."
    )
    parser.add_argument("-q", "--question", type=str, help="Question to process.")

    # RAGBench batch mode flags
    parser.add_argument("--ragbench", action="store_true",
                        help="Pull questions from RAGBench instead of --question.")
    parser.add_argument("--rb-subset", action="append",
                        help="RAGBench subset(s), repeatable. Ex: --rb-subset hotpotqa")
    parser.add_argument("--rb-split", action="append",
                        help="RAGBench split(s), repeatable. Ex: --rb-split validation")
    parser.add_argument("--rb-limit", type=int, default=10,
                        help="How many RAGBench questions to run (default: 10)")
    parser.add_argument("--rb-auto-install", action="store_true",
                        help="Auto-install `datasets` in ragbench helper if missing")

    # Retrieval controls
    parser.add_argument("--top-k", type=int, default=5, help="Top-K docs to retrieve (unique) (default: 5).")
    parser.add_argument("--n-probes", type=int, default=5, help="Number of keyword probes (default: 5).")
    parser.add_argument("--show-matrix", action="store_true", help="Print full probe similarity matrix.")
    parser.add_argument("--top-m", type=int, default=5, help="How many reweighted docs to display/save/pass (default: 5).")

    # Reweigh hyperparams
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for agreement (default: 1.0).")
    parser.add_argument("--beta",  type=float, default=0.6, help="Weight for citation_count (default: 0.6).")
    parser.add_argument("--gamma", type=float, default=0.5, help="Weight for redundancy penalty (default: 0.5).")
    parser.add_argument("--citation-top-n", type=int, default=None,
                        help="Soft-citation: count doc cited if in top-N per probe (overrides threshold).")
    parser.add_argument("--citation-sim-threshold", type=float, default=0.35,
                        help="Soft-citation similarity threshold (ignored if --citation-top-n set).")

    # Retrieval metadata filter controls
    parser.add_argument(
        "--allow-configs", type=str, default="",
        help="Comma-separated metadata.config values to include (e.g., 'finqa,tatqa')."
    )
    parser.add_argument(
        "--pc-filter-json", type=str, default="",
        help="Raw Pinecone metadata filter JSON string (overrides --allow-configs)."
    )

    # Final answer generation controls
    parser.add_argument("--no-final", action="store_true",
                        help="Skip final answer generation (default is to run it).")
    parser.add_argument("--ans-model", type=str, default=os.getenv("GEMINI_MODEL", GEMINI_MODEL),
                        help="Gemini model for final answer (default env GEMINI_MODEL or gemini-1.5-flash).")
    parser.add_argument("--ans-temp", type=float, default=0.2, help="Final answer temperature (default: 0.2).")
    parser.add_argument("--ans-max-tokens", type=int, default=256, help="Max tokens for final answer (default: 256).")
    parser.add_argument("--ans-doc-char-limit", type=int, default=1200,
                        help="Per-doc char limit in context (default: 1200).")

    # Output / save
    parser.add_argument("--save", type=str, default="",
                        help="Save probe+reweigh results to JSON (e.g., results/wrag_probe_reweigh.json).")
    parser.add_argument("--save-answer", type=str, default="",
                        help="Save final answer JSON (raw model text) to a file (e.g., results/final_answer.json).")

    args = parser.parse_args()

    warn_missing_env()

    # Build Pinecone metadata filter
    metadata_filter = None
    if args.pc_filter_json:
        try:
            metadata_filter = json.loads(args.pc_filter_json)
        except Exception as e:
            raise SystemExit(f"Invalid --pc-filter-json: {e}")
    elif args.allow_configs:
        cfgs = [c.strip() for c in args.allow_configs.split(",") if c.strip()]
        if cfgs:
            metadata_filter = {"config": {"$in": cfgs}}
    
    # RAGBench batch
    if args.ragbench:
        if get_ragbench_questions is None:
            raise SystemExit("ragbench_qeustion.py not found or import failed.")
        
        subsets = args.rb_subsset or None
        splits = tuple(args.rb_split) if args.rb_split else ("train", "validation", "test")

        try:
            questions = get_ragbench_questions(
                subsets=subsets,
                splits=splits,
                auto_install=args.rb_auto_install
            )
        except Exception as e:
            raise SystemExit(f"Failed to load RAGBench questions: {e}")
        
        if not questions:
            raise SystemExit("No RAGBench questions found for requested subsets/splits.")
        
        limit = max(1, int(args.rb_limit))
        print(f"[RAGBench] Running on {min(limit, len(questions))} / {len(questions)} questions "
              f"(subsets={subsets or 'ALL'}, splits={splits})")
            
        
        for idx, q in enumerate(questions[:limit], start=1):
            question = q.strip()
            if not question:
                continue

            print(f"\n========== RAGBench Q{idx}: {question} ==========")

            # === Probe Stage ===
            stage = run_probe_stage(
                question,
                top_k=args.top_k,
                n_probes=args.n_probes,
                metadata_filter=metadata_filter
            )

            print("\n=== WRAG • Probe Stage ===")
            print("Question:", stage["question"])
            print("Probes:")
            for i, p in enumerate(stage["probes"], 1):
                print(f"  {i}. {p}")
            print("Mean probe similarity:", round(stage["mean_probe_similarity"], 3))

            if args.show_matrix:
                sim = stage["probe_similarity_matrix"]
                print("\nProbe similarity matrix (rounded):")
                print(np.round(sim, 2))

            # === Reweigh Stage ===
            ranked = compute_doc_weights(
                pinecone_results=stage["pinecone_results"],
                probes=stage["probes"],
                mean_probe_similarity=stage["mean_probe_similarity"],
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
                citation_top_n=args.citation_top_n,
                citation_sim_threshold=args.citation_sim_threshold,
            )

            print("\n=== WRAG • Reweigh Stage ===")
            if not ranked:
                print("No documents to reweigh.")
            else:
                show_n = min(args.top_m, len(ranked))
                print(f"Top {show_n} documents by final_weight:")
                header = f"{'#':>2}  {'final_w':>8}  {'cites':>5}  {'red':>5}  {'retr':>6}  id / preview"
                print(header)
                print("-" * len(header))
                for i, d in enumerate(ranked[:show_n], 1):
                    preview = (d['text'][:97] + '...') if len(d['text']) > 100 else d['text']
                    print(f"{i:>2}  {d['final_weight']:>8.3f}  {d['citation_count']:>5}  "
                          f"{d['redundancy_penalty']:>5.2f}  {d['retrieval_score']:>6.3f}  "
                          f"{d['id']}  |  {preview}")

            # Save probe+reweigh JSON (optional) — NOTE: this overwrites for each Q if you reuse the same path
            if args.save:
                os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
                payload = {
                    "question": stage["question"],
                    "probes": stage["probes"],
                    "mean_probe_similarity": stage["mean_probe_similarity"],
                    "probe_similarity_matrix": to_serializable(stage["probe_similarity_matrix"]) if args.show_matrix else None,
                    "ranked_docs": [
                        {
                            "id": d["id"],
                            "final_weight": d["final_weight"],
                            "citation_count": d["citation_count"],
                            "redundancy_penalty": d["redundancy_penalty"],
                            "retrieval_score": d["retrieval_score"],
                            "text": d["text"],
                        }
                        for d in ranked[:args.top_m]
                    ],
                    "hyperparams": {
                        "alpha": args.alpha,
                        "beta": args.beta,
                        "gamma": args.gamma,
                        "citation_top_n": args.citation_top_n,
                        "citation_sim_threshold": args.citation_sim_threshold,
                        "top_k": args.top_k,
                        "n_probes": args.n_probes,
                        "metadata_filter": metadata_filter,
                    },
                }
                payload = {k: v for k, v in payload.items() if v is not None}
                with open(args.save, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                print(f"\nSaved results to: {args.save}")

            # === Final Answer Stage (default run unless --no-final) ===
            if not args.no_final:
                if not ranked:
                    print("Skipping final answer: no ranked docs.")
                else:
                    print("\n=== WRAG • Final Answer Generation ===")
                    answer_text = generate_final_answer(
                        question=question,
                        weighted_docs=ranked[:args.top_m],
                        model_name=args.ans_model,
                        top_m_docs=args.top_m,
                        doc_char_lim=args.ans_doc_char_limit,
                        temperature=args.ans_temp,
                        max_output_tokens=args.ans_max_tokens
                    )
                    print("\nModel output (expected JSON):")
                    print(answer_text)

                    if args.save_answer:
                        os.makedirs(os.path.dirname(args.save_answer) or ".", exist_ok=True)
                        with open(args.save_answer, "w", encoding="utf-8") as f:
                            f.write(answer_text if isinstance(answer_text, str) else json.dumps(answer_text))
                        print(f"\nSaved final answer to: {args.save_answer}")

        return 


    question = args.question or input("Enter your question: ").strip()
    if not question:
        raise SystemExit("No question provided.")

    # === Probe Stage ===
    stage = run_probe_stage(
        question,
        top_k=args.top_k,
        n_probes=args.n_probes,
        metadata_filter=metadata_filter  # pass filter to retrieval
    )

    print("\n=== WRAG • Probe Stage ===")
    print("Question:", stage["question"])
    print("Probes:")
    for i, p in enumerate(stage["probes"], 1):
        print(f"  {i}. {p}")
    print("Mean probe similarity:", round(stage["mean_probe_similarity"], 3))

    if args.show_matrix:
        sim = stage["probe_similarity_matrix"]
        print("\nProbe similarity matrix (rounded):")
        print(np.round(sim, 2))

    # === Reweigh Stage ===
    ranked = compute_doc_weights(
        pinecone_results=stage["pinecone_results"],
        probes=stage["probes"],
        mean_probe_similarity=stage["mean_probe_similarity"],
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        citation_top_n=args.citation_top_n,
        citation_sim_threshold=args.citation_sim_threshold,
    )

    print("\n=== WRAG • Reweigh Stage ===")
    if not ranked:
        print("No documents to reweigh.")
    else:
        show_n = min(args.top_m, len(ranked))
        print(f"Top {show_n} documents by final_weight:")
        header = f"{'#':>2}  {'final_w':>8}  {'cites':>5}  {'red':>5}  {'retr':>6}  id / preview"
        print(header)
        print("-" * len(header))
        for i, d in enumerate(ranked[:show_n], 1):
            preview = (d['text'][:97] + '...') if len(d['text']) > 100 else d['text']
            print(f"{i:>2}  {d['final_weight']:>8.3f}  {d['citation_count']:>5}  "
                  f"{d['redundancy_penalty']:>5.2f}  {d['retrieval_score']:>6.3f}  "
                  f"{d['id']}  |  {preview}")

    # Save probe+reweigh JSON (optional)
    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        payload = {
            "question": stage["question"],
            "probes": stage["probes"],
            "mean_probe_similarity": stage["mean_probe_similarity"],
            "probe_similarity_matrix": to_serializable(stage["probe_similarity_matrix"]) if args.show_matrix else None,
            "ranked_docs": [
                {
                    "id": d["id"],
                    "final_weight": d["final_weight"],
                    "citation_count": d["citation_count"],
                    "redundancy_penalty": d["redundancy_penalty"],
                    "retrieval_score": d["retrieval_score"],
                    "text": d["text"],
                }
                for d in ranked[:args.top_m]
            ],
            "hyperparams": {
                "alpha": args.alpha,
                "beta": args.beta,
                "gamma": args.gamma,
                "citation_top_n": args.citation_top_n,
                "citation_sim_threshold": args.citation_sim_threshold,
                "top_k": args.top_k,
                "n_probes": args.n_probes,
                "metadata_filter": metadata_filter,
            },
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved results to: {args.save}")

    # === Final Answer Stage (default: run; use --no-final to skip) ===
    if not args.no_final:
        if not ranked:
            raise SystemExit("No ranked documents available for final answer.")
        print("\n=== WRAG • Final Answer Generation ===")
        answer_text = generate_final_answer(
            question=question,
            weighted_docs=ranked[:args.top_m],  # pass top-M
            model_name=args.ans_model,
            top_m_docs=args.top_m,
            doc_char_lim=args.ans_doc_char_limit,
            temperature=args.ans_temp,
            max_output_tokens=args.ans_max_tokens
        )
        print("\nModel output (expected JSON):")
        print(answer_text)

        if args.save_answer:
            os.makedirs(os.path.dirname(args.save_answer) or ".", exist_ok=True)
            with open(args.save_answer, "w", encoding="utf-8") as f:
                f.write(answer_text if isinstance(answer_text, str) else json.dumps(answer_text))
            print(f"\nSaved final answer to: {args.save_answer}")

if __name__ == "__main__":
    main()
