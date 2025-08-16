#  final_ans.py
import os, time, random
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False)



from probe_stage import run_probe_stage
from reweight_stage import compute_doc_weights

GEMINI_KEY   = os.getenv("GEMINI_API_KEY", "realkey")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
TOP_M_DOCS   = 5       # how many reweighted docs to pass
DOC_CHAR_LIM = 1200    # trim per-doc to keep prompt small

genai.configure(api_key=GEMINI_KEY)

def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n]

def _format_context(weighted_docs, top_m=TOP_M_DOCS):
    lines = []
    for i, d in enumerate(weighted_docs[:top_m], 1):
        lines.append(f"[Doc {i} | id={d['id']} | w={d['final_weight']:.3f}] { _truncate(d['text'], DOC_CHAR_LIM) }")
    return "\n\n".join(lines), [d["id"] for d in weighted_docs[:top_m]]

def generate_final_answer(question, weighted_docs, model_name=GEMINI_MODEL):
    context_str, doc_ids = _format_context(weighted_docs, TOP_M_DOCS)
    prompt = (
        "You are a careful QA assistant. Use ONLY the provided passages.\n"
        "Answer the question concisely and cite the supporting Doc IDs.\n"
        "Return JSON with fields: answer (string) and citations (array of doc IDs).\n\n"
        f"Question: {question}\n\n"
        f"Passages:\n{context_str}\n\n"
        "Rules:\n"
        " - Do not use outside knowledge.\n"
        " - If insufficient evidence, say so.\n"
        " - Citations must be a subset of the given Doc IDs.\n"
        "JSON:"
    )

    model = genai.GenerativeModel(model_name)

    for attempt in range(6):
        try:
            resp = model.generate_content(
                prompt,
                generation_config={"temperature": 0.2, "max_output_tokens": 256}
            )
            return resp.text
        except ResourceExhausted:
            wait = min(32, (2 ** attempt) + random.random())
            print(f"[Gemini 429] Backing off {wait:.1f}s (attempt {attempt+1}/6)")
            time.sleep(wait)
    raise RuntimeError("Gemini rate limit/quota reached repeatedly.")

if __name__ == "__main__":
    question = "what is the yearly amortization rate related to the trademarks?"
    stage = run_probe_stage(question)
    ranked = compute_doc_weights(stage["pinecone_results"], stage["probes"], stage["mean_probe_similarity"])
    answer_json = generate_final_answer(question, ranked)
    print(answer_json)
