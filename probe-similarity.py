# probe-similarity.py
import os, time, random
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---- Pinecone utils ----
from Quering_RAG.client_embedder import init_pinecone, embed
from Quering_RAG.config_pinecone import NAMESPACE  # ensure this exists in config
# If your Quering_RAG.search_top_k.search lacks a namespace param, we’ll just query inline.
# from Quering_RAG.search_top_k import search

# ---- CONFIG ----
GEMINI_KEY =  "AIzaSyBjGsKCyQnvjh4pVFGUskfGe9e5DHmImGY"  # set GEMINI_API_KEY in your env
GEMINI_MODEL = "gemini-1.5-flash"  # cheaper than 1.5-pro
TOP_K = 5
N_PROBES = 5
PER_DOC_CHAR_LIMIT = 700   # trim long passages
TOTAL_CONTEXT_DOCS = 5     # cap number of docs in prompt

# ---- Setup ----
embed_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
genai.configure(api_key=GEMINI_KEY)

# ---- Helpers ----
def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n]

def pinecone_search(index, query_vector, top_k=5, namespace=None):
    # Query Pinecone v3 directly to ensure namespace is honored
    resp = index.query(
        vector=query_vector.tolist(),
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )
    return resp.to_dict() if hasattr(resp, "to_dict") else resp

def parse_probe_lines(text: str, n_probes: int):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out = []
    for ln in lines:
        parts = [p.strip() for p in ln.split(",") if p.strip()]
        if len(parts) >= 3:
            out.append(", ".join(parts[:3]))
        else:
            # fallback: keep line as-is
            out.append(ln)
        if len(out) == n_probes:
            break
    return out

# ---- Generate N probes in ONE call (+ backoff) ----
def generate_probes_gemini(question, context_docs=None, n_probes=5, model_name=GEMINI_MODEL):
    sys_msg = (
        "Given the following context passages, return EXACTLY {n} lines. "
        "Each line must contain THREE comma-separated keywords for the answer to the question. "
        "Make each line distinct. Output ONLY the lines."
    ).format(n=n_probes)

    context_docs = context_docs or []
    # Trim and cap context
    docs = [_truncate(d or "", PER_DOC_CHAR_LIMIT) for d in context_docs[:TOTAL_CONTEXT_DOCS]]
    context_str = "\n\n".join(f"[Doc {i+1}]: {d}" for i, d in enumerate(docs)) if docs else "(no context)"

    prompt = f"{sys_msg}\n\nContext Passages:\n{context_str}\n\nQuestion: {question}"

    model = genai.GenerativeModel(model_name)
    max_tokens = max(24, n_probes * 12)  # small, since we expect n short lines

    # Simple exponential backoff for transient 429s
    for attempt in range(6):
        try:
            resp = model.generate_content(
                prompt,
                generation_config={"temperature": 0.7, "max_output_tokens": max_tokens}
            )
            probes = parse_probe_lines(resp.text or "", n_probes)
            # If model returned fewer than n_probes lines, top up once with a follow-up prompt
            if len(probes) < n_probes:
                need = n_probes - len(probes)
                follow = (
                    f"Add {need} MORE lines, each being THREE comma-separated keywords. "
                    "No numbering. No extra text."
                )
                resp2 = model.generate_content(
                    prompt + "\n\n" + follow,
                    generation_config={"temperature": 0.7, "max_output_tokens": max(24, need * 12)}
                )
                probes += parse_probe_lines(resp2.text or "", need)
                probes = probes[:n_probes]
            return probes
        except ResourceExhausted as e:
            # If you’ve hit daily quota, backoff won’t help—switch models or upgrade
            wait = min(32, (2 ** attempt) + random.random())
            print(f"[Gemini 429] Backing off {wait:.1f}s (attempt {attempt+1}/6)")
            time.sleep(wait)
    raise RuntimeError("Gemini rate limit/quota reached repeatedly. Try a cheaper model or fewer probes.")

def embed_probes(probes, embed_model):
    return embed_model.encode(probes, convert_to_numpy=True)

def probe_similarity_matrix(probe_vecs):
    return cosine_similarity(probe_vecs)

# ---- Main ----
if __name__ == "__main__":
    question = "what is the yearly amortization rate related to the trademarks?"

    # Retrieve top-k docs
    index = init_pinecone()
    query_vec = embed([question])[0]
    results = pinecone_search(index, query_vec, top_k=TOP_K, namespace=NAMESPACE)
    matches = results.get("matches", []) or []
    top_k_docs = []
    for m in matches:
        meta = m.get("metadata", {}) or {}
        txt = meta.get("text") or meta.get("chunk") or meta.get("content") or ""
        if txt:
            top_k_docs.append(txt)

    # Generate N probes in ONE call (less quota)
    probes = generate_probes_gemini(question, context_docs=top_k_docs, n_probes=N_PROBES)
    print("Generated probes:", probes)

    # Embed probes and compute similarity
    probe_vecs = embed_probes(probes, embed_model)
    sim_matrix = probe_similarity_matrix(probe_vecs)
    print("Probe similarity matrix (rounded):")
    print(np.round(sim_matrix, 2))

    mean_sim = np.mean(sim_matrix[np.triu_indices(len(probes), k=1)]) if len(probes) > 1 else 1.0
    print(f"Mean probe similarity: {mean_sim:.2f}")
