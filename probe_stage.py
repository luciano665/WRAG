# probe_stage.py
import os, time, random
import numpy as np
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv, find_dotenv

# Load env (try secrets.env first, then .env)
load_dotenv(find_dotenv(filename="secrets.env", usecwd=True) or find_dotenv(), override=False)

from Quering_RAG.client_embedder import init_pinecone, embed
from Quering_RAG.config_pinecone import NAMESPACE

# ====== CONFIG ======
GEMINI_KEY   = os.getenv("GEMINI_API_KEY", "realkey")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
TOP_K        = 5
N_PROBES     = 5
PER_DOC_CHAR_LIMIT  = 700
TOTAL_CONTEXT_DOCS  = 5

# Fetch-until-unique controls
FETCH_STEP   = 8      # how many additional hits to request per fetch iteration
FETCH_MAX    = 100    # hard cap on total hits considered
DEDUPE_BY    = "chunk"  # "chunk" -> (config, doc_index, chunk_index) | "text" -> exact text match

# ====== MODELS ======
embed_model_mpnet = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
genai.configure(api_key=GEMINI_KEY)

# ====== HELPERS ======
def _truncate(s: str, n: int) -> str:
    return s if s is not None and len(s) <= n else (s[:n] if s else "")

def _dedupe_matches(matches, by="chunk"):
    """
    Deduplicate Pinecone matches by doc identity.
    by="chunk": uses (config, doc_index, chunk_index)
    by="text" : uses exact lowercased text
    """
    seen, out = set(), []
    for m in matches or []:
        meta = m.get("metadata", {}) or {}
        if by == "chunk":
            key = (meta.get("config"), meta.get("doc_index"), meta.get("chunk_index"))
        else:  # by text
            key = (meta.get("text") or "").strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
    return out

def pinecone_search(index, query_vector, top_k=5, namespace=None, metadata_filter=None):
    """
    Query Pinecone with optional metadata filter.
    """
    resp = index.query(
        vector=query_vector.tolist(),
        top_k=top_k,
        include_metadata=True,
        namespace=namespace,
        filter=metadata_filter  # <-- new
    )
    return resp.to_dict() if hasattr(resp, "to_dict") else resp

def fetch_until_unique(index, qvec, namespace, want_k=5, step=8, max_hits=100, dedupe_by="chunk",
                       metadata_filter=None):
    """
    Pull results in batches until we collect 'want_k' unique docs (or hit 'max_hits').
    Returns a Pinecone-like dict with 'matches' replaced by the unique slice (<= want_k).
    """
    gathered = []
    fetched = 0
    last_resp = {"matches": []}
    while len(gathered) < want_k and fetched < max_hits:
        batch_k = min(step, max_hits - fetched)
        # Always query for (already_fetched + batch_k) to expand the pool
        resp = pinecone_search(
            index, qvec,
            top_k=fetched + batch_k,
            namespace=namespace,
            metadata_filter=metadata_filter  # <-- new
        )
        last_resp = resp  # keep the latest full response for non-match fields
        all_matches = resp.get("matches", []) or []
        fetched = len(all_matches)
        # Deduplicate across the enlarged pool, then keep the first want_k uniques
        deduped = _dedupe_matches(all_matches, by=dedupe_by)
        gathered = deduped[:want_k]
        if fetched >= max_hits:
            break

    out = dict(last_resp)
    out["matches"] = gathered
    return out

def parse_probe_lines(text: str, n_probes: int):
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    out = []
    for ln in lines:
        parts = [p.strip() for p in ln.split(",") if p.strip()]
        out.append(", ".join(parts[:3]) if parts else ln)
        if len(out) == n_probes:
            break
    return out

def generate_probes_gemini(question, context_docs=None, n_probes=5, model_name=GEMINI_MODEL):
    """
    Generate N keyword probes in ONE call (with lightweight backoff on 429).
    """
    sys_msg = (
        f"Given the following context passages, return EXACTLY {n_probes} lines. "
        "Each line must contain THREE comma-separated keywords for the answer. "
        "Make each line distinct. Output ONLY the lines."
    )
    docs = [(d or "") for d in (context_docs or [])][:TOTAL_CONTEXT_DOCS]
    docs = [_truncate(d, PER_DOC_CHAR_LIMIT) for d in docs]
    context_str = "\n\n".join(f"[Doc {i+1}]: {d}" for i, d in enumerate(docs)) if docs else "(no context)"
    prompt = f"{sys_msg}\n\nContext Passages:\n{context_str}\n\nQuestion: {question}"

    model = genai.GenerativeModel(model_name)
    max_tokens = max(24, n_probes * 12)

    for attempt in range(6):
        try:
            resp = model.generate_content(
                prompt,
                generation_config={"temperature": 0.7, "max_output_tokens": max_tokens}
            )
            probes = parse_probe_lines(resp.text, n_probes)
            if len(probes) < n_probes:
                need = n_probes - len(probes)
                follow = f"Add {need} MORE lines, each being THREE comma-separated keywords. No numbering."
                resp2 = model.generate_content(
                    prompt + "\n\n" + follow,
                    generation_config={"temperature": 0.7, "max_output_tokens": max(24, need * 12)}
                )
                probes += parse_probe_lines(resp2.text, need)
                probes = probes[:n_probes]
            return probes
        except ResourceExhausted:
            wait = min(32, (2 ** attempt) + random.random())
            print(f"[Gemini 429] Backing off {wait:.1f}s (attempt {attempt+1}/6)")
            time.sleep(wait)
    raise RuntimeError("Gemini rate limit/quota reached repeatedly.")

def compute_mean_probe_similarity(probes):
    """
    Returns (mean_offdiag_cosine, full_similarity_matrix)
    """
    if len(probes) < 2:
        return 1.0, np.ones((len(probes), len(probes))) if probes else np.zeros((0, 0))
    vecs = embed_model_mpnet.encode(probes, convert_to_numpy=True)
    sim = cosine_similarity(vecs)
    return float(np.mean(sim[np.triu_indices(len(probes), k=1)])), sim

# ====== PUBLIC ENTRYPOINT ======
def run_probe_stage(
    question: str,
    top_k: int = TOP_K,
    n_probes: int = N_PROBES,
    dedupe_by: str = DEDUPE_BY,
    fetch_step: int = FETCH_STEP,
    fetch_max: int = FETCH_MAX,
    metadata_filter: dict | None = None  # <-- new
):
    """
    Retrieves up to K UNIQUE docs, generates keyword probes using those docs as context,
    and returns probes + agreement metrics + the unique Pinecone results.
    """
    index = init_pinecone()
    qvec  = embed([question])[0]

    # Guarantee up to K unique docs for context & downstream stages
    pc_res = fetch_until_unique(
        index=index,
        qvec=qvec,
        namespace=NAMESPACE,
        want_k=top_k,
        step=fetch_step,
        max_hits=fetch_max,
        dedupe_by=dedupe_by,
        metadata_filter=metadata_filter  # <-- new
    )

    matches = pc_res.get("matches", []) or []
    top_docs = []
    for m in matches:
        meta = m.get("metadata", {}) or {}
        txt  = meta.get("text") or meta.get("chunk") or meta.get("content") or ""
        if txt:
            top_docs.append(txt)

    probes = generate_probes_gemini(question, context_docs=top_docs, n_probes=n_probes)
    mean_sim, sim_matrix = compute_mean_probe_similarity(probes)

    return {
        "question": question,
        "probes": probes,
        "mean_probe_similarity": mean_sim,
        "probe_similarity_matrix": sim_matrix,
        "pinecone_results": pc_res,  # matches are unique here
        "top_docs": top_docs         # len(top_docs) <= top_k, all unique by chosen key
    }

# ====== CLI Demo ======
if __name__ == "__main__":
    out = run_probe_stage("what is the yearly amortization rate related to the trademarks?")
    print("Probes:", out["probes"])
    print("Mean probe similarity:", round(out["mean_probe_similarity"], 2))
