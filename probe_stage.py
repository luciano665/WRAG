# probe_stage.py
import os, re, time, random
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter

import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv, find_dotenv

# ── Env loading (secrets.env first, then .env) ────────────────────────────────────
load_dotenv(find_dotenv(filename="secrets.env", usecwd=True) or find_dotenv(filename=".env", usecwd=True), override=False)

# Pinecone helpers from your repo
from Quering_RAG.client_embedder import init_pinecone, embed
from Quering_RAG.config_pinecone import NAMESPACE

# ====== CONFIG ======
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-r1-distill-llama-8b")

# Optional (nice-to-have metadata for OpenRouter)
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "WRAG Probe Stage")

TOP_K               = 5
N_PROBES            = 5
PER_DOC_CHAR_LIMIT  = 700
TOTAL_CONTEXT_DOCS  = 5   # hard requirement: at least this many distinct context passages (by text) if available

# Fetch-until-unique controls
FETCH_STEP   = 8        # base step; we will auto-broaden if needed
FETCH_MAX    = 100      # base max; we will auto-broaden if needed
DEDUPE_BY    = "text"   # IMPORTANT: ensure uniqueness by exact text (not by chunk identity)

# Debug: set PROBE_DEBUG=1 to log raw LLM output and internals
PROBE_DEBUG  = os.getenv("PROBE_DEBUG", "0") == "1"

# ====== MODELS ======
# Embedding model for probe-agreement metric
embed_model_mpnet = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

# ====== HELPERS ======
def _truncate(s: Optional[str], n: int) -> str:
    return s if (s and len(s) <= n) else (s[:n] if s else "")

def _dedupe_matches(matches, by="text"):
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
        else:
            key = (meta.get("text") or meta.get("chunk") or meta.get("content") or "").strip().lower()
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
        filter=metadata_filter
    )
    return resp.to_dict() if hasattr(resp, "to_dict") else resp

def fetch_until_unique(index, qvec, namespace, want_k=5, step=8, max_hits=100, dedupe_by="text",
                       metadata_filter=None):
    """
    Pull progressively larger result sets until we have `want_k` unique matches
    (or we hit `max_hits`). Returns a Pinecone-like dict with only the unique slice.
    """
    gathered = []
    fetched = 0
    last_resp = {"matches": []}
    while len(gathered) < want_k and fetched < max_hits:
        batch_k = min(step, max_hits - fetched)
        resp = pinecone_search(
            index, qvec,
            top_k=fetched + batch_k,
            namespace=namespace,
            metadata_filter=metadata_filter
        )
        last_resp = resp
        all_matches = resp.get("matches", []) or []
        fetched = len(all_matches)
        deduped = _dedupe_matches(all_matches, by=dedupe_by)
        gathered = deduped[:want_k]
        if fetched >= max_hits:
            break
    out = dict(last_resp)
    out["matches"] = gathered
    return out

# ── Robust probe parsing ───────────────────────────────────────────────────────────
_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-*•]+|\d+[\).\:]|\(\d+\))\s*")
_CODE_FENCE_RE    = re.compile(r"```.*?```", flags=re.DOTALL | re.IGNORECASE)
_TAGS_RE          = re.compile(r"<[^>]+>")

def _clean_probe_block(text: str) -> str:
    # strip code fences and HTML-ish tags
    text = _CODE_FENCE_RE.sub("", text or "")
    text = _TAGS_RE.sub("", text)
    return text.strip()

def _split_candidate_lines(text: str) -> List[str]:
    # Prefer newlines; if single line, split on semicolons
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) <= 1:
        alt = []
        for seg in re.split(r"[;\n]+", text):
            seg = seg.strip()
            if seg:
                alt.append(seg)
        if len(alt) > len(lines):
            lines = alt
    return lines

def _normalize_keyword_triplet(s: str) -> Optional[str]:
    """
    Try to extract three keywords from a line.
    Accept separators: commas, semicolons, slashes, pipes.
    Remove bullet/numbering prefixes. Keep short alphanumeric tokens.
    Strip trailing punctuation.
    """
    if not s:
        return None
    s = _BULLET_PREFIX_RE.sub("", s)
    parts = re.split(r"[,\;/\|]+", s)
    toks = []
    for p in parts:
        # split further on whitespace, keep alnum-ish words
        for w in re.split(r"\s+", p.strip()):
            w = re.sub(r"[^A-Za-z0-9\-_/\.]+", "", w)  # keep -, _, /, .
            w = w.rstrip(".,;:")                       # drop trailing punc
            if 2 <= len(w) <= 32:
                toks.append(w)
        if len(toks) >= 3:
            break
    unique = []
    seen = set()
    for t in toks:
        low = t.lower()
        if low in seen:
            continue
        seen.add(low)
        unique.append(t)
        if len(unique) == 3:
            break
    if len(unique) < 3:
        return None
    return ", ".join(unique[:3])

def parse_probe_lines(text: str, n_probes: int) -> List[str]:
    """
    Robustly parse N lines of "a, b, c" triplets from LLM output.
    If the model returned extra prose, bullets, or code fences, this cleans it up.
    """
    if not text:
        return []
    cleaned = _clean_probe_block(text)
    lines = _split_candidate_lines(cleaned)
    out: List[str] = []
    for ln in lines:
        trip = _normalize_keyword_triplet(ln)
        if trip:
            out.append(trip)
        if len(out) == n_probes:
            break
    return out[:n_probes]

# ── Fallback: synthesize probes from retrieved docs ────────────────────────────────
_STOP = set("""
a an the and or but if then when while for to from by with without not no yes on in at of as up down over under
is are was were be been being have has had do does did can could should would may might must will
this that these those here there it its their his her our your my we you they them he she i one two three four five
about into across within between among after before during against through per such which who whom whose what where why how
""".split())

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-\_/\.]{1,31}")

def _extract_keywords(text: str, limit: int = 20) -> List[str]:
    words = [w.lower() for w in _WORD_RE.findall(text or "")]
    words = [w for w in words if w not in _STOP and not w.isdigit()]
    counts = Counter(words)
    # Prefer longer tokens slightly to avoid stop-wordy short junk
    scored = [(w, cnt * (1.0 + min(len(w), 12) / 12.0)) for w, cnt in counts.items()]
    scored.sort(key=lambda x: (-x[1], -len(x[0])))
    return [w for (w, _) in scored[:limit]]

def _synthesize_probes_from_docs(question: str, docs: List[str], n_probes: int) -> List[str]:
    """
    Build reasonable "a, b, c" triplets from top-doc keywords.
    """
    kw_global: List[str] = []
    for d in docs[:TOTAL_CONTEXT_DOCS]:
        kw_global.extend(_extract_keywords(d, limit=15))
    # ensure question terms at the front to keep relevance
    kw_q = _extract_keywords(question, limit=10)
    pool = []
    seen = set()
    for w in kw_q + kw_global:
        if w not in seen:
            seen.add(w)
            pool.append(w)
    # chunk into triples
    triples = []
    i = 0
    while len(triples) < n_probes and i + 2 < len(pool):
        trip = ", ".join([pool[i], pool[i+1], pool[i+2]])
        triples.append(trip)
        i += 3
    # If still short, pad with simple variants on question terms
    pad_i = 0
    while len(triples) < n_probes:
        base = kw_q[pad_i % max(1, len(kw_q))] if kw_q else f"probe{pad_i}"
        triples.append(f"{base}, detail, evidence")
        pad_i += 1
    return triples[:n_probes]

# ── OpenRouter LLM call ────────────────────────────────────────────────────────────
def _or_headers() -> Dict[str, str]:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY missing. Put it in your .env or secrets.env")
    hdrs = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    if OPENROUTER_SITE_URL:
        hdrs["HTTP-Referer"] = OPENROUTER_SITE_URL
    if OPENROUTER_APP_NAME:
        hdrs["X-Title"] = OPENROUTER_APP_NAME
    return hdrs

def _or_post(payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    url = "https://openrouter.ai/api/v1/chat/completions"
    r = requests.post(url, json=payload, headers=_or_headers(), timeout=timeout)
    if r.status_code == 429:
        raise requests.HTTPError("429", response=r)
    r.raise_for_status()
    return r.json()

def _extract_text_from_or(resp_json: Dict[str, Any]) -> str:
    """
    Handle string or structured content arrays.
    """
    try:
        msg = resp_json["choices"][0]["message"]["content"]
    except Exception:
        return ""
    if isinstance(msg, str):
        return msg
    if isinstance(msg, list):
        # Some models return a list of content parts (OpenAI tool format)
        parts = []
        for part in msg:
            t = part.get("text") or part.get("content") or ""
            if isinstance(t, str):
                parts.append(t)
        return "\n".join(parts)
    return ""

def generate_probes_openrouter(question: str, context_docs: Optional[List[str]] = None, n_probes: int = 5,
                               model_name: Optional[str] = None) -> List[str]:
    """
    Generate N keyword probes via OpenRouter with strict formatting + retries/backoff.
    """
    model_name = model_name or OPENROUTER_MODEL

    sys_msg = (
        f"You are generating probe keywords to improve document retrieval.\n"
        f"Output EXACTLY {n_probes} lines.\n"
        f"Each line MUST be THREE keywords separated by commas, e.g.:\n"
        f"alpha-tubulin, nuclear fraction, nucleolus\n"
        f"No numbering, no bullets, no markdown, no extra text.\n"
        f"Prefer including at least one named entity (place or organization) or a year per line if present in the context."
    )

    docs = [(d or "") for d in (context_docs or [])][:TOTAL_CONTEXT_DOCS]
    docs = [_truncate(d, PER_DOC_CHAR_LIMIT) for d in docs]
    context_str = "\n\n".join(f"[Doc {i+1}]: {d}" for i, d in enumerate(docs)) if docs else "(no context)"
    user_msg = (
        f"Context Passages:\n{context_str}\n\n"
        f"Question: {question}\n\n"
        f"Now produce EXACTLY {n_probes} lines, each line = three keywords separated by commas."
    )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": sys_msg},
            {"role": "user",   "content": user_msg},
        ],
        "temperature": 0.7,
        "max_tokens": max(48, n_probes * 16)
    }

    attempts = 6
    for attempt in range(attempts):
        try:
            resp = _or_post(payload)
            text = _extract_text_from_or(resp)
            if PROBE_DEBUG:
                print("\n[probe/debug] RAW LLM OUTPUT:\n" + (text or "<empty>") + "\n")
            probes = parse_probe_lines(text, n_probes)
            if len(probes) < n_probes:
                # Ask to add the missing lines, still keeping it strict
                need = n_probes - len(probes)
                follow_user = (
                    f"You returned {len(probes)} lines. Add {need} MORE lines to reach exactly {n_probes}.\n"
                    f"Each new line MUST be exactly three keywords separated by commas. No markdown or bullets.\n"
                    f"Output ONLY the new lines."
                )
                payload_follow = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": sys_msg},
                        {"role": "user",   "content": user_msg},
                        {"role": "user",   "content": follow_user},
                    ],
                    "temperature": 0.7,
                    "max_tokens": max(32, need * 16)
                }
                resp2 = _or_post(payload_follow)
                text2 = _extract_text_from_or(resp2)
                if PROBE_DEBUG:
                    print("\n[probe/debug] RAW LLM FOLLOW-UP OUTPUT:\n" + (text2 or "<empty>") + "\n")
                probes += parse_probe_lines(text2, need)
                probes = probes[:n_probes]

            if len(probes) == n_probes:
                return probes

            # Not enough even after follow-up; fall through to fallback
            break

        except requests.HTTPError as e:
            code = getattr(e.response, "status_code", None)
            if code == 429:
                wait = min(32, (2 ** attempt) + random.random())
                print(f"[OR 429] Backing off {wait:.1f}s (attempt {attempt+1}/{attempts})")
                time.sleep(wait)
                continue
            # transient 5xx
            if code and 500 <= code < 600:
                wait = min(16, 2 ** attempt)
                print(f"[OR {code}] Retrying in {wait:.1f}s (attempt {attempt+1}/{attempts})")
                time.sleep(wait)
                continue
            raise
        except (requests.Timeout, requests.ConnectionError):
            wait = min(16, 2 ** attempt)
            print(f"[OR net] Retrying in {wait:.1f}s (attempt {attempt+1}/{attempts})")
            time.sleep(wait)
            continue

    # Fallback if the model did not return compliant lines
    return []

def compute_mean_probe_similarity(probes: List[str]) -> Tuple[float, np.ndarray]:
    """
    Returns (mean_offdiag_cosine, full_similarity_matrix).
    If fewer than 2 probes, returns (1.0, NxN ones) as a sentinel.
    """
    if len(probes) < 2:
        n = len(probes)
        return (1.0, np.ones((n, n)) if n else np.zeros((0, 0)))
    vecs = embed_model_mpnet.encode(probes, convert_to_numpy=True)
    sim = cosine_similarity(vecs)
    return float(np.mean(sim[np.triu_indices(len(probes), k=1)])), sim

# ── Utility: enforce at least N distinct context passages (by text) ────────────────
def _extract_unique_texts(matches: List[Dict[str, Any]], limit: int) -> List[str]:
    uniq = []
    seen = set()
    for m in matches or []:
        meta = m.get("metadata", {}) or {}
        txt = meta.get("text") or meta.get("chunk") or meta.get("content") or ""
        key = txt.strip().lower()
        if not txt or key in seen:
            continue
        seen.add(key)
        uniq.append(txt)
        if len(uniq) == limit:
            break
    return uniq

# ====== PUBLIC ENTRYPOINT ======
def run_probe_stage(
    question: str,
    top_k: int = TOP_K,
    n_probes: int = N_PROBES,
    dedupe_by: str = DEDUPE_BY,
    fetch_step: int = FETCH_STEP,
    fetch_max: int = FETCH_MAX,
    metadata_filter: Optional[dict] = None
) -> Dict[str, Any]:
    """
    Retrieves up to K UNIQUE docs (by text), generates keyword probes (OpenRouter),
    and returns probes + agreement metrics + the unique Pinecone results.
    Also enforces at least TOTAL_CONTEXT_DOCS distinct context passages if available.
    """
    # Init Pinecone + embed the query
    index = init_pinecone()
    qvec  = embed([question])[0]

    # Ensure up to K unique matches (by text)
    pc_res = fetch_until_unique(
        index=index,
        qvec=qvec,
        namespace=NAMESPACE,
        want_k=top_k,
        step=fetch_step,
        max_hits=fetch_max,
        dedupe_by=dedupe_by,
        metadata_filter=metadata_filter
    )

    matches = pc_res.get("matches", []) or []
    top_docs = _extract_unique_texts(matches, limit=TOTAL_CONTEXT_DOCS)

    # If we didn't get enough distinct context passages, broaden and refetch (up to two more rounds)
    attempts = 0
    cur_step, cur_max = fetch_step, fetch_max
    while len(top_docs) < min(TOTAL_CONTEXT_DOCS, top_k) and attempts < 2:
        attempts += 1
        cur_step = max(cur_step * 2, 16)
        cur_max  = max(cur_max * 2, 200)
        if PROBE_DEBUG:
            print(f"[probe/debug] Only {len(top_docs)} distinct contexts; broadening search (step={cur_step}, max={cur_max})")
        pc_res = fetch_until_unique(
            index=index,
            qvec=qvec,
            namespace=NAMESPACE,
            want_k=top_k,
            step=cur_step,
            max_hits=cur_max,
            dedupe_by="text",
            metadata_filter=metadata_filter
        )
        matches = pc_res.get("matches", []) or []
        top_docs = _extract_unique_texts(matches, limit=TOTAL_CONTEXT_DOCS)

    # 1) Try LLM probes
    probes = generate_probes_openrouter(question, context_docs=top_docs, n_probes=n_probes)

    # 2) If still short, synthesize from docs
    if len(probes) < n_probes:
        if PROBE_DEBUG:
            print(f"[probe/debug] LLM returned {len(probes)} valid lines; synthesizing {n_probes - len(probes)} fallback lines.")
        synth = _synthesize_probes_from_docs(question, top_docs, n_probes)
        # If LLM produced some valid lines, keep them and fill the rest from synth (dedup across lines)
        if probes:
            existing = set([p.lower() for p in probes])
            for s in synth:
                if s.lower() not in existing:
                    probes.append(s)
                if len(probes) == n_probes:
                    break
        else:
            probes = synth[:n_probes]

    mean_sim, sim_matrix = compute_mean_probe_similarity(probes)

    return {
        "question": question,
        "probes": probes,
        "mean_probe_similarity": mean_sim,
        "probe_similarity_matrix": sim_matrix,
        "pinecone_results": pc_res,  # matches are unique by text slice and reflect broadened fetch if used
        "top_docs": top_docs         # <= TOTAL_CONTEXT_DOCS unique passages by exact text
    }

# ====== CLI Demo ======
if __name__ == "__main__":
    out = run_probe_stage("Where did the first known cases of MERS occur?")
    print("Probes:")
    for p in out["probes"]:
        print("-", p)
    print("Mean probe similarity:", round(out["mean_probe_similarity"], 3))
    ids = [m.get("id") for m in out["pinecone_results"].get("matches", [])]
    print("Retrieved IDs:", ids)
    print(f"Distinct context passages: {len(out['top_docs'])}")
