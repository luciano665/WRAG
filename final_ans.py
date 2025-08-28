# final_ans.py
import os, re, json, time, random
from typing import Dict, Any, List, Tuple, Optional

import requests
from dotenv import load_dotenv, find_dotenv

# ── Env loading (try secrets.env first, then .env) ────────────────────────────────
load_dotenv(find_dotenv(filename="secrets.env", usecwd=True) or find_dotenv(filename=".env", usecwd=True) or find_dotenv(), override=False)

from probe_stage import run_probe_stage
from reweight_stage import compute_doc_weights

# ── OpenRouter config ────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-r1-distill-llama-8b")
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "WRAG Final Answer")

# ── Answer formatting config ─────────────────────────────────────────────────────
TOP_M_DOCS   = 5       # how many reweighted docs to pass
DOC_CHAR_LIM = 880     # trim per-doc to keep prompt small (reduced from 1200)
MAX_TOKENS   = 512     # more room for clean JSON (raised from 256)
TEMP         = 0.2     # low temp for factuality

# ── Helpers ─────────────────────────────────────────────────────────────────────
def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n]

def _format_context(weighted_docs: List[Dict[str, Any]], top_m: int = TOP_M_DOCS) -> Tuple[str, List[str]]:
    lines, ids = [], []
    for i, d in enumerate(weighted_docs[:top_m], 1):
        ids.append(d["id"])
        lines.append(f"[Doc {i} | id={d['id']} | w={d['final_weight']:.3f}] {_truncate(d['text'], DOC_CHAR_LIM)}")
    return "\n\n".join(lines), ids

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
        # Expose the 429 for caller backoff
        raise requests.HTTPError("429", response=r)
    r.raise_for_status()
    return r.json()

def _extract_text_from_or(resp_json: Dict[str, Any]) -> str:
    try:
        msg = resp_json["choices"][0]["message"]["content"]
    except Exception:
        return ""
    if isinstance(msg, str):
        return msg
    if isinstance(msg, list):
        parts = []
        for part in msg:
            t = part.get("text") or part.get("content") or ""
            if isinstance(t, str):
                parts.append(t)
        return "\n".join(parts)
    return ""

def _intersect_citations(cites: Any, allowed_ids: List[str]) -> List[str]:
    allowed = set(str(x) for x in allowed_ids)
    out: List[str] = []
    if isinstance(cites, list):
        for c in cites:
            sc = str(c)
            if sc in allowed:
                out.append(sc)
    return out

def _force_json_object(raw: str, fallback_answer: str, allowed_ids: List[str]) -> Dict[str, Any]:
    """
    Try hard to coerce RAW model text into a JSON object with fields:
    - answer: str
    - citations: list[str] (subset of allowed_ids)
    """
    s = raw or ""
    # Pre-clean code fences before regexing for {...}
    s = re.sub(r"```(?:json)?\s*", "", s)
    s = s.replace("```", "")

    # Extract the largest {...} block if present
    if "{" in s and "}" in s:
        try:
            start = s.index("{")
            end = s.rindex("}")
            candidate = s[start:end+1]
            obj = json.loads(candidate)
        except Exception:
            obj = None
    else:
        obj = None

    if not isinstance(obj, dict):
        # Fallback minimal object
        obj = {"answer": fallback_answer.strip(), "citations": []}

    ans = obj.get("answer")
    if not isinstance(ans, str) or not ans.strip():
        ans = fallback_answer.strip()

    cites = _intersect_citations(obj.get("citations"), allowed_ids)

    return {"answer": ans.strip(), "citations": cites}

# ── Core generation ──────────────────────────────────────────────────────────────
def generate_final_answer(question: str, weighted_docs: List[Dict[str, Any]], model_name: Optional[str] = None) -> Dict[str, Any]:
    model_name = model_name or OPENROUTER_MODEL
    context_str, doc_ids = _format_context(weighted_docs, TOP_M_DOCS)

    system_msg = (
        "You are a careful QA assistant. Use ONLY the provided passages.\n"
        "Answer the question concisely and cite the supporting Doc IDs.\n"
        "Return JSON with fields: answer (string) and citations (array of doc IDs)."
    )
    user_msg = (
        f"Question: {question}\n\n"
        f"Passages:\n{context_str}\n\n"
        "Rules:\n"
        " - Do not use outside knowledge.\n"
        " - If insufficient evidence, say so.\n"
        " - Citations must be a subset of the given Doc IDs.\n"
        "JSON only:"
    )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        "temperature": TEMP,
        "max_tokens": MAX_TOKENS,
        # Force JSON and stop at a stray code-fence if the model tries to add one
        "response_format": {"type": "json_object"},
        "stop": ["```"],
    }

    attempts = 5
    last_text = ""
    for attempt in range(attempts):
        try:
            resp = _or_post(payload)
            text = _extract_text_from_or(resp) or ""
            last_text = text
            result = _force_json_object(text, fallback_answer=text, allowed_ids=doc_ids)

            # Ensure citations are subset; if empty but answer looks confident, you can optionally attach top-1 doc
            if not result["citations"] and doc_ids:
                # Heuristic: attach the highest-weight doc if the model gave a non-empty answer
                if result["answer"]:
                    result["citations"] = [doc_ids[0]]

            return result

        except requests.HTTPError as e:
            code = getattr(e.response, "status_code", None)
            if code == 429:
                wait = min(32, (2 ** attempt) + random.random())
                print(f"[OpenRouter 429] Backing off {wait:.1f}s (attempt {attempt+1}/{attempts})")
                time.sleep(wait)
                continue
            if code and 500 <= code < 600:
                wait = min(16, 2 ** attempt)
                print(f"[OpenRouter {code}] Retrying in {wait:.1f}s (attempt {attempt+1}/{attempts})")
                time.sleep(wait)
                continue
            raise
        except (requests.Timeout, requests.ConnectionError):
            wait = min(16, 2 ** attempt)
            print(f"[OpenRouter net] Retrying in {wait:.1f}s (attempt {attempt+1}/{attempts})")
            time.sleep(wait)
            continue

    # Final fallback if we never parsed JSON
    return {"answer": (last_text or "").strip(), "citations": [doc_ids[0]] if doc_ids else []}

# ── CLI demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example end-to-end run
    q = "Where did the first known cases of MERS occur?"
    stage = run_probe_stage(q)
    ranked = compute_doc_weights(stage["pinecone_results"], stage["probes"], stage["mean_probe_similarity"])
    result = generate_final_answer(q, ranked)

    out = {
        "question": q,
        "probes": stage["probes"],
        "top_ids": [m.get("id") for m in stage["pinecone_results"].get("matches", [])],
        "answer": result,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
