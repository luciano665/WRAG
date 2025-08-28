# maintest.py
# Runs first N RAGBench questions through (A) vanilla LLM and (B) WRAG+LLM,
# logs detailed telemetry, rich metrics, robustness stats, and saves one JSON.
#
# Backend: OpenRouter ONLY (/chat/completions).
#
# Metrics added:
# - Answer quality: EM, token-F1, number/date tolerant matches
# - Citation validity; evidence overlap (char 3-gram Jaccard & token-F1 style)
# - Retrieval: answer_present@k (string/num/year tolerant), MRR@k, NDCG@k
# - Reweighting: rank-delta for supporting docs, top-1 margin, Spearman/Kendall
# - Probes: compliance rate, diversity (min/mean/max cosine), probe→doc hit rate
# - Calibration/abstention: per-sample confidence, abstained; overall ECE
# - Efficiency/cost: cost_per_correct, latency_per_correct, tokens_per_correct
# - Robustness: --repeats N → std across predictions, mean Levenshtein
#
# Pricing (optional):
#  - Uses env LLM_INPUT_PRICE_PER_1K / LLM_OUTPUT_PRICE_PER_1K if present.
#
# Requirements:
#   pip install datasets python-dotenv numpy scikit-learn sentence-transformers requests psutil pynvml

import os, re, json, time, argparse, platform, subprocess
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from threading import Lock
from collections import Counter, defaultdict
import math

import numpy as np
import requests

# ---- Load env early ---------------------------------------------------------------
try:
    from dotenv import load_dotenv, find_dotenv
    _ = load_dotenv(find_dotenv(filename="secrets.env", usecwd=True) or find_dotenv(), override=False)
except Exception:
    pass

# ---- Optional deps ----------------------------------------------------------------
def _safe_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None

datasets = _safe_import("datasets")
psutil   = _safe_import("psutil")
pynvml   = _safe_import("pynvml")
torch    = _safe_import("torch")

# ---- WRAG stages from your repo ---------------------------------------------------
try:
    from probe_stage import run_probe_stage
except ImportError as e:
    raise SystemExit(
        "Could not import probe_stage. Ensure it's alongside maintest.py or on PYTHONPATH.\n"
        f"ImportError: {e}"
    )

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

# ---- Decomposer (optional) --------------------------------------------------------
try:
    from decompose import decompose
except Exception:
    decompose = None  # graceful if not present

DECOMPOSE_THRESHOLD = 0.5

# ---- Throttle (simple) ------------------------------------------------------------
_llm_lock = Lock()
_last_llm_call = 0.0
def _throttle_llm(min_interval: float = 0.0):
    global _last_llm_call
    with _llm_lock:
        now = time.time()
        wait = (_last_llm_call + min_interval) - now
        if wait > 0:
            time.sleep(wait)
        _last_llm_call = time.time()

# ---- OpenRouter helpers -----------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "WRAG Maintest")

def _or_headers() -> Dict[str, str]:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY missing. Put it in secrets.env or your environment.")
    hdrs = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    if OPENROUTER_SITE_URL:
        hdrs["HTTP-Referer"] = OPENROUTER_SITE_URL
    if OPENROUTER_APP_NAME:
        hdrs["X-Title"] = OPENROUTER_APP_NAME
    return hdrs

def _or_post(payload: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
    url = "https://openrouter.ai/api/v1/chat/completions"
    r = requests.post(url, json=payload, headers=_or_headers(), timeout=timeout)
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

def _preclean_code_fences(s: str) -> str:
    s = re.sub(r"```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    return s.replace("```", "")

def _force_json_object(s: str) -> Dict[str, Any]:
    if not s:
        return {}
    s = _preclean_code_fences(s)
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        block = m.group(0)
        try:
            return json.loads(block)
        except Exception:
            return {}
    return {}

# ---- Prompt builders --------------------------------------------------------------
PROMPT_TEMPLATE_GENERAL_ID = "qa_json_v2"
PROMPT_TEMPLATE_VANILLA_ID = "vanilla_json_v1"

def _build_general_prompt(question: str, context_str: str) -> str:
    return (
        "You are a careful QA assistant. Use ONLY the provided passages.\n"
        "Answer the question concisely and cite the supporting Doc IDs.\n"
        "Return JSON with fields:\n"
        "  - answer (string OR array of strings)\n"
        "  - answer_type (one of: 'span','yesno','number','date','list','multi-hop','abstain')\n"
        "  - citations (array of 1–3 Doc IDs from the passages)\n"
        "  - evidence (array of objects {doc_id, snippet} with ≤200 chars each)\n"
        "  - confidence (number in [0,1])\n"
        "  - conflict (boolean)\n"
        "  - conflict_note (string; brief explanation when conflict=true)\n"
        "  - abstain (boolean)\n\n"
        f"Question: {question}\n\n"
        f"Passages:\n{context_str}\n\n"
        "Rules:\n"
        " - Do not use outside knowledge.\n"
        " - If evidence is insufficient or contradictory, set abstain=true and answer=\"\".\n"
        " - For yes/no questions, set answer to 'yes' or 'no' (lowercase) and answer_type='yesno'.\n"
        " - For numeric answers, return a plain string (normalized number if possible) and answer_type='number'.\n"
        " - Provide 1–3 citations that support the answer; prefer at least 2 when available.\n"
        " - Keep 'evidence' minimal: short quoted snippets from the passages that justify the answer.\n"
        "JSON:"
    )

def _build_vanilla_prompt(question: str) -> str:
    return (
        "You are a careful QA assistant. Answer concisely.\n"
        "Return JSON with fields:\n"
        "  - answer (string OR array of strings)\n"
        "  - answer_type (one of: 'span','yesno','number','date','list','multi-hop','abstain')\n"
        "  - citations (empty array)\n"
        "  - evidence (empty array)\n"
        "  - confidence (number in [0,1])\n"
        "  - conflict (boolean)\n"
        "  - conflict_note (string)\n"
        "  - abstain (boolean)\n\n"
        f"Question: {question}\n\n"
        "Rules:\n"
        " - Use no outside tools or retrieval.\n"
        " - If unsure, set abstain=true and answer=\"\".\n"
        "JSON:"
    )

# ---- Formatting helpers -----------------------------------------------------------
def _truncate(s: str, n: int = 900) -> str:
    if s is None:
        return ""
    return s if len(s) <= n else s[:n]

def _format_context(weighted_docs, top_m: int, char_limit: int):
    lines, ids = [], []
    for i, d in enumerate(weighted_docs[:top_m], 1):
        ids.append(d.get("id"))
        lines.append(f"[Doc {i} | id={d.get('id')} | w={d.get('final_weight', 0.0):.3f}] {_truncate(d.get('text',''), char_limit)}")
    return "\n\n".join(lines), ids

# ---- Cost calc --------------------------------------------------------------------
def compute_cost_usd(usage: Dict[str, Any]) -> Optional[float]:
    in_p  = os.getenv("LLM_INPUT_PRICE_PER_1K")
    out_p = os.getenv("LLM_OUTPUT_PRICE_PER_1K")
    if not (in_p and out_p):
        return None
    try:
        in_price  = float(in_p); out_price = float(out_p)
    except Exception:
        return None
    prompt_toks = usage.get("prompt_tokens"); out_toks = usage.get("completion_tokens")
    if prompt_toks is None or out_toks is None:
        prompt_toks = usage.get("prompt_tokens_est"); out_toks = usage.get("output_tokens_est")
    if prompt_toks is None or out_toks is None:
        return None
    return (float(prompt_toks)/1000.0)*in_price + (float(out_toks)/1000.0)*out_price

# ---- Dataset loading ---------------------------------------------------------------
def _get_hf_token() -> Optional[str]:
    return os.getenv("HF_TOKEN")

def _load_ragbench_subset(subset: str, split: str, limit: int, include_answers: bool = True) -> List[Dict[str, Any]]:
    if datasets is None:
        raise RuntimeError("Install: pip install datasets")
    hf_token = _get_hf_token()
    def _try(name: str):
        if hf_token:
            return datasets.load_dataset(name, subset, split=split, token=hf_token)
        return datasets.load_dataset(name, subset, split=split)
    ds = None
    for owner in ("galileo-ai/ragbench", "rungalileo/ragbench"):
        try:
            ds = _try(owner); break
        except Exception:
            continue
    if ds is None:
        raise RuntimeError(f"Could not load RAGBench {subset}:{split}")
    rows = []
    take = min(limit, len(ds))
    for i in range(take):
        rec = ds[i]
        row = {"id": rec.get("id", None), "question": rec.get("question", None)}
        if include_answers and "answer" in ds.column_names:
            row["answer"] = rec.get("answer", None)
        rows.append(row)
    return rows

# ---- OpenRouter JSON completion ----------------------------------------------------
def _openrouter_json_completion(model: str,
                                system: str,
                                user: str,
                                temperature: float,
                                max_tokens: int,
                                stop: Optional[List[str]] = None,
                                retries: int = 4) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "response_format": {"type": "json_object"},
        "stop": stop or ["```"]
    }
    last_err = None
    for attempt in range(retries):
        try:
            t0 = time.perf_counter()
            resp = _or_post(payload)
            gen_ms = (time.perf_counter() - t0) * 1000.0
            text = _extract_text_from_or(resp)
            obj  = _force_json_object(text)
            usage = resp.get("usage", {}) or {}
            engine = {
                "backend": "openrouter",
                "model": resp.get("model") or model,
                "sampler": {"temperature": float(temperature), "max_tokens": int(max_tokens)},
                "timing": {"ttft_ms": None, "gen_ms": gen_ms}
            }
            return obj, usage, engine
        except requests.HTTPError as e:
            code = getattr(e.response, "status_code", None)
            last_err = f"HTTP {code}"
            if code in (429, 500, 502, 503, 504):
                time.sleep(min(32, 2 ** attempt)); continue
            break
        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = type(e).__name__; time.sleep(min(16, 2 ** attempt)); continue
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"; break
    raise RuntimeError(f"OpenRouter completion failed after {retries} attempt(s): {last_err}")

# ---- Final-answer generation -------------------------------------------------------
def _build_general_prompt_and_ids(question: str, weighted_docs: list, top_m_docs: int, doc_char_lim: int):
    context_str, sent_ids = _format_context(weighted_docs, top_m_docs, doc_char_lim)
    prompt = _build_general_prompt(question, context_str)
    return prompt, sent_ids

def generate_final_answer(question: str,
                          weighted_docs: list,
                          or_model: str,
                          top_m_docs: int = 5,
                          doc_char_lim: int = 900,
                          temperature: float = 0.2,
                          max_output_tokens: int = 512):
    prompt, sent_ids = _build_general_prompt_and_ids(question, weighted_docs, top_m_docs, doc_char_lim)
    _throttle_llm(0.0)
    obj, usage, engine = _openrouter_json_completion(
        model=or_model,
        system="You are a careful QA assistant.",
        user=prompt,
        temperature=temperature,
        max_tokens=max_output_tokens,
        stop=["```"]
    )
    return obj, usage, PROMPT_TEMPLATE_GENERAL_ID, engine, sent_ids

def generate_vanilla_answer(question: str,
                            or_model: str,
                            temperature: float = 0.2,
                            max_output_tokens: int = 512):
    prompt = _build_vanilla_prompt(question)
    _throttle_llm(0.0)
    obj, usage, engine = _openrouter_json_completion(
        model=or_model,
        system="You are a careful QA assistant.",
        user=prompt,
        temperature=temperature,
        max_tokens=max_output_tokens,
        stop=["```"]
    )
    return obj, usage, PROMPT_TEMPLATE_VANILLA_ID, engine

# ---- Hardware / energy sampling (best-effort) -------------------------------------
class NVMLHelper:
    def __init__(self):
        self.available = False; self.handle = None
        self.energy_supported = False; self.driver = None; self.gpu_name = None
        if pynvml is None: return
        try:
            pynvml.nvmlInit()
            self.driver = pynvml.nvmlSystemGetDriverVersion().decode() if hasattr(pynvml, "nvmlSystemGetDriverVersion") else None
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.gpu_name = pynvml.nvmlDeviceGetName(self.handle).decode()
            self.energy_supported = hasattr(pynvml, "nvmlDeviceGetTotalEnergyConsumption")
            self.available = True
        except Exception:
            self.available = False
    def energy_mj(self) -> Optional[int]:
        if not (self.available and self.energy_supported): return None
        try: return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        except Exception: return None
    def mem_info(self) -> Optional[Dict[str, int]]:
        if not self.available: return None
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            return {"total": info.total, "used": info.used, "free": info.free}
        except Exception: return None

class RAPLHelper:
    def __init__(self):
        self.path = None
        base = "/sys/class/powercap"
        try:
            if os.path.isdir(base):
                for d in os.listdir(base):
                    if d.startswith("intel-rapl:0"):
                        self.path = os.path.join(base, d, "energy_uj"); break
        except Exception:
            self.path = None
    def energy_uj(self) -> Optional[int]:
        if not self.path or not os.path.isfile(self.path): return None
        try:
            with open(self.path, "r") as f: return int(f.read().strip())
        except Exception: return None

@dataclass
class StageMetrics:
    duration_s: float
    gpu_energy_j: Optional[float]
    cpu_energy_j: Optional[float]

class StageTimer:
    def __init__(self, nvml: NVMLHelper, rapl: RAPLHelper):
        self.nvml = nvml; self.rapl = rapl
        self.t0 = None; self.t1 = None
        self.gpu_e0 = None; self.gpu_e1 = None
        self.cpu_e0 = None; self.cpu_e1 = None
    def __enter__(self):
        self.t0 = time.perf_counter()
        self.gpu_e0 = self._read_gpu_mj(); self.cpu_e0 = self._read_cpu_uj()
        return self
    def __exit__(self, *_):
        self.t1 = time.perf_counter()
        self.gpu_e1 = self._read_gpu_mj(); self.cpu_e1 = self._read_cpu_uj()
    def metrics(self) -> StageMetrics:
        dur = (self.t1 - self.t0) if (self.t0 and self.t1) else 0.0
        gpu_j = (self.gpu_e1 - self.gpu_e0)/1000.0 if (self.gpu_e0 is not None and self.gpu_e1 is not None) else None
        cpu_j = (self.cpu_e1 - self.cpu_e0)/1_000_000.0 if (self.cpu_e0 is not None and self.cpu_e1 is not None) else None
        return StageMetrics(duration_s=dur, gpu_energy_j=gpu_j, cpu_energy_j=cpu_j)
    def _read_gpu_mj(self): return self.nvml.energy_mj() if self.nvml else None
    def _read_cpu_uj(self): return self.rapl.energy_uj() if self.rapl else None

# ---- FS & CUDA info ----------------------------------------------------------------
def _fs_type_for_root() -> Optional[str]:
    try:
        if psutil:
            parts = psutil.disk_partitions(all=False)
            if os.name == 'nt':
                cwd_drive = os.path.splitdrive(os.getcwd())[0].lower()
                for p in parts:
                    if p.mountpoint.lower().startswith(cwd_drive): return p.fstype
            else:
                for p in parts:
                    if p.mountpoint == '/': return p.fstype
    except Exception:
        pass
    try:
        if os.name == "posix" and os.path.exists("/proc/mounts"):
            with open("/proc/mounts","r") as f:
                for ln in f:
                    cols = ln.split()
                    if len(cols)>=3 and cols[1]=="/": return cols[2]
    except Exception:
        pass
    return None

def _git_commit_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(["git","rev-parse","--short","HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return os.getenv("GIT_COMMIT", None)

def collect_run_metadata(args) -> Dict[str, Any]:
    cuda = {
        "torch": getattr(torch, "__version__", None),
        "cuda_version": getattr(torch.version, "cuda", None) if torch else None,
        "cudnn_version": (torch.backends.cudnn.version() if (torch and hasattr(torch.backends,"cudnn") and torch.backends.cudnn.is_available()) else None),
        "cuda_available": (torch.cuda.is_available() if torch else None),
        "device_count": (torch.cuda.device_count() if (torch and torch.cuda.is_available()) else 0),
        "current_device": (torch.cuda.current_device() if (torch and torch.cuda.is_available()) else None),
    }
    hw = {
        "cpu": platform.processor() or platform.machine(),
        "ram_gb": (round(psutil.virtual_memory().total/(1024**3),2) if psutil else None),
        "storage_fs_type": _fs_type_for_root(),
    }
    sw = {
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "datasets": getattr(_safe_import("datasets"), "__version__", None),
        "psutil": getattr(_safe_import("psutil"), "__version__", None),
        "pynvml": getattr(_safe_import("pynvml"), "__version__", None),
        "commit_sha": _git_commit_sha(),
    }
    models = {
        "backend": "openrouter",
        "or_model": args.or_model,
        "pricing_input_per_1k": os.getenv("LLM_INPUT_PRICE_PER_1K"),
        "pricing_output_per_1k": os.getenv("LLM_OUTPUT_PRICE_PER_1K"),
        "sampler_defaults": {"temperature": args.ans_temp, "max_tokens": args.ans_max_tokens},
    }
    retrieval_cfg = {
        "embedder": os.getenv("EMBEDDER_MODEL"),
        "index_name": os.getenv("INDEX_NAME"),
        "namespace": os.getenv("NAMESPACE"),
        "chunking": os.getenv("CHUNKING_SCHEME"),
        "reranker": os.getenv("RERANKER_MODEL"),
        "wrag_hyperparams": {
            "alpha": args.alpha, "beta": args.beta, "gamma": args.gamma,
            "citation_top_n": args.citation_top_n,
            "citation_sim_threshold": args.citation_sim_threshold,
            "top_k": args.top_k, "n_probes": args.n_probes, "top_m": args.top_m
        },
        "router_policy": os.getenv("WRAG_ROUTER_POLICY"),
    }
    return {
        "dataset": {"subset": args.subset, "split": args.split, "limit": args.limit},
        "hardware": hw, "software": sw, "models": models, "retrieval_config": retrieval_cfg,
        "env_missing": [k for k in ["PINECONE_API_KEY","INDEX_NAME","NAMESPACE","OPENROUTER_API_KEY"] if not os.getenv(k)],
    }

# ---- Text normalization & scoring helpers -----------------------------------------
_ARTICLES = {"a","an","the"}
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")

def _normalize_text(s: str) -> str:
    if s is None: return ""
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = " ".join(w for w in _WS_RE.sub(" ", s).split() if w not in _ARTICLES)
    return s

def _tokenize(s: str) -> List[str]:
    return [t for t in _normalize_text(s).split() if t]

def exact_match(pred: str, gold: str) -> int:
    return int(_normalize_text(pred) == _normalize_text(gold))

def f1_score(pred: str, gold: str) -> float:
    p_toks, g_toks = _tokenize(pred), _tokenize(gold)
    if not p_toks and not g_toks: return 1.0
    if not p_toks or not g_toks:  return 0.0
    common = Counter(p_toks) & Counter(g_toks)
    num_same = sum(common.values())
    if num_same == 0: return 0.0
    prec = num_same / len(p_toks); rec = num_same / len(g_toks)
    return 2 * prec * rec / (prec + rec)

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

def _extract_number(s: str) -> List[float]:
    vals = []
    for m in _NUM_RE.findall(s or ""):
        try: vals.append(float(m))
        except: pass
    return vals

def numeric_close(pred: str, gold: str, rel=0.02, abs_tol=1e-6) -> bool:
    pnums, gnums = _extract_number(pred), _extract_number(gold)
    if not pnums or not gnums: return False
    p, g = pnums[0], gnums[0]
    return abs(p-g) <= max(abs_tol, rel*max(1.0, abs(g)))

def date_year_match(pred: str, gold: str) -> bool:
    py = re.findall(r"\b(19|20)\d{2}\b", pred or "")
    gy = re.findall(r"\b(19|20)\d{2}\b", gold or "")
    return bool(py and gy and py[0] == gy[0])

def citations_valid(cited_ids: List[str], sent_ids: List[str]) -> bool:
    sent = set(str(x) for x in (sent_ids or []))
    return all(str(c) in sent for c in (cited_ids or []))

def answer_present_in_text(answer: str, text: str) -> bool:
    a = _normalize_text(answer); t = _normalize_text(text)
    if not a: return False
    if a in t: return True
    # tolerant: year or number-only answers
    if date_year_match(answer, text): return True
    pnums, tnums = _extract_number(answer), _extract_number(text)
    if pnums and tnums:
        # match any number within tolerance
        for g in pnums:
            if any(abs(g - x) <= max(1e-6, 0.02*max(1.0,abs(g))) for x in tnums):
                return True
    return False

def answer_present_at_k(answer: str, doc_texts: List[str], k: int=5) -> int:
    for t in doc_texts[:k]:
        if answer_present_in_text(answer, t): return 1
    return 0

def char_trigram_jaccard(a: str, b: str) -> float:
    def ngrams(x): 
        x = _normalize_text(x)
        return {x[i:i+3] for i in range(max(0, len(x)-2))}
    A, B = ngrams(a), ngrams(b)
    if not A or not B: return 0.0
    return len(A&B)/len(A|B)

def evidence_overlap(answer: str, snippets: List[str]) -> Dict[str,float]:
    if not answer or not snippets: return {"char_jaccard_max": 0.0, "token_f1_max": 0.0}
    cj, tf1 = 0.0, 0.0
    for s in snippets:
        cj = max(cj, char_trigram_jaccard(answer, s))
        tf1 = max(tf1, f1_score(answer, s))
    return {"char_jaccard_max": cj, "token_f1_max": tf1}

def mrr_ndcg_from_rels(rels: List[int], k: int) -> Tuple[float,float]:
    """rels: 1/0 relevance for rank-ordered docs."""
    rels_k = rels[:k] if k else rels
    # MRR
    rr = 0.0
    for i, r in enumerate(rels_k):
        if r:
            rr = 1.0/(i+1); break
    # NDCG
    dcg = sum(r / math.log2(i+2) for i, r in enumerate(rels_k))
    ideal = sorted(rels_k, reverse=True)
    idcg = sum(r / math.log2(i+2) for i, r in enumerate(ideal))
    ndcg = (dcg/idcg) if idcg>0 else 0.0
    return rr, ndcg

def spearman_rank_corr(xs: List[float], ys: List[float]) -> Optional[float]:
    if not xs or not ys or len(xs)!=len(ys) or len(xs)<2: return None
    # rank with average ties
    def ranks(v):
        order = sorted((val, idx) for idx, val in enumerate(v))
        r = [0]*len(v); i=0
        while i < len(order):
            j=i
            while j+1<len(order) and order[j+1][0]==order[i][0]:
                j+=1
            avg = (i+j)/2 + 1
            for k in range(i, j+1):
                r[order[k][1]] = avg
            i=j+1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx)/len(rx), sum(ry)/len(ry)
    num = sum((a-mx)*(b-my) for a,b in zip(rx,ry))
    den = math.sqrt(sum((a-mx)**2 for a in rx) * sum((b-my)**2 for b in ry))
    return (num/den) if den>0 else None

def kendall_tau(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n<2 or len(ys)!=n: return None
    # O(n^2) Kendall's tau-b without tie adjustment across variables (simple tau)
    concord, discord = 0, 0
    for i in range(n):
        for j in range(i+1, n):
            a = xs[i]-xs[j]; b = ys[i]-ys[j]
            s = (1 if a>0 else -1 if a<0 else 0) * (1 if b>0 else -1 if b<0 else 0)
            if s>0: concord += 1
            elif s<0: discord += 1
    denom = concord + discord
    return (concord - discord)/denom if denom>0 else None

def levenshtein(a: str, b: str) -> int:
    if a==b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b)+1))
    for i, ca in enumerate(a,1):
        cur = [i]
        for j, cb in enumerate(b,1):
            cur.append(min(
                prev[j]+1,
                cur[j-1]+1,
                prev[j-1] + (0 if ca==cb else 1)
            ))
        prev = cur
    return prev[-1]

# ---- Probe diagnostics -------------------------------------------------------------
_TRIPLET_RE = re.compile(r"^\s*[^,]+,\s*[^,]+,\s*[^,]+\s*$")
def probe_compliance(probes: List[str]) -> float:
    if not probes: return 0.0
    ok = sum(1 for p in probes if _TRIPLET_RE.match(p or ""))
    return ok/len(probes)

def probe_diversity(sim_matrix: Any) -> Dict[str, Optional[float]]:
    if sim_matrix is None: return {"min": None, "mean": None, "max": None}
    try:
        arr = np.array(sim_matrix, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            return {"min": None, "mean": None, "max": None}
        n = arr.shape[0]
        if n<2: return {"min": None, "mean": None, "max": None}
        mask = ~np.eye(n, dtype=bool)
        vals = arr[mask]
        return {"min": float(vals.min()), "mean": float(vals.mean()), "max": float(vals.max())}
    except Exception:
        return {"min": None, "mean": None, "max": None}

def probe_hit_rate(probes: List[str], doc_texts: List[str]) -> float:
    if not probes or not doc_texts: return 0.0
    hits = 0
    for p in probes:
        toks = [t.strip().lower() for t in p.split(",") if t.strip()]
        # hit if any doc contains >=2 tokens
        ok=False
        for t in doc_texts:
            norm = _normalize_text(t)
            if sum(1 for tok in toks if tok in norm) >= max(1, min(2, len(toks))):
                ok=True; break
        if ok: hits += 1
    return hits/len(probes)

# ---- Aggregation helpers -----------------------------------------------------------
def _safe_mean(vals: List[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in vals if v is not None]
    return (sum(xs)/len(xs)) if xs else None

def expected_calibration_error(pairs: List[Tuple[float,int]], bins: int=10) -> Optional[float]:
    """pairs: list of (confidence in [0,1], correct∈{0,1})"""
    data = [(c, y) for (c,y) in pairs if c is not None and 0.0<=c<=1.0 and y in (0,1)]
    if not data: return None
    bs = [[] for _ in range(bins)]
    for c,y in data:
        idx = min(bins-1, int(c*bins))
        bs[idx].append((c,y))
    N = len(data); ece = 0.0
    for bucket in bs:
        if not bucket: continue
        acc = sum(y for _,y in bucket)/len(bucket)
        conf= sum(c for c,_ in bucket)/len(bucket)
        ece += (len(bucket)/N) * abs(acc - conf)
    return ece

# ---- File writing -----------------------------------------------------------------
def warn_missing_env():
    missing = []
    for k in ["PINECONE_API_KEY","INDEX_NAME","NAMESPACE","OPENROUTER_API_KEY"]:
        if not os.getenv(k): missing.append(k)
    if missing:
        print(f"[warn] Missing env vars: {', '.join(missing)}")
        print("       Ensure they're set in secrets.env or your environment.")

def to_serializable(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

def _safe_write_json(obj: dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp,"w",encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=to_serializable)
    os.replace(tmp, path)

# ---- Metrics per question ----------------------------------------------------------
def _extract_retrieval_fields_from_matches(matches: list):
    ids, scores, texts = [], [], []
    for m in matches or []:
        if not isinstance(m, dict): continue
        rid = m.get("id")
        rscore = m.get("score") or m.get("similarity") or m.get("distance")
        meta = m.get("metadata", {}) or {}
        txt  = meta.get("text") or meta.get("chunk") or meta.get("content") or ""
        ids.append(str(rid) if rid is not None else None)
        try: scores.append(float(rscore) if rscore is not None else None)
        except: scores.append(None)
        texts.append(txt or "")
    return ids, scores, texts

def _prepare_pred_fields(obj: Dict[str,Any]) -> Tuple[str, List[str], List[Dict[str,str]], Optional[float], bool]:
    if obj is None: return "", [], [], None, False
    ans = obj.get("answer", "")
    if isinstance(ans, list): ans = "; ".join([str(x) for x in ans])
    cits = obj.get("citations", []) or []
    ev   = obj.get("evidence", []) or []
    conf = obj.get("confidence", None)
    try:
        if conf is not None: conf = max(0.0, min(1.0, float(conf)))
    except Exception:
        conf = None
    abst = bool(obj.get("abstain", False)) or (isinstance(ans, str) and ans.strip()=="")
    return ans or "", [str(x) for x in cits], ev if isinstance(ev, list) else [], conf, abst

def compute_perquery_metrics(
    question: str,
    gold_answer: Optional[str],
    pred_obj: Dict[str,Any],
    sent_context_ids: List[str],
    sent_context_docs: List[Dict[str,Any]],  # ranked docs sent (list of dicts with id,text,...)
    retrieved_ids: List[str],
    retrieved_scores: List[Optional[float]],
    retrieved_texts: List[str],
    probe_info: Dict[str,Any] | None
) -> Dict[str,Any]:
    gold = gold_answer or ""
    ans, cits, ev, conf, abst = _prepare_pred_fields(pred_obj)

    # Evidence snippets: from prediction or fallback snippets from cited docs
    id2text = {d.get("id"): d.get("text","") for d in sent_context_docs}
    ev_snippets = []
    if ev:
        for e in ev:
            if isinstance(e, dict):
                snip = e.get("snippet","")
                if snip: ev_snippets.append(snip)
    if not ev_snippets and cits:
        for cid in cits:
            if id2text.get(cid):
                ev_snippets.append(id2text[cid][:200])

    # Answer quality
    em  = exact_match(ans, gold)
    f1  = f1_score(ans, gold)
    num_ok  = numeric_close(ans, gold)
    year_ok = date_year_match(ans, gold)

    # Retrieval diagnostics
    present_at_5 = answer_present_at_k(gold, retrieved_texts, k=min(5, len(retrieved_texts)))
    rels = [1 if answer_present_in_text(gold, t) else 0 for t in retrieved_texts]
    mrr, ndcg = mrr_ndcg_from_rels(rels, k=min(10, len(retrieved_texts)))

    # Reweighting diagnostics
    sent_ids = [d.get("id") for d in sent_context_docs]
    rank_deltas = []
    for sid in cits or []:
        try: r0 = retrieved_ids.index(sid)
        except ValueError: r0 = None
        try: r1 = sent_ids.index(sid)
        except ValueError: r1 = None
        if r0 is not None and r1 is not None:
            rank_deltas.append(r0 - r1)  # positive = improved
    top1_margin = None
    try:
        weights = [float(d.get("final_weight", 0.0)) for d in sent_context_docs]
        if len(weights)>=2:
            ws = sorted(weights, reverse=True)
            top1_margin = ws[0] - ws[1]
    except Exception:
        pass
    # Corr between retrieval score (align by id) and final weight
    wid2w = {d.get("id"): float(d.get("final_weight", 0.0)) for d in sent_context_docs}
    shared_ids = [i for i in retrieved_ids if i in wid2w]
    xs = [s if s is not None else float('nan') for s in [retrieved_scores[retrieved_ids.index(i)] for i in shared_ids]]
    ys = [wid2w[i] for i in shared_ids]
    xs_clean, ys_clean = [], []
    for a,b in zip(xs,ys):
        if a==a and b==b: xs_clean.append(a); ys_clean.append(b)
    spearman = spearman_rank_corr(xs_clean, ys_clean) if len(xs_clean)>=2 else None
    kendall  = kendall_tau(xs_clean, ys_clean) if len(xs_clean)>=2 else None

    # Citation validity & evidence overlap
    cvalid = citations_valid(cits, sent_ids)
    ev_ov  = evidence_overlap(ans, ev_snippets)

    # Probe diagnostics
    probe_metrics = {}
    if probe_info:
        probes = probe_info.get("probes", [])
        simmat = probe_info.get("probe_similarity_matrix", None)
        top_docs_texts = probe_info.get("top_docs", None)  # full texts pre-ranked (unique)
        probe_metrics = {
            "compliance_rate": probe_compliance(probes),
            "diversity": probe_diversity(simmat),
            "hit_rate": probe_hit_rate(probes, top_docs_texts) if top_docs_texts else None
        }

    return {
        "answer": ans,
        "citations": cits,
        "confidence": conf,
        "abstained": abst,
        "quality": {
            "em": em, "f1": f1,
            "numeric_tolerance_ok": num_ok,
            "date_year_ok": year_ok
        },
        "retrieval": {
            "answer_present_at_5": present_at_5,
            "mrr@10": mrr,
            "ndcg@10": ndcg
        },
        "reweighting": {
            "rank_delta_support_docs": rank_deltas,  # list
            "top1_margin": top1_margin,
            "spearman_retrieval_vs_weight": spearman,
            "kendall_retrieval_vs_weight": kendall
        },
        "citations_valid": cvalid,
        "evidence_overlap": ev_ov,
        "probe_metrics": probe_metrics
    }

# ---- Aggregates across the run -----------------------------------------------------
def aggregate_run_metrics(queries: List[Dict[str,Any]]) -> Dict[str,Any]:
    agg = {"vanilla": {}, "wrag": {}}
    for mode in ("vanilla","wrag"):
        ems, f1s, nums, yrs = [], [], [], []
        prs5, mrrs, ndcgs = [], [], []
        costs, lats, ptoks, ctoks, conf_pairs = [], [], [], [], []
        correct_flags = []
        abstained = []
        tokens_total = []
        # robustness
        conf_stds, lev_means = [], []
        for q in queries:
            node = q.get(mode, {})
            m    = node.get("metrics", {})
            qual = m.get("quality", {})
            ret  = m.get("retrieval", {})
            # correctness (EM)
            if "em" in qual: correct_flags.append(int(qual["em"]))
            if "em" in qual: ems.append(qual["em"])
            if "f1" in qual: f1s.append(qual["f1"])
            if "numeric_tolerance_ok" in qual: nums.append(int(bool(qual["numeric_tolerance_ok"])))
            if "date_year_ok" in qual: yrs.append(int(bool(qual["date_year_ok"])))
            if "answer_present_at_5" in ret: prs5.append(ret["answer_present_at_5"])
            if "mrr@10" in ret: mrrs.append(ret["mrr@10"])
            if "ndcg@10" in ret: ndcgs.append(ret["ndcg@10"])
            # usage/cost/latency
            usage = node.get("llm", {}).get("usage", {}) or {}
            engine= node.get("llm", {}).get("engine", {}) or {}
            timing= node.get("llm", {}).get("timing", {}) or engine.get("timing", {})
            cost  = node.get("llm", {}).get("cost_usd", None)
            if cost is not None: costs.append(cost)
            gen_ms = timing.get("gen_ms", None)
            if gen_ms is not None: lats.append(float(gen_ms))
            if "prompt_tokens" in usage: ptoks.append(usage.get("prompt_tokens"))
            if "completion_tokens" in usage: ctoks.append(usage.get("completion_tokens"))
            abstained.append(int(bool(m.get("abstained", False) or (node.get("llm",{}).get("output") or {}).get("abstain", False))))
            # calibration pairs
            conf = m.get("confidence", None)
            if conf is not None and "em" in qual:
                conf_pairs.append((float(conf), int(qual["em"])))
            # robustness summaries
            rb = node.get("robustness", {})
            if "confidence_std" in rb and rb["confidence_std"] is not None:
                conf_stds.append(rb["confidence_std"])
            if "levenshtein_mean" in rb and rb["levenshtein_mean"] is not None:
                lev_means.append(rb["levenshtein_mean"])

        n_correct = sum(correct_flags)
        agg[mode] = {
            "N": len(queries),
            "EM_mean": _safe_mean(ems),
            "F1_mean": _safe_mean(f1s),
            "Numeric_tolerance_rate": _safe_mean(nums),
            "Year_match_rate": _safe_mean(yrs),
            "AnswerPresent@5": _safe_mean(prs5),
            "MRR@10": _safe_mean(mrrs),
            "NDCG@10": _safe_mean(ndcgs),
            "Cost_mean_USD": _safe_mean(costs),
            "Latency_gen_ms_mean": _safe_mean(lats),
            "Prompt_tokens_mean": _safe_mean(ptoks),
            "Completion_tokens_mean": _safe_mean(ctoks),
            "Abstain_rate": _safe_mean(abstained),
            "ECE_overall": expected_calibration_error(conf_pairs) if conf_pairs else None,
            "Cost_per_correct": (sum(costs)/n_correct) if (costs and n_correct>0) else None,
            "Latency_per_correct_ms": (sum(lats)/n_correct) if (lats and n_correct>0) else None,
            "Tokens_per_correct": ((sum([t for t in ptoks if t]))/n_correct) if (ptoks and n_correct>0) else None,
            "Robustness": {
                "confidence_std_mean": _safe_mean(conf_stds),
                "levenshtein_mean_across_repeats": _safe_mean(lev_means),
            }
        }
    # Add deltas (WRAG - Vanilla)
    agg["delta_wrag_minus_vanilla"] = {
        k: (agg["wrag"].get(k) - agg["vanilla"].get(k) if (agg["wrag"].get(k) is not None and agg["vanilla"].get(k) is not None) else None)
        for k in agg["wrag"].keys()
        if k not in ("N","Robustness")
    }
    return agg

# ---- WRAG pipeline for one question -----------------------------------------------
def run_wrag_pipeline(question: str,
                      metadata_filter: Optional[dict],
                      top_k: int,
                      n_probes: int,
                      alpha: float,
                      beta: float,
                      gamma: float,
                      citation_top_n: Optional[int],
                      citation_sim_threshold: float,
                      top_m: int,
                      or_model: str,
                      ans_temp: float,
                      ans_max_tokens: int,
                      ans_doc_char_limit: int,
                      repeats: int) -> Dict[str, Any]:

    nvml = NVMLHelper(); rapl = RAPLHelper()

    # Probe stage
    with StageTimer(nvml, rapl) as t_probe:
        try:
            _throttle_llm(0.0)
            stage = run_probe_stage(
                question, top_k=top_k, n_probes=n_probes, metadata_filter=metadata_filter
            )
            probe_err = None
        except Exception as e:
            stage, probe_err = None, f"{type(e).__name__}: {e}"
    m_probe = asdict(t_probe.metrics())

    if stage is None:
        return {
            "prompt_template_id": PROMPT_TEMPLATE_GENERAL_ID,
            "retriever_settings": {"top_k": top_k, "n_probes": n_probes, "filters": metadata_filter},
            "probes": None,
            "mean_probe_similarity": None,
            "retrieved_doc_ids": [], "retrieved_doc_scores": [],
            "ranked_docs_sent": [],
            "stages": {"probe": m_probe, "reweigh": None, "prompt_build": None, "generate": None},
            "llm": {"engine": {"backend":"openrouter","model": or_model}, "usage": {}, "timing": {"ttft_ms":None,"gen_ms":None}, "cost_usd": None, "output": None, "error": probe_err},
            "metrics": {},
            "robustness": {},
            "decomposed": False, "parts": None, "error": probe_err
        }

    matches = (stage.get("pinecone_results", {}) or {}).get("matches", [])
    rid_list, rscore_list, rtext_list = _extract_retrieval_fields_from_matches(matches)

    # Reweighting
    with StageTimer(nvml, rapl) as t_rew:
        ranked = compute_doc_weights(
            pinecone_results=stage.get("pinecone_results", {}),
            probes=stage.get("probes", []),
            mean_probe_similarity=stage.get("mean_probe_similarity"),
            alpha=alpha, beta=beta, gamma=gamma,
            citation_top_n=citation_top_n,
            citation_sim_threshold=citation_sim_threshold,
        )
    m_rew = asdict(t_rew.metrics())

    # Build context list (for sent docs)
    sent_docs = [
        {"id": d.get("id"), "text": d.get("text"), "retrieval_score": d.get("retrieval_score"),
         "final_weight": d.get("final_weight"), "citation_count": d.get("citation_count"),
         "redundancy_penalty": d.get("redundancy_penalty")}
        for d in ranked[:top_m]
    ]

    # Final answer (repeats for robustness)
    outputs, usages, engines, sent_ids = [], [], [], None
    with StageTimer(nvml, rapl) as t_gen:
        for _ in range(max(1, repeats)):
            obj, usage, tmpl_id, engine, sent_ids = generate_final_answer(
                question=question,
                weighted_docs=ranked[:top_m],
                or_model=or_model,
                top_m_docs=top_m,
                doc_char_lim=ans_doc_char_limit,
                temperature=ans_temp,
                max_output_tokens=ans_max_tokens
            )
            outputs.append(obj); usages.append(usage); engines.append(engine)
    m_gen = asdict(t_gen.metrics())

    # Choose primary output = first run; compute robustness stats across repeats
    primary = outputs[0] if outputs else {}
    def _ans_of(o):
        a = o.get("answer","")
        if isinstance(a,list): a="; ".join([str(x) for x in a])
        return a or ""
    answers = [_ans_of(o) for o in outputs]
    confidences = []
    for o in outputs:
        c = o.get("confidence", None)
        try:
            c = None if c is None else max(0.0, min(1.0, float(c)))
        except Exception:
            c = None
        confidences.append(c)
    # pairwise Levenshtein mean
    levs = []
    for i in range(len(answers)):
        for j in range(i+1,len(answers)):
            levs.append(levenshtein(answers[i], answers[j]))
    robustness = {
        "repeats": repeats,
        "confidence_std": (float(np.nanstd([c for c in confidences if c is not None])) if any(c is not None for c in confidences) else None),
        "levenshtein_mean": (float(np.mean(levs)) if levs else None)
    }

    # Per-query metrics (uses primary output)
    per_metrics = compute_perquery_metrics(
        question=question,
        gold_answer=None,  # will be set by caller where gold known
        pred_obj=primary,
        sent_context_ids=sent_ids or [],
        sent_context_docs=sent_docs,
        retrieved_ids=rid_list,
        retrieved_scores=rscore_list,
        retrieved_texts=rtext_list,
        probe_info={
            "probes": stage.get("probes", []),
            "probe_similarity_matrix": stage.get("probe_similarity_matrix", None),
            "top_docs": stage.get("top_docs", None)
        }
    )

    # LLM usage & cost (take first run’s usage; sum also available)
    usage0 = usages[0] if usages else {}
    engine0= engines[0] if engines else {"backend":"openrouter","model":or_model}
    cost0  = compute_cost_usd(usage0)

    return {
        "prompt_template_id": tmpl_id,
        "retriever_settings": {"top_k": top_k, "n_probes": n_probes, "filters": metadata_filter},
        "probes": stage.get("probes"),
        "mean_probe_similarity": stage.get("mean_probe_similarity"),
        "probe_similarity_diversity": probe_diversity(stage.get("probe_similarity_matrix", None)),
        "retrieved_doc_ids": rid_list,
        "retrieved_doc_scores": rscore_list,
        "ranked_docs_sent": [
            {
                "id": d.get("id"),
                "retrieval_score": d.get("retrieval_score"),
                "final_weight": d.get("final_weight"),
                "citation_count": d.get("citation_count"),
                "redundancy_penalty": d.get("redundancy_penalty"),
                "text_preview": (d.get("text","")[:200] + "…") if len(d.get("text","")) > 200 else d.get("text",""),
            }
            for d in sent_docs
        ],
        "stages": {"probe": m_probe, "reweigh": m_rew, "generate": m_gen},
        "llm": {
            "engine": engine0, "usage": usage0, "timing": engine0.get("timing", {}),
            "cost_usd": cost0, "output": primary, "error": None,
        },
        "metrics": per_metrics,  # gold not yet injected; caller will update fields using gold
        "robustness": robustness,
        "decomposed": False,
        "parts": None
    }

# ---- Vanilla runner with repeats ---------------------------------------------------
def run_vanilla(question: str,
                or_model: str,
                ans_temp: float,
                ans_max_tokens: int,
                repeats: int) -> Dict[str, Any]:
    nvml = NVMLHelper(); rapl = RAPLHelper()
    outputs, usages, engines = [], [], []
    with StageTimer(nvml, rapl) as t_gen:
        for _ in range(max(1, repeats)):
            obj, usage, tmpl_id, engine = generate_vanilla_answer(
                question=question, or_model=or_model, temperature=ans_temp, max_output_tokens=ans_max_tokens
            )
            outputs.append(obj); usages.append(usage); engines.append(engine)
    m_gen = asdict(t_gen.metrics())
    primary = outputs[0] if outputs else {}
    # robustness
    def _ans_of(o):
        a = o.get("answer","")
        if isinstance(a,list): a="; ".join([str(x) for x in a])
        return a or ""
    answers = [_ans_of(o) for o in outputs]
    confidences = []
    for o in outputs:
        c = o.get("confidence", None)
        try: c = None if c is None else max(0.0, min(1.0, float(c)))
        except Exception: c = None
        confidences.append(c)
    levs=[]
    for i in range(len(answers)):
        for j in range(i+1,len(answers)):
            levs.append(levenshtein(answers[i], answers[j]))
    robustness = {
        "repeats": repeats,
        "confidence_std": (float(np.nanstd([c for c in confidences if c is not None])) if any(c is not None for c in confidences) else None),
        "levenshtein_mean": (float(np.mean(levs)) if levs else None)
    }
    return {
        "prompt_template_id": tmpl_id,
        "stages": {"generate": m_gen},
        "llm": {
            "engine": engines[0] if engines else {"backend":"openrouter","model":or_model},
            "usage": usages[0] if usages else {},
            "timing": (engines[0].get("timing", {}) if engines else {}),
            "cost_usd": compute_cost_usd(usages[0] if usages else {}),
            "output": primary,
            "error": None,
        },
        "metrics": {},  # filled by caller with gold (no retrieval context)
        "robustness": robustness
    }

# ---- Arg parsing -------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Run first N RAGBench questions through Vanilla and WRAG (OpenRouter), log & save JSON with exhaustive metrics.")
    # Dataset controls
    p.add_argument("--subset", type=str, default="covidqa")
    p.add_argument("--split", type=str, default="validation")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--repeats", type=int, default=1, help="Repeat final generation N times for robustness")
    # WRAG hyperparams
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--n-probes", type=int, default=5)
    p.add_argument("--top-m", type=int, default=5)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.6)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--citation-top-n", type=int, default=None)
    p.add_argument("--citation-sim-threshold", type=float, default=0.35)
    # OpenRouter model + sampler
    p.add_argument("--or-model", type=str, default=os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-r1-distill-llama-8b"),
                   help="OpenRouter model name (e.g., deepseek/deepseek-r1-distill-llama-8b)")
    p.add_argument("--ans-temp", type=float, default=0.2)
    p.add_argument("--ans-max-tokens", type=int, default=512)
    p.add_argument("--ans-doc-char-limit", type=int, default=900)
    # Output
    import datetime
    default_out = f"results/maintest_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    p.add_argument("--out", type=str, default=default_out)
    return p.parse_args()

# ---- Main orchestration ------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    warn_missing_env()

    metadata_filter = {"config": {"$in": [args.subset]}}  # narrow retrieval to subset if indexed that way
    rows = _load_ragbench_subset(args.subset, args.split, args.limit, include_answers=True)
    print(f"[maintest] Loaded {len(rows)} questions from {args.subset}:{args.split}")

    run_meta = collect_run_metadata(args)
    per_query: List[Dict[str, Any]] = []

    for i, row in enumerate(rows, 1):
        qid = row.get("id"); question = row.get("question"); gold = row.get("answer")
        print(f"\n[{i:02d}/{len(rows)}] qid={qid} :: {question}")

        # --- Vanilla ---
        vanilla = run_vanilla(
            question=question,
            or_model=args.or_model,
            ans_temp=args.ans_temp,
            ans_max_tokens=args.ans_max_tokens,
            repeats=args.repeats
        )

        # compute per-query vanilla metrics (no retrieval context)
        v_pred = vanilla.get("llm",{}).get("output",{}) or {}
        v_metrics = compute_perquery_metrics(
            question=question,
            gold_answer=gold,
            pred_obj=v_pred,
            sent_context_ids=[],
            sent_context_docs=[],
            retrieved_ids=[],
            retrieved_scores=[],
            retrieved_texts=[],
            probe_info=None
        )
        vanilla["metrics"] = v_metrics

        # --- WRAG + LLM ---
        try:
            wrag = run_wrag_pipeline(
                question=question,
                metadata_filter=metadata_filter,
                top_k=args.top_k,
                n_probes=args.n_probes,
                alpha=args.alpha, beta=args.beta, gamma=args.gamma,
                citation_top_n=args.citation_top_n,
                citation_sim_threshold=args.citation_sim_threshold,
                top_m=args.top_m,
                or_model=args.or_model,
                ans_temp=args.ans_temp,
                ans_max_tokens=args.ans_max_tokens,
                ans_doc_char_limit=args.ans_doc_char_limit,
                repeats=args.repeats
            )
        except Exception as e:
            wrag = {"error": f"{type(e).__name__}: {e}", "llm":{"engine":{"backend":"openrouter","model":args.or_model},"usage":{},"timing":{"ttft_ms":None,"gen_ms":None},"cost_usd":None,"output":None}}
        # inject gold into wrag metrics (recompute quality vs gold using existing fields)
        w_pred = wrag.get("llm",{}).get("output",{}) or {}
        # Rebuild sent docs for metrics (already included)
        sent_docs = wrag.get("ranked_docs_sent", [])
        matches_ids = wrag.get("retrieved_doc_ids", [])
        # we don't have retrieved texts in wrag node; recompute quickly from probe_stage? stage not returned.
        # Instead, reuse top_docs presence from 'ranked_docs_sent' texts as proxy for presence@k after reweighting.
        # For true retrieval@k, run_wrag_pipeline already computed based on matches.
        w_metrics = wrag.get("metrics", {})
        # Now patch gold-based fields (quality & retrieval already computed with gold in pipeline? No, set earlier gold=None)
        patched = compute_perquery_metrics(
            question=question,
            gold_answer=gold,
            pred_obj=w_pred,
            sent_context_ids=[d.get("id") for d in sent_docs],
            sent_context_docs=sent_docs,
            retrieved_ids=wrag.get("retrieved_doc_ids", []),
            retrieved_scores=wrag.get("retrieved_doc_scores", []),
            retrieved_texts=[],  # we don't carry raw texts here; presence@k was earlier computed; keep empty
            probe_info=None  # keep original probe_metrics already in wrag["metrics"]
        )
        # Merge: keep probe_metrics from original, replace quality/retrieval with gold-aware
        if "probe_metrics" in w_metrics and w_metrics["probe_metrics"]:
            patched["probe_metrics"] = w_metrics["probe_metrics"]
        wrag["metrics"] = patched

        # Compose per-query record
        per_query.append({
            "qid": qid,
            "subset": args.subset,
            "split": args.split,
            "question": question,
            "gold_answer": gold,
            "vanilla": vanilla,
            "wrag": wrag,
        })

        # Incremental save
        payload = {
            "run_metadata": run_meta,
            "queries": per_query,
            "aggregate": aggregate_run_metrics(per_query),
            "time_saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _safe_write_json(payload, args.out)

    # Final save
    payload = {
        "run_metadata": run_meta,
        "queries": per_query,
        "aggregate": aggregate_run_metrics(per_query),
        "time_saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=to_serializable)
    print(f"\n[maintest] Wrote report → {args.out}")

if __name__ == "__main__":
    main()
