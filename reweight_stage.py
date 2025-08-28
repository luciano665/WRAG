# reweight_stage.py

# **CAN LOOK INTO ADDING RETRIEVAL SCORE INTO REWEIGHTING EQ**

import os
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from Quering_RAG.client_embedder import embed  # BGE embeddings

# ---- Hyperparameters ----
ALPHA = 1.0   # agreement weight
BETA  = 0.6   # citation count weight
GAMMA = 0.5   # redundancy penalty weight

# Soft-citation settings
CITATION_TOP_N: Optional[int] = None     # if set, doc is "cited" when it's in top-N for a probe
CITATION_SIM_THRESHOLD: float = 0.6      # or count as cited if sim >= threshold


# ---- Defensive de-dupe (optional but recommended) ----
def _dedupe_matches(pinecone_results: Dict[str, Any], by: str = "text") -> Dict[str, Any]:
    """
    Drop duplicates before weighting.
    by="text"  -> dedupe exact text matches
    by="chunk" -> dedupe by (config, doc_index, chunk_index)
    """
    matches = (pinecone_results.get("matches", []) or [])
    seen, deduped = set(), []
    for m in matches:
        meta = m.get("metadata", {}) or {}
        if by == "chunk":
            key = (meta.get("config"), meta.get("doc_index"), meta.get("chunk_index"))
        else:
            key = (meta.get("text") or "").strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(m)
    pruned = dict(pinecone_results)
    pruned["matches"] = deduped
    return pruned


def _collect_doc_texts(pinecone_results: Dict[str, Any]) -> Tuple[List[str], List[str], np.ndarray]:
    texts, ids, scores = [], [], []
    for m in (pinecone_results.get("matches", []) or []):
        meta = m.get("metadata", {}) or {}
        txt  = meta.get("text") or meta.get("chunk") or meta.get("content") or ""
        if txt:
            texts.append(txt)
            ids.append(m.get("id"))
            scores.append(float(m.get("score", 0.0)))
    return ids, texts, np.array(scores, dtype=float) if scores else np.zeros((0,), dtype=float)


def _probe_doc_similarity(probes: List[str], doc_texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      sim_doc_probe: [num_docs, num_probes] cosine similarities
      doc_vecs:      [num_docs, dim] document embeddings (for redundancy)
    """
    # Always embed docs
    doc_vecs = embed(doc_texts) if doc_texts else np.zeros((0, 768), dtype=float)  # dim placeholder

    # If no probes, return empty sim matrix
    if not probes:
        sim = np.zeros((len(doc_texts), 0), dtype=float)
        return sim, doc_vecs

    probe_vecs = embed(probes)
    sim = cosine_similarity(doc_vecs, probe_vecs) if len(doc_texts) else np.zeros((0, len(probes)), dtype=float)
    return sim, doc_vecs


def _compute_citation_count(sim_doc_probe: np.ndarray,
                            top_n: Optional[int] = None,
                            sim_threshold: Optional[float] = None) -> np.ndarray:
    """
    Count, for each doc, how many probes 'cite' it.
    - If top_n is given: a probe cites the doc if the doc is in probe's top_n by similarity.
    - Else use sim_threshold (default 0.6): cite if similarity >= threshold.
    Returns int array shape [num_docs].
    """
    num_docs, num_probes = sim_doc_probe.shape
    if num_docs == 0:
        return np.zeros((0,), dtype=int)
    if num_probes == 0:
        return np.zeros((num_docs,), dtype=int)

    if top_n is not None:
        counts = np.zeros((num_docs,), dtype=int)
        for j in range(num_probes):
            top_idx = np.argsort(sim_doc_probe[:, j])[::-1][:int(top_n)]
            counts[top_idx] += 1
        return counts

    thr = 0.6 if sim_threshold is None else float(sim_threshold)
    return np.sum(sim_doc_probe >= thr, axis=1).astype(int)


def _compute_redundancy_penalty(doc_vecs: np.ndarray) -> np.ndarray:
    """
    MMR-style redundancy: for each doc i, penalty = max cosine sim to any other doc (j != i).
    Higher penalty if doc is near-duplicate to others.
    """
    n = len(doc_vecs)
    if n <= 1:
        return np.zeros((n,), dtype=float)
    pair = cosine_similarity(doc_vecs)
    np.fill_diagonal(pair, -1.0)  # ignore self
    penalty = pair.max(axis=1)    # max similarity to any other doc
    return np.clip(penalty, 0.0, 1.0)


def compute_doc_weights(
    pinecone_results: Dict[str, Any],
    probes: List[str],
    mean_probe_similarity: float,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
    citation_top_n: Optional[int] = CITATION_TOP_N,
    citation_sim_threshold: Optional[float] = CITATION_SIM_THRESHOLD,
    dedupe_by: str = "text",
) -> List[Dict[str, Any]]:
    """
    Returns list of dicts per doc:
      {
        id, text, retrieval_score,
        agreement, citation_count, redundancy_penalty, final_weight
      }
    Sorted by final_weight desc.
    """
    # 🔹 Defensive de-dupe (probe_stage already fetches uniques, but this is cheap insurance)
    pinecone_results = _dedupe_matches(pinecone_results, by=dedupe_by)

    ids, texts, retr_scores = _collect_doc_texts(pinecone_results)
    if not ids:
        return []

    agreement = float(mean_probe_similarity)

    sim_doc_probe, doc_vecs = _probe_doc_similarity(probes, texts)

    citation_count = _compute_citation_count(
        sim_doc_probe,
        top_n=citation_top_n,
        sim_threshold=citation_sim_threshold
    )

    redundancy_penalty = _compute_redundancy_penalty(doc_vecs)

    # Final weight
    final_weight = alpha * agreement + beta * citation_count - gamma * redundancy_penalty

    ranked: List[Dict[str, Any]] = []
    for i in range(len(ids)):
        ranked.append({
            "id": ids[i],
            "text": texts[i],
            "retrieval_score": float(retr_scores[i]) if i < len(retr_scores) else 0.0,
            "agreement": agreement,
            "citation_count": int(citation_count[i]),
            "redundancy_penalty": float(redundancy_penalty[i]),
            "final_weight": float(final_weight[i]),
        })

    ranked.sort(key=lambda x: x["final_weight"], reverse=True)
    return ranked


# Example CLI glue if you want to test this module quickly:
if __name__ == "__main__":
    # This relies on probe_stage.py which calls OpenRouter under the hood.
    from probe_stage import run_probe_stage

    q = "Where did the first known cases of MERS occur?"
    stage = run_probe_stage(q, top_k=5, n_probes=5)
    ranked = compute_doc_weights(
        stage["pinecone_results"],
        stage["probes"],
        stage["mean_probe_similarity"],
    )
    print("Top 3 by final_weight:")
    for r in ranked[:3]:
        print(r["id"], round(r["final_weight"], 3), "| cites:", r["citation_count"], "| red:", round(r["redundancy_penalty"], 2))
