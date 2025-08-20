from google.colab import drive
drive.mount('/content/drive')

!pip -q install "pinecone-client[grpc]" sentence-transformers datasets transformers tqdm
# Grab the latest PyTorch w/ CUDA if not pre-installed
import os, sys, importlib, subprocess, json, time, platform
print("✅  Libraries installed — if PyTorch complains about CUDA, "
      "Menu ▸ Runtime ▸ Restart runtime and run again")

!pip uninstall -y pinecone-client pinecone && \
pip install --upgrade pinecone pinecone[grpc]
!pip -q install "pinecone[grpc]" sentence-transformers datasets transformers tqdm


import pinecone
from pinecone import Pinecone, ServerlessSpec
print("Pinecone SDK version:", pinecone.__version__)
"""
RAGBench → Pinecone chunk-ingester  (v3 • Colab edition)
Runs on any CUDA-enabled Colab GPU.
"""
import os, json, time, random, torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging              # --- FIX
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec, PineconeApiException
from tqdm.auto import tqdm

# ─── USER CONFIG ────────────────────────────────────────────────────────
PINECONE_API_KEY = "pcsk_393uFT_EoJvwP6wuWxRnb2Eu7tYeDnY1kY3qgYm6QpRfLYXLrTPC2zXK3C6fDugemeRhJU"                     # <─ add your key
PINECONE_REGION  = "us-east-1"
INDEX_NAME       = "wrag-v3"
NAMESPACE        = "ragbench"

EMBED_MODEL      = "BAAI/bge-base-en-v1.5"
CHUNK_TOKENS     = 512
OVERLAP_TOKENS   = 100
BATCH            = 256
META_BYTE_LIMIT  = 40_000
CONFIG_NAMES     = [
    "finqa","tatqa","pubmedqa","covidqa",
    "cuad","delucionqa","emanual","techqa",
    "expertqa","hagrid","hotpotqa","msmarco"
]

# EDIT: resume logic
RESUME_FROM = "covidqa"
if RESUME_FROM in CONFIG_NAMES:
    idx = CONFIG_NAMES.index(RESUME_FROM)
    CONFIG_NAMES = CONFIG_NAMES[idx+1:]
    print(f"Resuming ingestion, now processing: {CONFIG_NAMES}")

# --- EDIT: constant for safety clip -------------------------------------
MAX_MODEL_TOKENS = 512        # model’s true max input length
# ------------------------------------------------------------------------

# ─── HELPERS ─────────────────────────────────────────────────────────────
def utf8_truncate(text: str, max_bytes: int) -> str:
    data = text.encode("utf-8")
    if len(data) <= max_bytes:
        return text
    data = data[:max_bytes]
    # keep cutting until we are at a char boundary
    while data and (data[-1] & 0xC0) == 0x80:
        data = data[:-1]
    return data.decode("utf-8", errors="ignore")

def token_chunks(ids, size, overlap):
    step = size - overlap
    for s in range(0, len(ids), step):
        yield ids[s: s + size]

# ─── MAIN ────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ── Tokeniser (HF) ────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)

    # --- FIX: silence HF long-sequence warnings --------------------------
    hf_logging.set_verbosity_error()           # hide warning banner
    tokenizer.model_max_length = 10_000_000    # effectively “no limit”
    # ---------------------------------------------------------------------

    # ── Sentence-Transformers embedder ───────────────────────────
    embedder  = SentenceTransformer(EMBED_MODEL, device=device)
    embedder.max_seq_length = MAX_MODEL_TOKENS     # --- FIX: hard clip
    VECTOR_DIM = embedder.get_sentence_embedding_dimension()   # 768

    # ── Pinecone init ───────────────────────────────────────────
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_names = pc.list_indexes().names()

    if INDEX_NAME not in index_names:
        print("Creating index …")
        pc.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
        )
    else:
        existing_dim = pc.describe_index(INDEX_NAME)["dimension"]
        if existing_dim != VECTOR_DIM:
            raise ValueError(
                f"Pinecone index '{INDEX_NAME}' has dimension {existing_dim}, "
                f"but model outputs {VECTOR_DIM}. Delete & recreate the index "
                f"or switch to a {existing_dim}-dim model."
            )
    index = pc.Index(INDEX_NAME)


    # ─── STREAM CHUNKS ───────────────────────────────────────────────
    def stream_chunks():
        for cfg in CONFIG_NAMES:
            print(f"▶ {cfg}-train")
            ds = load_dataset("galileo-ai/ragbench", cfg, split="train", streaming=True)
            for ex in tqdm(ds, desc=cfg):
                ex_id = ex["id"]
                for d_i, doc in enumerate(ex["documents"]):
                    ids = tokenizer(doc, add_special_tokens=False,                       # --- FIX: explicit
                                     truncation=False).input_ids
                    for c_i, chunk_ids in enumerate(token_chunks(ids, CHUNK_TOKENS, OVERLAP_TOKENS)):

                        # 1) Clip raw IDs to model max
                        safe_ids = chunk_ids[:MAX_MODEL_TOKENS]

                        # 2) Decode for metadata
                        chunk_text = tokenizer.decode(safe_ids, skip_special_tokens=True)

                        # 3) Re-tokenize & re-clip before embedding
                        tokens_for_embed = tokenizer(chunk_text,
                                                     add_special_tokens=False,
                                                     truncation=True,
                                                     max_length=MAX_MODEL_TOKENS).input_ids
                        safe_text = tokenizer.decode(tokens_for_embed, skip_special_tokens=True)

                        # 4) Embed safely
                        vec = embedder.encode(safe_text)

                        # ASCII-safe UID
                        uid = f"{ex_id}_{cfg}_{d_i}_{c_i}"
                        uid = uid.encode("ascii", "ignore").decode()

                        meta = {
                            "config":      cfg,
                            "example_id":  ex_id,
                            "doc_index":   d_i,
                            "chunk_index": c_i,
                            "text":        utf8_truncate(chunk_text, META_BYTE_LIMIT)
                        }
                        yield uid, vec.tolist(), meta

    # ── BATCH UPSERT LOOP ─────────────────────────────────────────────
    buf, total, batches = [], 0, 0
    start = time.time()

    for uid, vec, meta in stream_chunks():
        buf.append((uid, vec, meta))
        if len(buf) == BATCH:
            index.upsert(buf, namespace=NAMESPACE)
            total += len(buf)
            batches += 1
            buf.clear()
            if batches % 20 == 0:
                print(f"⏩  {total:,} chunks in {batches} batches")

    if buf:
        index.upsert(buf, namespace=NAMESPACE)
        total += len(buf)

    elapsed = (time.time() - start) / 60
    print(f"✅ Done – {total:,} chunks in {elapsed:.1f} min")

# ─── RUN ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()


#  HTTP response body: {"code":8,"message":"Request failed. You've reached your write unit limit for the current month (2000000). To continue writing data, upgrade your plan.","details":[]}

import os
from datasets import load_dataset
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# ---- Fill in your details ----
PINECONE_API_KEY = "pcsk_393uFT_EoJvwP6wuWxRnb2Eu7tYeDnY1kY3qgYm6QpRfLYXLrTPC2zXK3C6fDugemeRhJU"           # <--- your key
PINECONE_REGION  = "us-east-1"
INDEX_NAME       = "wrag-test"
NAMESPACE        = "ragbench"
EMBED_MODEL      = "BAAI/bge-base-en-v1.5"
MAX_MODEL_TOKENS = 512

# ---- Setup ----
tok      = AutoTokenizer.from_pretrained(EMBED_MODEL)
embedder = SentenceTransformer(EMBED_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
dim      = embedder.get_sentence_embedding_dimension()

pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in pc.list_indexes().names():
    print("Creating index …")
    pc.create_index(
        name=INDEX_NAME,
        dimension=dim,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
    )
index = pc.Index(INDEX_NAME)

print("Index dim:", pc.describe_index(INDEX_NAME)["dimension"])
print("Model dim:", dim)

# ---- Embed and upsert just one short chunk ----
ds = load_dataset("galileo-ai/ragbench", "finqa", split="train", streaming=True)
ex = next(iter(ds))
doc = ex["documents"][0]
ids = tok(doc, add_special_tokens=False).input_ids
chunk_ids = ids[:MAX_MODEL_TOKENS]
chunk = tok.decode(chunk_ids)

vec = embedder.encode(chunk)
print("Vector shape:", vec.shape if hasattr(vec, "shape") else len(vec))

uid = f"smoketest_{ex['id']}_0"
meta = {"config": "finqa", "example_id": ex["id"], "chunk_index": 0, "text": chunk[:100]}

# ---- Upsert and query ----
index.upsert([(uid, vec.tolist(), meta)], namespace=NAMESPACE)
print("Upserted 1 vector.")

res = index.query(vector=vec.tolist(), top_k=5, namespace=NAMESPACE, include_metadata=True)
print("Query result:", res)
if res["matches"]:
    print("Top match ID:", res["matches"][0]["id"])
    print("Top match score:", res["matches"][0]["score"])
    print("Top match metadata snippet:", str(res["matches"][0]["metadata"])[:200])
else:
    print("No matches returned.")
