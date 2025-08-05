"""
RAGBench → Pinecone chunk-ingester  (v3 • Modal edition)
-------------------------------------------------------
Runs on an NVIDIA H100 GPU in Modal.
"""
# --- IMPORTS -----------------------------------------------------------------
import os, json, sys, time
from typing import Iterable, List, Tuple

# --- CHANGE: new modal import
import modal                                     # <-- NEW

# keep all other imports
from pinecone import Pinecone, ServerlessSpec
from datasets import load_dataset
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# --- CONFIG ------------------------------------------------------------------
PINECONE_API_KEY = "**insert your key here**"
PINECONE_REGION  = "us-east-1"
INDEX_NAME       = "wrag-v3"
NAMESPACE        = "ragbench"
EMBED_MODEL      = "BAAI/bge-base-en-v1.5"          # 768-dim

CHUNK_TOKENS   = 384
OVERLAP_TOKENS = 96
META_BYTE_LIMIT = 40_000
BATCH            = 256

CONFIG_NAMES = [
    "finqa", "tatqa", "pubmedqa", "covidqa",
    "cuad", "delucionqa", "emanual", "techqa",
    "expertqa", "hagrid", "hotpotqa", "msmarco"
]

# --- CHANGE: build Modal image with deps --------------------------------------
image = (
    modal.Image.debian_slim()
    .pip_install(
        "pinecone",
        "sentence-transformers",
        "datasets",
        "transformers",
        "tqdm"
    )
)

app = modal.App(name="wrag-ingest")

# --- MAIN INGESTION FUNCTION (runs on GPU) -----------------------------------
@app.function(gpu="H100", image=image, timeout=60 * 60 * 15)   # 15-h timeout
def ingest_modal():
    start_time = time.time()

    # inside Modal container ─ re-import libs
    from pinecone import Pinecone, ServerlessSpec
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from sentence_transformers import SentenceTransformer
    from tqdm.auto import tqdm
 

    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    embedder  = SentenceTransformer(EMBED_MODEL, device="cuda")  # use GPU
    VECTOR_DIM = embedder.get_sentence_embedding_dimension()

    pc = Pinecone(api_key=PINECONE_API_KEY)
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
        )
    index = pc.Index(INDEX_NAME)

    def utf8_truncate(text: str, max_bytes: int) -> str:
        data = text.encode("utf-8")
        if len(data) <= max_bytes:
            return text
        data = data[:max_bytes]
        while data and (data[-1] & 0xC0) == 0x80:
            data = data[:-1]
        return data.decode("utf-8", errors="ignore")

    def token_chunks(tokens, size, overlap):
        step = size - overlap
        for start in range(0, len(tokens), step):
            yield tokens[start : start + size]

    def doc_to_chunks(text: str):
        ids = tokenizer(text, add_special_tokens=False).input_ids
        for chunk_ids in token_chunks(ids, CHUNK_TOKENS, OVERLAP_TOKENS):
            yield tokenizer.decode(chunk_ids)

    def ingest(split="train"):
        for cfg in CONFIG_NAMES:
            print(f"▶ Streaming '{cfg}' ({split}) …")
            ds = load_dataset("galileo-ai/ragbench", cfg, split=split, streaming=True)
            for ex in tqdm(ds, desc=f"{cfg}-{split}"):
                ex_id = ex["id"]
                for doc_i, doc_text in enumerate(ex["documents"]):
                    for c_idx, chunk in enumerate(doc_to_chunks(doc_text)):
                        meta_text = utf8_truncate(chunk, META_BYTE_LIMIT)
                        meta = {
                            "config": cfg,
                            "example_id": ex_id,
                            "doc_index": doc_i,
                            "chunk_index": c_idx,
                            "text": meta_text
                        }
                        uid = f"{ex_id}_{cfg}_{doc_i}_{c_idx}"
                        vec = embedder.encode(chunk)
                        yield uid, vec, meta

    def batched(it, size=BATCH):
        batch = []
        for item in it:
            batch.append(item)
            if len(batch) == size:
                yield batch
                batch = []
        if batch:
            yield batch

    chunk_total = batch_total = 0
    for batch in batched(ingest()):
        ids, vecs, metas = zip(*batch)
        index.upsert(
            vectors=[(ids[i], vecs[i].tolist(), metas[i]) for i in range(len(ids))],
            namespace=NAMESPACE
        )
        chunk_total += len(batch)
        batch_total += 1
        if batch_total % 20 == 0:   #  ← replace `if batch_total % 20 == 0:` with True
            print(f"⏩ {chunk_total:,} chunks in {batch_total} batches …")
    elapsed = time.time() - start_time
    print(f"\n✅ Finished {chunk_total:,} chunks in {elapsed/60:.1f} min")

# --- CHANGE: local entry point that triggers Modal run ------------------------
if __name__ == "__main__":
    # Push the code + run on Modal
    app.run(ingest_modal)
