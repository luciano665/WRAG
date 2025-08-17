import os
import itertools
from pinecone import Pinecone
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()


# Config for Pinecone and env

PINECONE_API_KEY="pcsk_2mE6zj_AP8ycdGJHwDn1z539GZdxJJqu1rLxt5c5MD7J9MjyihEKeSosUGr6bQuHKfuqRE"
PINECONE_ENV="us-east-1-aws"




# Get index name from env

PINECONE_INDEX="wrag-v2"
# Batch size for embed/upsert at one
BATCH_SIZE = 64

# RagBecn sub-datasets
SUBSETS = [
    "covidqa", "cuad", "delucionqa", "emanual", "expertqa",
    "finqa", "hagrid", "hotpotqa", "msmarco", "pubmedqa",
    "tatqa", "techqa",
]

# Global counter for upserted vectors
_id_counter = itertools.count()

# Init Pinecone & Embedder
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pc.Index(PINECONE_INDEX)

embedder = SentenceTransformer("all-mpnet-base-v2")
vector_dim = embedder.get_sentence_embedding_dimension()
print(f"✅ Connected to Pinecone index '{PINECONE_INDEX}' (dim={vector_dim})")


# Helpers
def batched(iterable, batch_size):
    """
    Yield successive batches (lists) of size <= batch_size from iterable.
    """
    # Get iterator
    it = iter(iterable)
    while True:
        # Slice out the next batch_size items
        batch = list(itertools.islice(it, batch_size))
        if not batch:
            break # end when no more items
        yield batch # yield current batch

def docs_generator(split="train"):
    """
    Stream RAGBench examples, flatten each example's 'documents' list into
    tuples of (unique_id, document_text, metadata payload).
    """
    for subset in SUBSETS:
        ds = load_dataset("galileo-ai/ragbench", subset, split=split, streaming=True)
        for example in ds:
            ex_id = example['id']
            for doc_text in example['documents']:
                unique_id = str(next(_id_counter))
                payload = {"subset": subset, "example_id": ex_id, "text": doc_text}   # <-- ADD "text": doc_text here!
                yield unique_id, doc_text, payload


# Ingestion into Pinecone
print("⏳ Beginning to embed & upsert RAGBench documents…")
# Iterate over vatches over the generator
for batch in batched(docs_generator(), BATCH_SIZE):
    # Extarct ids, texts, and metadata from the batch
    ids, texts, metas =zip(*batch)
    
    # 1) Embed teh batch of document texts into vectors
    vectors = embedder.encode(texts, convert_to_numpy=True)

    # 2) Prepare list to of tuples to be upserted
    to_upsert = [
        (ids[i], vectors[i].tolist(), metas[i])
        for i in range(len(ids))
    ]

    # 3) Upsert batch of vectors into Pinecone
    index.upsert(vectors=to_upsert)
    print(f"✅ Upserted batch of {len(to_upsert)} vectors")

print("🎉 All RAGBench docs are now indexed in Pinecone!")