from Quering_RAG.client_embedder import init_pinecone, embed
from Quering_RAG.config_pinecone import EMBEDDING_MODEL, INDEX_NAME, NAMESPACE
from sentence_transformers import SentenceTransformer
import pinecone
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False)

def search(index, query_vector, top_k=5, namespace=None):
    return index.query(
        vector=query_vector.tolist(),
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )

if __name__ == "__main__":
    question = "what is the yearly amortization rate related to the trademarks?"
    top_k = 5

    print("Embedding model:", EMBEDDING_MODEL)
    print("Index name from config:", INDEX_NAME)
    print("Pinecone SDK:", getattr(pinecone, "__version__", "unknown"))

    # Build query vector using the SAME function used at ingestion
    query_vector = embed([question])[0]

    # Connect to index
    index = init_pinecone()

    # Show index stats (includes namespace counts)
    try:
        stats = index.describe_index_stats()
        print("Index stats:", stats)
    except Exception as e:
        print("Could not fetch index stats:", e)

    ns = NAMESPACE  # from config
    print(f"Querying index='{INDEX_NAME}' namespace='{ns}' top_k={top_k}")
    resp = search(index, query_vector, top_k=top_k, namespace=ns)

    # Pinecone v3 returns a QueryResponse; normalize to a dict for printing
    results = resp.to_dict() if hasattr(resp, "to_dict") else resp
    matches = results.get("matches", [])
    print("Raw query response:", results)

    print(f"\nResults for: '{question}' (top {top_k})\n" + "="*60)
    if not matches:
        print("No matches returned. Check that:")
        print(" - Vectors exist in this index (see namespace counts in stats)")
        print(" - You used the SAME embedding model & normalization for ingestion and query")
        print(" - You are querying the correct namespace")
    else:
        for i, m in enumerate(matches, 1):
            meta = m.get("metadata", {}) or {}
            text = meta.get("text") or meta.get("chunk") or meta.get("content") or ""
            preview = (text[:200] + ("..." if len(text) > 200 else "")) if text else "[No text found]"
            print(f"Result {i}:")
            print("ID:    ", m.get("id"))
            print("Score: ", m.get("score"))
            print("Keys:  ", list(meta.keys()))
            print("Preview:", preview)
            print("-" * 40)
