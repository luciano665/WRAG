from Quering_RAG.client_embedder import init_pinecone, embed
from Quering_RAG.config_pinecone import NAMESPACE

def search(index, query_vector, top_k=5, namespace=None):
    return index.query(
        vector=query_vector.tolist(),
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )

if __name__ == "__main__":
    index = init_pinecone()
    query = "This is a test query."
    query_vector = embed([query])[0]
    resp = search(index, query_vector, top_k=5, namespace=NAMESPACE)

    results = resp.to_dict() if hasattr(resp, "to_dict") else resp
    matches = results.get("matches", [])
    if not matches:
        print("No matches.")
    else:
        for match in matches:
            meta = match.get("metadata", {}) or {}
            text = meta.get("text") or meta.get("chunk") or meta.get("content") or ""
            print("ID:", match.get("id"))
            print("Score:", match.get("score"))
            print("Text:", text[:200] + ("..." if len(text) > 200 else ""))
            print("---")
