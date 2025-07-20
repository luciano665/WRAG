# search_top_k.py

from client_embedder import init_pinecone, embed

# function that searches vector db for given query

def search(index, query_vector, top_k=5):
    return index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
  
# embedds query, and aggregates results for tok-k similar docs

index = init_pinecone()

query = "This is a test query."
query_vector = embed([query])[0]

results = search(index, query_vector, top_k=5)

# prints out data relating to top-k docs

for match in results['matches']:
    print("ID:", match['id'])
    print("Score:", match['score'])
    print("Text:", match['metadata']['text'])
    print("---")
