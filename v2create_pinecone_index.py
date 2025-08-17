from pinecone import Pinecone, ServerlessSpec

API_KEY = "pcsk_2mE6zj_AP8ycdGJHwDn1z539GZdxJJqu1rLxt5c5MD7J9MjyihEKeSosUGr6bQuHKfuqRE"
NEW_INDEX_NAME = "wrag-v2"      # Or whatever name you want
EMBEDDING_DIM = 768             # Should match your model's output dimension
REGION = "us-east-1"            # Or your correct region, e.g., "us-east-1"

pc = Pinecone(api_key=API_KEY)

if NEW_INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=NEW_INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=REGION)
    )
    print(f"Index '{NEW_INDEX_NAME}' created!")
else:
    print(f"Index '{NEW_INDEX_NAME}' already exists.")
