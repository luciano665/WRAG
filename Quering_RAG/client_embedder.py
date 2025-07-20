# client_embedder.py

# initializing Pinecone client below
import pinecone
from config_pinecone import API_KEY, ENVIRONMENT, INDEX_NAME

def init_pinecone():
    pinecone.init(api_key=API_KEY, environment=ENVIRONMENT)
    return pinecone.Index(INDEX_NAME)

# embedding function below

from sentence_transformers import SentenceTransformer
from config_pinecone import EMBEDDING_MODEL

model = SentenceTransformer(EMBEDDING_MODEL)

def embed(texts, normalize=True):
    return model.encode(texts, normalize_embeddings=normalize)
