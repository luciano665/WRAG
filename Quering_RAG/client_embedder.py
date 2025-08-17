from pinecone import Pinecone
from Quering_RAG.config_pinecone import API_KEY, INDEX_NAME
from sentence_transformers import SentenceTransformer
from Quering_RAG.config_pinecone import EMBEDDING_MODEL

def init_pinecone():
    if not API_KEY:
        raise RuntimeError("PINECONE_API_KEY is not set. Export it in your environment.")
    pc = Pinecone(api_key=API_KEY)
    return pc.Index(INDEX_NAME)

_model = SentenceTransformer(EMBEDDING_MODEL)

def embed(texts, normalize=True):
    return _model.encode(texts, normalize_embeddings=normalize)
