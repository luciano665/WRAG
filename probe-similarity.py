from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# ---- SETUP ----

# 1. Load embedding model (same as used for the passage DB)
embed_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

# 2. Set up OpenRouter LLM API
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key= os.getenv('OPENROUTER_API_KEY') 
)
# ---- FUNCTION: Generate N Probes ----

def generate_probes(question, n_probes=5, model_name="mistralai/mistral-7b-instruct"):
    sys_msg = "Return three comma-separated keywords for the answer to the question."
    probes = []
    for _ in range(n_probes):
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": question},
            ],
            max_tokens=20,
            temperature=0.7
        )
        probes.append(resp.choices[0].message.content.strip())
    return probes

# ---- FUNCTION: Embed Probes ----

def embed_probes(probes, embed_model):
    """Embed each probe using the embedding model."""
    probe_vecs = embed_model.encode(probes, convert_to_numpy=True)
    return probe_vecs

# ---- FUNCTION: Compute Semantic Similarity ----

def probe_similarity_matrix(probe_vecs):
    """Compute the cosine similarity matrix between probe embeddings."""
    sim_matrix = cosine_similarity(probe_vecs)
    return sim_matrix

# ---- EXAMPLE USAGE ----

if __name__ == "__main__":
    # Example question (replace with your actual question)
    question = "What is the theory of relativity?"

    # 2. Generate N probes from LLM
    n_probes = 5
    probes = generate_probes(question, n_probes=n_probes)
    print("Generated probes:", probes)

    # 3. Embed the probes
    probe_vecs = embed_probes(probes, embed_model)

    # 4. Compute semantic similarity
    sim_matrix = probe_similarity_matrix(probe_vecs)
    print("Probe similarity matrix (rounded):")
    print(np.round(sim_matrix, 2))

    # Optional: Print average similarity (off-diagonal)
    mean_sim = np.mean(sim_matrix[np.triu_indices(n_probes, k=1)])
    print(f"Mean probe similarity: {mean_sim:.2f}")

