import os
from dotenv import load_dotenv, find_dotenv

print("--- ENV DIAGNOSTIC ---")
print("CWD:", os.getcwd())

# Find .env (searches up the directory tree)
dotenv_path = find_dotenv(usecwd=True)
print("find_dotenv ->", repr(dotenv_path))

# Load it (won't override existing env vars)
loaded = load_dotenv(dotenv_path, override=False)
print("load_dotenv loaded:", loaded)

# Show a few key variables
keys = [
    "GEMINI_API_KEY",
    "PINECONE_API_KEY",
    "INDEX_NAME",
    "NAMESPACE",
    "PINECONE_REGION",
]
for k in keys:
    v = os.getenv(k)
    print(f"{k} =", repr(v))

# Also list any pre-existing env-like vars that start with these prefixes
print("\n--- All matching env vars in process ---")
for k, v in os.environ.items():
    if k in keys or k.startswith(("GEMINI_", "PINECONE_", "INDEX_", "NAMESPACE")):
        print(f"{k}={repr(v)}")
