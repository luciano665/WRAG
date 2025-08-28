import requests
api_key = "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"
resp = requests.get(
    "https://openrouter.ai/api/v1/key",
    headers={"Authorization": f"Bearer {api_key}"}
)
print(resp.json())
