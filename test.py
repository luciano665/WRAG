import google.generativeai as genai

genai.configure(api_key="AIzaSyCtmJhReG417tRL6bXnOBhr0UNXLwiLG1w")
model = genai.GenerativeModel("gemini-2.0-flash")
prompt = (
    "Given the context: Adenovirus infection can suppress T-cell count.\n\n"
    "Question: What role does T-cell count play in severe HAdV-55 infection?\n\n"
    "Respond with ONLY three comma-separated keywords that answer the question. "
    "Do NOT include an explanation, introduction, or sentence. For example: keyword1, keyword2, keyword3"
)

resp = model.generate_content(
    prompt,
    generation_config={"max_output_tokens": 20}
)
print(resp.text)
