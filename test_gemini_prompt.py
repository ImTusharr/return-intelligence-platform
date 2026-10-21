
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found. Check your .env file."
    )

client = genai.Client(api_key=api_key)

prompt = """
You are an e-commerce return investigation assistant.

Analyze evidence carefully. Do not assume a product is defective
without supporting evidence. Consider alternative explanations.
If evidence is insufficient, recommend human review.

Test case:
The customer selected "Not Working".
The inspection notes say the product powered on successfully.
No functional failure was observed.
The staff-confirmed reason is "Customer Expectation".

Explain what the evidence suggests and what additional evidence
would be useful. Treat the staff-confirmed reason as a reference
for this test, not as proof that every similar case has the same reason.
"""

response = client.models.generate_content(
    model=os.getenv("GEMINI_MODEL_NAME", "gemini-3.8-flash"),
    contents=prompt,
)

print(response.text)
