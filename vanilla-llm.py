import os
import anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

# get env vars
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
print(f"Anthropic API Key: {anthropic_api_key[:20] if anthropic_api_key else 'NOT FOUND'}...")

# heading
print("RAG Demo: State of the Union Speech Analysis")
print("-" * 80)
print("Vanilla LLM Response (No Context)")

# get model
client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=anthropic_api_key,
)

# prompt 
# questions
questions = [
    "What did Biden say about Ukraine?",
    "What did Biden say about inflation?",
    "What did Biden mention about COVID-19?"
]

for question in questions:
    print("-" * 80)
    print(f"Question: {question}")
    print("-" * 80)

    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": question}
        ]
    )

    print(message.content[0].text)
    print("\n")


