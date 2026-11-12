
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from genai.prompts import SYSTEM_PROMPT


# Load .env from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_PATH)


def get_groq_client():
    """
    Create and return an authenticated Groq client.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Check your .env file."
        )

    return Groq(api_key=api_key)


def generate_response(
    user_prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
    temperature: float = 0.2,
    max_tokens: int = 1024
) -> str:
    """
    Generate a response using the Groq LLM API.
    """

    if not user_prompt or not user_prompt.strip():
        raise ValueError("User prompt cannot be empty.")

    model = os.getenv("GROQ_MODEL")

    if not model:
        raise ValueError(
            "GROQ_MODEL is missing from the .env file."
        )

    client = get_groq_client()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=temperature,
        max_completion_tokens=max_tokens
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError("Groq returned an empty response.")

    return content.strip()


if __name__ == "__main__":

    prompt = (
        "Explain what transfer learning is and "
        "how ResNet18 uses it for image classification."
    )

    try:
        print("\nConnecting to Groq...\n")

        answer = generate_response(prompt)

        print("=" * 60)
        print("GROQ RESPONSE")
        print("=" * 60)
        print(answer)
        print("=" * 60)

    except Exception as error:
        print(f"\nGroq integration failed: {error}")
