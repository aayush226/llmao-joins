# file: llmao_joins/llm_client.py
import os
from openai import OpenAI
from typing import Literal

LLMModel = Literal["gpt-4o-mini"]

# Create OpenAI client
client = OpenAI(api_key=os.getenv("LLM_API_KEY"))

def set_openai_key(api_key: str) -> None:
    global client
    client = OpenAI(api_key=api_key)

def ask_if_synonyms(left: str, right: str, model: LLMModel = "gpt-4o-mini") -> str:
    """
    Ask the LLM if two strings are synonyms using a strict YES/NO prompt.
    Returns: "YES", "NO", or "UNKNOWN"
    """
    prompt = f"""You are a strict entity matcher. Answer with only YES or NO.

Are the following two values synonyms?

Left: "{left}"
Right: "{right}"
Answer:"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a strict matcher. Answer only YES or NO."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        answer = response.choices[0].message.content.strip().upper()
        if answer in {"YES", "NO"}:
            return answer
        return "UNKNOWN"
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return "UNKNOWN"
