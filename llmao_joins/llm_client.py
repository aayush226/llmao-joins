# file: llmao_joins/llm_client.py
import os
import openai
from typing import Literal

# Set the API key from env
openai.api_key = os.getenv("LLM_API_KEY") or ""

LLMModel = Literal["gpt-4o-mini"] # Currently only one model is supported

def set_openai_key(api_key: str) -> None:
    openai.api_key = api_key

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
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a strict matcher. Answer only YES or NO."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        answer = response["choices"][0]["message"]["content"].strip().upper()
        if answer in {"YES", "NO"}:
            return answer
        return "UNKNOWN"
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return "UNKNOWN"
