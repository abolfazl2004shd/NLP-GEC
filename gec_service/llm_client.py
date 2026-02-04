import os
import json
from typing import Dict
import openai
from pydantic import ValidationError
from .config import settings
from .models import CorrectionResponse



def ensure_api_key():
    key = settings.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set in env or config")
    openai.api_key = key


def call_llm(prompt: str, max_tokens: int = 256) -> Dict:
    ensure_api_key()
    for attempt in range(2):
        resp = openai.ChatCompletion.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a linguistics expert. Output only a JSON object as specified."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        text = resp["choices"][0]["message"]["content"].strip()
        try:
            jstart = text.find("{")
            jend = text.rfind("}")
            if jstart != -1 and jend != -1:
                j = json.loads(text[jstart : jend + 1])
                try:
                    CorrectionResponse(**j)
                    return j
                except ValidationError:
                    
                    mapped = {
                        "input": j.get("input", ""),
                        "reasoning": j.get("reasoning", j.get("explanation", "")),
                        "correction": j.get("correction", j.get("corrected", "")),
                        "error_type": j.get("error_type", None),
                    }
                    try:
                        CorrectionResponse(**mapped)
                        return mapped
                    except ValidationError:
                        pass
        except Exception:
            pass
        prompt = prompt + "\n\nIMPORTANT: Return only a single valid JSON object with keys: input, reasoning, correction, error_type."

    return {"input": "", "reasoning": text if text else "", "correction": "", "error_type": None}
