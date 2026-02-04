import os
import json
from typing import Dict
import openai
from pydantic import ValidationError
from .config import settings
from .models import CorrectionResponse
from .logger import logger
import asyncio



def ensure_api_key():
    key = settings.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set in env or config")
    openai.api_key = key


def call_llm(prompt: str, max_tokens: int = 256) -> Dict:
    ensure_api_key()

    def extract_json(text: str) -> dict | None:
        try:
            jstart = text.find("{")
            jend = text.rfind("}")
            if jstart != -1 and jend != -1:
                j = json.loads(text[jstart : jend + 1])
                return j
        except Exception:
            return None
        return None

    def normalize_candidate(j: dict) -> dict | None:
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
                return None

    last_text = ""
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
        last_text = text
        j = extract_json(text)
        if j:
            norm = normalize_candidate(j)
            if norm:
                return norm

        # repair attempt: ask the model to return only the JSON and include the previous output for context
        prompt = prompt + "\n\nThe previous response was not a valid JSON object. Previous output:\n" + text + "\n\nIMPORTANT: Return only a single valid JSON object with keys: input, reasoning, correction, error_type."

    # second chance failed: attempt to extract a line after 'Correction:' as fallback
    try:
        # look for pattern 'Correction:'
        idx = last_text.find("Correction:")
        if idx != -1:
            cand = last_text[idx + len("Correction:"):].strip()
            # take first line as corrected sentence
            corr = cand.splitlines()[0].strip(' \"')
            if corr:
                return {"input": "", "reasoning": last_text, "correction": corr, "error_type": None}
    except Exception:
        pass

    return {"input": "", "reasoning": last_text if last_text else "", "correction": "", "error_type": None}


async def call_llm_async(prompt: str, max_tokens: int = 256) -> Dict:
    ensure_api_key()
    def extract_json(text: str) -> dict | None:
        try:
            jstart = text.find("{")
            jend = text.rfind("}")
            if jstart != -1 and jend != -1:
                j = json.loads(text[jstart : jend + 1])
                return j
        except Exception:
            return None
        return None

    def normalize_candidate(j: dict) -> dict | None:
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
                return None

    last_text = ""
    for attempt in range(2):
        try:
            resp = await openai.ChatCompletion.acreate(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a linguistics expert. Output only a JSON object as specified."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )
        except Exception as e:
            logger.exception("LLM async call failed on attempt %s: %s", attempt, e)
            continue
        try:
            text = resp["choices"][0]["message"]["content"].strip()
            last_text = text
            j = extract_json(text)
            if j:
                norm = normalize_candidate(j)
                if norm:
                    return norm
        except Exception:
            pass
        prompt = prompt + "\n\nThe previous response was not a valid JSON object. Previous output:" + "\n" + last_text + "\n\nIMPORTANT: Return only a single valid JSON object with keys: input, reasoning, correction, error_type."

    # fallback: try to extract 'Correction:' line
    try:
        idx = last_text.find("Correction:")
        if idx != -1:
            cand = last_text[idx + len("Correction:"):].strip()
            corr = cand.splitlines()[0].strip(' \"')
            if corr:
                return {"input": "", "reasoning": last_text, "correction": corr, "error_type": None}
    except Exception:
        pass

    return {"input": "", "reasoning": last_text if last_text else "", "correction": "", "error_type": None}
