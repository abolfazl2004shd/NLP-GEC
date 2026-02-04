from typing import List, Dict


SYSTEM_PROMPT = (
    "You are an expert linguist. Follow MaxMatch standard.\n"
    "Given Input, produce a clear reasoning (Chain-of-Thought) explaining the error,\n"
    "then produce a correction. Output MUST be a single valid JSON object and nothing else with fields: `input`, `reasoning`, `correction`, `error_type`."
)


def build_prompt(input_text: str, retrieved: List[Dict], top_k: int = 5) -> str:
    shots = []
    for r in retrieved[:top_k]:
        v = r.get("value") or r
        example = (
            f"Example Input: {v.get('input')}\nReasoning: {v.get('reasoning')}\nCorrection: {v.get('correction')}\nError Type: {v.get('error_type')}\n"
        )
        shots.append(example)

    prompt = (
        f"{SYSTEM_PROMPT}\n\nReference Examples:\n{''.join(shots)}\nTask:\nInput: {input_text}\n\nPlease provide:\n1) A `reasoning` section that explains the grammatical issue.\n2) A `correction` section with the corrected sentence.\n3) An `error_type` label (VT/PREP/DET/SVA/etc.).\n\nReturn only the JSON object."
    )
    return prompt
