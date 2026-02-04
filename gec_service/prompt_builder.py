from typing import List, Dict


SYSTEM_PROMPT = (
    "You are an expert linguist. Follow MaxMatch standard.\n"
    "Given Input, produce a clear reasoning (Chain-of-Thought) explaining the error,\n"
    "then produce a correction. Output MUST be a single valid JSON object and nothing else with fields: `input`, `reasoning`, `correction`, `error_type`."
)


def build_prompt(input_text: str, retrieved: List[Dict], top_k: int = 5, max_chars: int | None = None) -> str:
    """Build a CoT prompt including up to `top_k` retrieved examples.

    If `max_chars` is provided, attempt to keep the prompt length <= max_chars by
    reducing the number of retrieved examples (diversity selection) and, if needed,
    truncating the reference examples section while preserving the Task and Input.
    """
    # select up to top_k examples, attempt diversity by spacing if more available
    sel = list(retrieved or [])
    if len(sel) > top_k:
        # pick roughly evenly spaced examples to maximize diversity
        stride = max(1, len(sel) // top_k)
        sel = [sel[i] for i in range(0, len(sel), stride)][:top_k]

    shots: List[str] = []
    for r in sel:
        v = r.get("value") or r
        example = (
            f"Example Input: {v.get('input')}\nReasoning: {v.get('reasoning')}\nCorrection: {v.get('correction')}\nError Type: {v.get('error_type')}\n"
        )
        shots.append(example)

    ref_section = "" if not shots else "Reference Examples:\n" + "\n".join(shots) + "\n"

    prompt = (
        f"{SYSTEM_PROMPT}\n\n{ref_section}Task:\nInput: {input_text}\n\nPlease provide:\n1) A `reasoning` section that explains the grammatical issue.\n2) A `correction` section with the corrected sentence.\n3) An `error_type` label (VT/PREP/DET/SVA/etc.).\n\nReturn only the JSON object."
    )

    if max_chars is None:
        return prompt

    # enforce max_chars by progressively reducing examples
    if len(prompt) <= max_chars:
        return prompt

    # try reducing selected shots count progressively
    for k in range(max(0, top_k - 1), -1, -1):
        if k == 0:
            candidate_ref = ""
        else:
            stride = max(1, len(retrieved) // k)
            sel2 = [retrieved[i] for i in range(0, len(retrieved), stride)][:k]
            shots2 = [f"Example Input: { (r.get('value') or r).get('input')}\nReasoning: {(r.get('value') or r).get('reasoning')}\nCorrection: {(r.get('value') or r).get('correction')}\nError Type: {(r.get('value') or r).get('error_type')}\n" for r in sel2]
            candidate_ref = "Reference Examples:\n" + "\n".join(shots2) + "\n"

        candidate = f"{SYSTEM_PROMPT}\n\n{candidate_ref}Task:\nInput: {input_text}\n\nPlease provide:\n1) A `reasoning` section that explains the grammatical issue.\n2) A `correction` section with the corrected sentence.\n3) An `error_type` label (VT/PREP/DET/SVA/etc.).\n\nReturn only the JSON object."
        if len(candidate) <= max_chars:
            return candidate

    # as last resort, truncate the reference section to fit
    # keep Task and Input intact
    base = f"{SYSTEM_PROMPT}\n\nTask:\nInput: {input_text}\n\nPlease provide:\n1) A `reasoning` section that explains the grammatical issue.\n2) A `correction` section with the corrected sentence.\n3) An `error_type` label (VT/PREP/DET/SVA/etc.).\n\nReturn only the JSON object."
    # truncate base if it's still longer than max_chars
    if len(base) <= max_chars:
        # prepend as much of reference examples as fits
        ref_all = "Reference Examples:\n" + "\n".join(shots) + "\n"
        remaining = max_chars - len(base)
        ref_part = ref_all[:remaining]
        return f"{ref_part}{base}"
    else:
        return base[:max_chars]
