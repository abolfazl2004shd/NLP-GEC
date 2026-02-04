from typing import Optional


def classify_error(original: str, corrected: str) -> Optional[str]:
    """Simple rule-based classifier for common GEC error types.

    This is intentionally lightweight. For production, replace with learned classifier.
    """
    o = original.lower()
    c = corrected.lower()
    # simple deterministic checks
    if (" a " in o or " an " in o or " the " in o) and (" a " not in c and " an " not in c and " the " not in c):
        return "DET"
    # preposition heuristics
    prep_list = ["in", "on", "at", "by", "for", "to", "with", "about", "against", "between", "into", "through"]
    for p in prep_list:
        if f" {p} " in o and f" {p} " not in c:
            return "PREP"
    # subject-verb agreement: naive number mismatch check
    o_tokens = o.split()
    c_tokens = c.split()
    if len(o_tokens) >= 2 and len(c_tokens) >= 2 and o_tokens[0] != c_tokens[0]:
        return "SVA"

    # fallback
    return None


class ErrorTaxonomy:
    def __init__(self):
        self._map = {"VT": "Verb Tense", "PREP": "Preposition", "DET": "Determiner", "SVA": "Subject-Verb Agreement"}

    def lookup(self, code: str) -> str:
        return self._map.get(code, "Other")

    def register(self, code: str, desc: str):
        self._map[code] = desc


taxonomy = ErrorTaxonomy()
