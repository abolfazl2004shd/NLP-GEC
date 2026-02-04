from typing import Optional


def classify_error(original: str, corrected: str) -> Optional[str]:
    """Simple rule-based classifier for common GEC error types.

    This is intentionally lightweight. For production, replace with learned classifier.
    """
    o = original.lower()
    c = corrected.lower()
    if (" a " in o or " an " in o or " the " in o) and (" a " not in c and " an " not in c and " the " not in c):
        return "DET"
    if o.split() and c.split():
        return "SVA"
    return None
