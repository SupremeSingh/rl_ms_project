"""Deterministic text-family grouping, not semantic decontamination."""
import hashlib
import json
import re
import unicodedata

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def family_key(text):
    """Conservative number-substitution deduplication, not semantic deduplication."""
    text = unicodedata.normalize('NFKC', text).lower()
    text = re.sub(r'\d+(?:[.,]\d+)*', ' NUMBER ', text)
    return re.sub(r'\W+', ' ', text).strip()
