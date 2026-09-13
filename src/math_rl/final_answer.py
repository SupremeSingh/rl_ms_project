"""Conservative numeric final-answer candidate; no symbolic math or LLM judge."""
import re
from decimal import Decimal

RULE_VERSION = "final-numeric-v1"
NUMBER = r"[+-]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?"
BOX = re.compile(r"\\boxed\s*\{([^{}]*)\}")
STATEMENT = re.compile(
    r"^(?:(?:therefore|thus|so)[,:]?\s+)?"
    r"(?:(?:the\s+)?(?:final\s+)?answer\s*(?:is\s*|:\s*)|"
    r"[^\n.!?]*?\b(?:is|are|will have|can make|spent)\s+)", re.I)
UNCERTAIN = re.compile(r"\b(?:or|maybe|approximately|about|possibly|could|not|instead)\b", re.I)


def extract_final_number(response):
    # Ignore fenced code/output as evidence, but require a closed fence.
    if response.count('```') % 2:
        return None
    prose = re.sub(r"```.*?```", "", response, flags=re.S).strip()
    if not prose:
        return None
    boxes = list(BOX.finditer(prose))
    if len(boxes) != prose.count(r"\boxed"):
        return None
    candidates = []
    for box in boxes:
        value = box.group(1).strip()
        if not re.fullmatch(NUMBER, value):
            return None
        candidates.append(Decimal(value.replace(',', '')))
    # Replace boxes and basic math delimiters for sentence parsing.
    text = BOX.sub(lambda m: m.group(1), prose)
    for delimiter in (r'\(', r'\)', '$'):
        text = text.replace(delimiter, '')
    lines = [line.strip().replace('**', '') for line in text.splitlines() if line.strip()]
    final_statement = False
    for index, line in enumerate(lines):
        match = STATEMENT.match(line)
        if not match:
            continue
        tail = line[match.end():].strip()
        # Only a numeric answer followed by optional plain-text units/prose.
        value = re.fullmatch(r'\(?\s*(' + NUMBER + r')\s*\)?(?:\s+[A-Za-z][A-Za-z /-]*)?[.!]?', tail)
        if not value or UNCERTAIN.search(line):
            if index == len(lines) - 1 or re.match(r"^(?:the )?(?:final )?answer\b", line, re.I):
                return None
            continue
        candidates.append(Decimal(value.group(1).replace(',', '')))
        final_statement |= index == len(lines) - 1
    # Require a box in the final nonempty line or an explicit final statement.
    last = [line for line in prose.splitlines() if line.strip()][-1]
    if not final_statement and not BOX.search(last):
        return None
    if UNCERTAIN.search(last):
        return None
    return candidates[0] if candidates and len(set(candidates)) == 1 else None
