"""Box-first candidate; unsupported or conflicting answers remain unscored."""
import re
from decimal import Decimal
from math_rl.final_answer import BOX, NUMBER

RULE_VERSION = 'final-numeric-v3'
EXPLICIT = re.compile(r'\b(?:the\s+)?(?:(?:correct|final)\s+)?answer\s*(?:is\s*|:\s*)', re.I)
RELATION = re.compile(r'\b(?:is|are|will have|can make|spent)\s+', re.I)
VALUE = re.compile(r'\(?\s*(' + NUMBER + r')\s*\)?(?:\s+[A-Za-z][A-Za-z /-]*)?[.!]?')
UNCERTAIN = re.compile(r'\b(?:or|maybe|approximately|about|possibly|could|not|instead)\b', re.I)


def numeric_tail(text):
    match = VALUE.fullmatch(text.strip())
    return Decimal(match.group(1).replace(',', '')) if match and not UNCERTAIN.search(text) else None


def extract_final_number(response):
    if response.count('```') % 2:
        return None
    prose = re.sub(r'```.*?```', '', response, flags=re.S).strip()
    boxes = list(BOX.finditer(prose))
    if len(boxes) != prose.count(r'\boxed'):
        return None
    values = []
    for box in boxes:
        content = box.group(1).strip()
        if not re.fullmatch(NUMBER, content):
            return None
        values.append(Decimal(content.replace(',', '')))
    if boxes:
        after = prose[boxes[0].end():]
        if '?' in after or re.search(r'(?im)^\s*(?:question|solution)\s*:', after):
            return None
    text = BOX.sub(lambda m: m.group(1), prose)
    for delimiter in (r'\(', r'\)', r'\[', r'\]', '$', '**'):
        text = text.replace(delimiter, '')
    sentences = [s.strip() for s in re.split(r'(?<!\d)[.!]\s+|(?<=\d)\.(?=\s+[A-Z])|\n+', text) if s.strip()]
    for sentence in sentences:
        for marker in EXPLICIT.finditer(sentence):
            # Statements about checking an answer are not answer declarations.
            prefix = sentence[:marker.start()]
            if re.search(r"\b(?:ensure|check|verify|confirm)\s+(?:that\s+)?$", prefix, re.I):
                continue
            tail = sentence[marker.end():].strip().lstrip(':').strip()
            if not tail:
                continue
            value = numeric_tail(tail)
            if value is None:
                return None
            values.append(value)
    if boxes:
        # Ignore intermediate quantities. Only an explicit final declaration
        # competes with a boxed answer; semantic equivalence still needs auditing.
        last = sentences[-1] if sentences else ''
        if not BOX.search(prose.splitlines()[-1]):
            matches = list(RELATION.finditer(last))
            if matches:
                value = numeric_tail(last[matches[-1].end():])
                if value is not None:
                    values.append(value)
        return values[0] if values and len(set(values)) == 1 else None
    if not sentences:
        return None
    final = sentences[-1]
    markers = list(EXPLICIT.finditer(final)) or list(RELATION.finditer(final))
    if not markers:
        # A clear answer may precede supporting calculations. Do not require it
        # to occupy the last line, but reject a subsequent new question.
        declarations = list(EXPLICIT.finditer(text))
        if not declarations or not values:
            return None
        after = text[declarations[0].end():]
        if '?' in after or re.search(r'(?im)^\s*(?:question|solution)\s*:', after):
            return None
        return values[0] if len(set(values)) == 1 else None
    value = numeric_tail(final[markers[-1].end():])
    if value is None:
        return None
    values.append(value)
    return value if len(set(values)) == 1 else None
