"""Strict decimal/integer reference parsing; generation scoring uses Math-Verify."""
import re
from decimal import Decimal

# Accept decimal numbers and correctly grouped thousands separators.
NUMBER = re.compile(r"[+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?")


def parse_number(text):
    text = str(text).strip()
    if not NUMBER.fullmatch(text):
        return None
    return Decimal(text.replace(",", ""))
