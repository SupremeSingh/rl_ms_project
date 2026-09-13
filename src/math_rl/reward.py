import re
from decimal import Decimal

# Accept decimal numbers and correctly grouped thousands separators.
NUMBER = re.compile(r"[+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?")


def parse_number(text):
    text = str(text).strip()
    if not NUMBER.fullmatch(text):
        return None
    return Decimal(text.replace(",", ""))


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    target = parse_number(ground_truth)
    if target is None:
        raise ValueError(f"Invalid ground truth: {ground_truth!r}")
    # The entire final non-empty line must follow our answer contract.
    lines = solution_str.strip().splitlines()
    if not lines:
        return 0.0
    line = lines[-1].strip()
    if not line.startswith("Answer:"):
        return 0.0
    predicted = parse_number(line[len("Answer:"):])
    return float(predicted is not None and predicted == target)


def extract_answer(response, contract="answer"):
    lines = response.strip().splitlines()
    if not lines:
        return None
    final = lines[-1].strip()
    if contract == "answer":
        return parse_number(final[len("Answer:"):]) if final.startswith("Answer:") else None
    if contract == "numeric-box-v1":
        # Reference-independent extraction; semantic relevance requires human audit.
        if response.count(r"\boxed") != 1:
            return None
        match = re.search(r"\\boxed\s*\{([^{}]*)\}", response)
        return parse_number(match.group(1)) if match else None
    if contract != "boxed":
        raise ValueError(f"Unknown answer contract: {contract}")
    # A single numeric box on the final line; never search arbitrary numbers.
    if response.count(r"\boxed") != 1 or "Answer:" in response:
        return None
    match = re.fullmatch(r"\\boxed\{([^{}]+)\}", final)
    return parse_number(match.group(1)) if match else None


def score_contract(response, ground_truth, contract="answer"):
    target = parse_number(ground_truth)
    if target is None:
        raise ValueError(f"Invalid ground truth: {ground_truth!r}")
    predicted = extract_answer(response, contract)
    return float(predicted is not None and predicted == target)
