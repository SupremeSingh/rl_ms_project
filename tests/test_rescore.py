import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "rescore", Path(__file__).parents[1] / "scripts/rescore_stage1.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("text,expected", [
    (r"There are \(\boxed{56}\) crayons.", "56"),
    (r"It is \boxed { -1,200.50 }.", "-1200.50"),
    (r"\boxed{4} and \boxed{24}", None),
    (r"\boxed{4} and \boxed{4}", None),
    (r"\boxed{4} and \boxed{", None),
    (r"\boxed{\frac{8}{2}}", None),
    (r"\boxed{NaN}", None),
    (r"\boxed{1,20}", None),
    (r"\boxed{4", None),
    ("Answer: 56", None),
])
def test_extraction(text, expected):
    result = module.extract(text)
    assert (str(result) if result is not None else None) == expected


def test_rescore_and_integrity(tmp_path):
    rows = [{"id": str(i), "prompt_id": "p", "reward": 0,
             "ground_truth": "56", "finish_reason": "stop", "response": text}
            for i, text in enumerate([r"It is \boxed{56}.", r"It is \boxed{64}."])]
    payload = "\n".join(json.dumps(row) for row in rows).encode()
    path = tmp_path / "responses.jsonl"
    path.write_bytes(payload)
    (tmp_path / "metadata.json").write_text(json.dumps({
        "responses_sha256": hashlib.sha256(payload).hexdigest()}))
    report = module.rescore(tmp_path)
    assert report["diagnostic_reward_rate"] == 0.5
    assert report["mixed_pair_rate"] == 1
    assert report["stage1_pass"] is False
    assert path.read_bytes() == payload
    path.write_bytes(payload + b"\n")
    with pytest.raises(ValueError, match="changed"):
        module.rescore(tmp_path)
