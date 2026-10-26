"""One completion prompt; legacy parquet carriers do not change model inputs."""
from slice_critic.config import MODEL_ID, INSTRUCTION, completion_prompt, immutable_revision

AUDIT_INSTRUCTION = INSTRUCTION
# Loading compatibility only: this carrier is never rendered as a chat prompt.
_LEGACY_CARRIER = (
    "Solve the math problem step by step. End with a line containing only "
    "Answer: <number>, where <number> is a decimal number. "
    "Do not put the final answer in boxed notation or append units or punctuation."
)


def make_prompt(question):
    return [{"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": question}]


def validate_prompt(messages):
    if (len(messages) != 2 or messages[0]['role'] != 'system'
            or messages[0]['content'] not in (INSTRUCTION, _LEGACY_CARRIER)
            or messages[1]['role'] != 'user'):
        raise ValueError('Unsupported completion dataset prompt contract')


def validate_assets(assets):
    if assets.get('model_id') != MODEL_ID or not immutable_revision(assets.get('model_revision')):
        raise ValueError('Expected the pinned Qwen2.5-Math-1.5B snapshot')


def encode_completion(tokenizer, question):
    text = completion_prompt(question)
    return text, tokenizer.encode(text, add_special_tokens=False)
