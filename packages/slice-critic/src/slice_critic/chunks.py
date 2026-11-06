"""VinePPO text splitting and start-character token assignment, without retokenizing actions.

MATH: exact upstream in-place splitter. GSM8K: upstream newline/answer-prefix rules.
Token boundaries are snapped by start-character ownership, matching VinePPO's
broadcast. Empty token chunks are omitted; terminal EOS belongs to the last chunk.
"""
from bisect import bisect_right
from dataclasses import dataclass

MODES = ('token', 'math', 'gsm8k')
UPSTREAM_COMMIT = '9f716e68bb934e71da86c226c1699c36b5553faf'


def character_boundaries(text, mode):
    if mode == 'math':
        from ._vendor.vineppo_math import split_solution_inplace
        try:
            points = split_solution_inplace(text) if text else [0]
        except AssertionError as exc:
            # Upstream can remove its only part on short/blank responses.
            if len(text) < 20 or not text.strip():
                return [0, len(text)] if text else [0]
            raise ValueError('VinePPO MATH segmentation failed') from exc
    elif mode == 'gsm8k':
        prefix = '\n#### '
        parts = text.split(prefix)
        body, answer = parts if len(parts) == 2 else (text, None)
        steps = body.split('\n')
        nonempty = next((i for i, s in enumerate(steps) if s.strip()), None)
        if nonempty is not None and nonempty > 0:
            steps = ['\n'.join(steps[:nonempty+1])] + steps[nonempty+1:]
        if answer is not None:
            last = next((i for i in range(len(steps)-1, -1, -1) if steps[i].strip()), 0)
            steps = steps[:last] + ['\n'.join(steps[last:]) + prefix + answer]
        points = [0]
        for i, step in enumerate(steps):
            points.append(points[-1] + len(step) + (1 if i else 0))
    else:
        raise ValueError('Character splitting requires math or gsm8k')
    points = sorted(set(points))
    if not points or points[0] != 0 or points[-1] != len(text):
        raise ValueError('Splitter did not preserve the complete response')
    return points


@dataclass(frozen=True)
class ChunkPlan:
    starts: tuple[int, ...]
    ends: tuple[int, ...]
    tokens: int

    def __post_init__(self):
        if (self.tokens < 1 or not self.starts or self.starts[0] != 0
                or len(self.starts) != len(self.ends) or self.ends[-1] != self.tokens
                or any(a >= b for a, b in zip(self.starts, self.ends))
                or self.ends[:-1] != self.starts[1:]):
            raise ValueError('Chunks must partition every response token exactly once')

    def expand(self, values):
        import torch
        if len(values) != len(self.starts):
            raise ValueError('One value per chunk required')
        return torch.cat([values[i:i+1].expand(end-start)
                          for i, (start, end) in enumerate(zip(self.starts, self.ends))])


def token_chunks(tokenizer, tokens, mode='math'):
    tokens = list(tokens)
    if not tokens or mode not in MODES:
        raise ValueError('Nonempty tokens and a supported chunk mode required')
    if mode == 'token':
        return ChunkPlan(tuple(range(len(tokens))), tuple(range(1, len(tokens)+1)), len(tokens))
    if not getattr(tokenizer, 'is_fast', False):
        raise ValueError('Chunk alignment requires a fast tokenizer with offsets')
    has_eos = tokens[-1] == tokenizer.eos_token_id
    body = tokens[:-1] if has_eos else tokens
    if not body:
        return ChunkPlan((0,), (len(tokens),), len(tokens))
    text = tokenizer.decode(body, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded['input_ids']) != body:
        raise ValueError('Response decode/re-encode changed token IDs; refusing to alter sampled actions')
    offsets = encoded['offset_mapping']
    if (len(offsets) != len(body) or not text
            or any(not 0 <= a < b <= len(text) for a, b in offsets)
            or any(offsets[i][0] > offsets[i+1][0] for i in range(len(offsets)-1))):
        raise ValueError('Invalid tokenizer character offsets')
    points = character_boundaries(text, mode)
    groups = [bisect_right(points, a)-1 for a, _ in offsets]
    starts = [0] + [i for i in range(1, len(body)) if groups[i] != groups[i-1]]
    return ChunkPlan(tuple(starts), tuple(starts[1:]+[len(tokens)]), len(tokens))
