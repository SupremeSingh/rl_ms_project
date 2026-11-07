"""VinePPO text splitting and start-character token assignment, without retokenizing actions.

MATH: exact upstream in-place splitter. GSM8K: upstream newline/answer-prefix rules.
Token boundaries are snapped by start-character ownership, matching VinePPO's
broadcast. Empty token chunks are omitted; terminal EOS belongs to the last chunk.
"""
from bisect import bisect_right
from dataclasses import dataclass
import json

MODES = ('token', 'math', 'gsm8k')
DEFAULT_FALLBACK_TOKENS = 64
UPSTREAM_COMMIT = '9f716e68bb934e71da86c226c1699c36b5553faf'


def _upstream_boundaries(text, mode):
    if mode == 'math':
        from ._vendor.vineppo_math import split_solution_inplace
        points = split_solution_inplace(text) if text else [0]
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


def _boundaries_with_status(text, mode):
    if mode not in ('math', 'gsm8k'):
        raise ValueError('Character splitting requires math or gsm8k')
    try:
        return _upstream_boundaries(text, mode), None
    except (AssertionError, ValueError) as exc:
        if mode != 'math':
            raise
        # Upstream's placeholder reconstruction can lose the final boundary even
        # on long responses. Return a coverage marker and reason; token_chunks
        # partitions the ORIGINAL actions into fixed-size groups on this path.
        reason = 'upstream_assertion' if isinstance(exc, AssertionError) else 'invalid_partition'
        return ([0, len(text)] if text else [0]), reason


def character_boundaries(text, mode):
    """Text boundaries; a failed MATH split returns a whole-text coverage marker.

    Fixed-token fallback boundaries are constructed by token_chunks, not here.
    """
    return _boundaries_with_status(text, mode)[0]


@dataclass(frozen=True)
class ChunkPlan:
    starts: tuple[int, ...]
    ends: tuple[int, ...]
    tokens: int
    fallback_reason: str | None = None

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


def _bytelevel_alphabet():
    # Reversible byte alphabet used by GPT-2/Qwen ByteLevel BPE (not UTF-8 text).
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    alphabet = dict(zip(visible, map(chr, visible)))
    extra = 0
    for byte in range(256):
        if byte not in alphabet:
            alphabet[byte] = chr(256 + extra)
            extra += 1
    return {char: byte for byte, char in alphabet.items()}


_BYTE_DECODER = _bytelevel_alphabet()


def _utf8_owners(raw):
    """Map each original byte to its decoded character, including invalid UTF-8.

    UnicodeDecodeError spans implement Python's replacement-character semantics;
    continuation-byte tokens belong to the same character as their leading byte.
    No per-token decoding, which would corrupt split multibyte characters.
    """
    chars, owners, cursor = [], [0] * len(raw), 0

    def append_valid(text, offset):
        for char in text:
            end = offset + len(char.encode('utf-8'))
            owners[offset:end] = [len(chars)] * (end-offset)
            chars.append(char)
            offset = end

    while cursor < len(raw):
        try:
            text = raw[cursor:].decode('utf-8')
        except UnicodeDecodeError as exc:
            start, end = cursor + exc.start, cursor + exc.end
            append_valid(raw[cursor:start].decode('utf-8'), cursor)
            owners[start:end] = [len(chars)] * (end-start)
            chars.append('\ufffd')
            cursor = end
        else:
            append_valid(text, cursor)
            cursor = len(raw)
    return ''.join(chars), owners


def token_character_starts(tokenizer, tokens):
    """Original action-token ownership; never replace IDs with canonical BPE IDs."""
    text = tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    decoder = getattr(getattr(tokenizer, 'backend_tokenizer', None), 'decoder', None)
    state = json.loads(decoder.__getstate__()) if decoder is not None else {}
    if state.get('type') == 'ByteLevel':
        raw, byte_starts = bytearray(), []
        for token in tokenizer.convert_ids_to_tokens(tokens):
            if not isinstance(token, str) or not token:
                raise ValueError('Empty or unknown sampled ByteLevel token')
            byte_starts.append(len(raw))
            for char in token:
                if char in _BYTE_DECODER:
                    raw.append(_BYTE_DECODER[char])
                else:
                    raw.extend(char.encode('utf-8'))
        decoded, owners = _utf8_owners(bytes(raw))
        if decoded != text:
            raise ValueError('Original ByteLevel bytes disagree with tokenizer decoding')
        return text, [owners[i] for i in byte_starts]
    # Other fast tokenizers may use offsets only when they preserve sampled IDs.
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded['input_ids']) != tokens:
        raise ValueError('Response decode/re-encode changed token IDs for unsupported decoder')
    offsets = encoded['offset_mapping']
    if (len(offsets) != len(tokens) or not text
            or any(not 0 <= a < b <= len(text) for a, b in offsets)
            or any(offsets[i][0] > offsets[i+1][0] for i in range(len(offsets)-1))):
        raise ValueError('Invalid tokenizer character offsets')
    return text, [a for a, _ in offsets]


def token_chunks(tokenizer, tokens, mode='math', fallback_tokens=DEFAULT_FALLBACK_TOKENS):
    """Partition sampled actions, using fixed-token groups only on MATH failure.

    Fallback counts all original response tokens, including terminal EOS. The
    final group may be shorter. No token is decoded/re-encoded into a new action.
    """
    if type(fallback_tokens) is not int or fallback_tokens < 1:
        raise ValueError('fallback_tokens must be a positive integer')
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
    text, character_starts = token_character_starts(tokenizer, body)
    points, fallback_reason = _boundaries_with_status(text, mode)
    if fallback_reason is not None:
        starts = list(range(0, len(tokens), fallback_tokens))
        return ChunkPlan(tuple(starts), tuple(starts[1:] + [len(tokens)]), len(tokens), fallback_reason)
    groups = [bisect_right(points, a)-1 for a in character_starts]
    starts = [0] + [i for i in range(1, len(body)) if groups[i] != groups[i-1]]
    return ChunkPlan(tuple(starts), tuple(starts[1:]+[len(tokens)]), len(tokens), fallback_reason)
