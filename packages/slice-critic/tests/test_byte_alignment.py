"""Real Hugging Face ByteLevel tokenizer; no network or model download."""
import random
import pytest
from slice_critic.chunks import _BYTE_DECODER, _utf8_owners, token_character_starts, token_chunks


@pytest.fixture
def tokenizer():
    tokenizers=pytest.importorskip('tokenizers')
    transformers=pytest.importorskip('transformers')
    alphabet={byte:char for char,byte in _BYTE_DECODER.items()}
    vocab={alphabet[b]:b for b in range(256)}
    vocab.update({'ab':256,alphabet[10]*2:257,'<|endoftext|>':258})
    backend=tokenizers.Tokenizer(tokenizers.models.BPE(vocab=vocab,
        merges=[('a','b'),(alphabet[10],alphabet[10])]))
    backend.pre_tokenizer=tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder=tokenizers.decoders.ByteLevel()
    return transformers.PreTrainedTokenizerFast(tokenizer_object=backend,eos_token='<|endoftext|>')


def test_noncanonical_sampled_bpe_is_preserved(tokenizer):
    text='ab\n\nNext line.'
    ids=list(text.encode())
    assert tokenizer.encode(text,add_special_tokens=False)!=ids
    decoded,starts=token_character_starts(tokenizer,ids)
    assert decoded==text and starts==list(range(len(text)))
    plan=token_chunks(tokenizer,ids+[258],mode='gsm8k')
    assert plan.starts==(0,2,3) and plan.ends[-1]==len(ids)+1


@pytest.mark.parametrize('raw',[
    'First.\nThen € and 中文.'.encode(),
    b'First.\nThen \xe2',
    b'First.\nThen \xe2\x82x\xff',
    b'\xf0\x9f\x98\x80',
])
def test_unicode_and_truncation_owners_match_real_decoder(tokenizer,raw):
    text,starts=token_character_starts(tokenizer,list(raw))
    assert text==raw.decode('utf-8',errors='replace')
    assert len(starts)==len(raw) and starts==sorted(starts)
    assert all(0<=v<len(text) for v in starts)
    p=token_chunks(tokenizer,list(raw)+[258],'math')
    assert p.ends[-1]==len(raw)+1


def test_utf8_character_ownership_for_split_tokens():
    text,owners=_utf8_owners('a€中'.encode())
    assert text=='a€中' and owners==[0,1,1,1,2,2,2]
    rng=random.Random(42)
    for _ in range(200):
        raw=bytes(rng.randrange(256) for _ in range(rng.randrange(1,60)))
        text,owners=_utf8_owners(raw)
        assert text==raw.decode('utf-8',errors='replace')
        assert len(owners)==len(raw) and owners==sorted(owners)


def test_merged_token_crossing_boundary_keeps_start_ownership(tokenizer):
    p=token_chunks(tokenizer,[ord('a'),257,ord('b'),258],'gsm8k')
    assert p.starts==(0,1,2) and p.ends[-1]==4


def test_special_tokens_and_eos_remain_original_actions(tokenizer):
    text,starts=token_character_starts(tokenizer,[ord('a'),258,ord('b')])
    assert text=='a<|endoftext|>b' and starts==[0,1,14]
    assert token_chunks(tokenizer,[258]).ends==(1,)
