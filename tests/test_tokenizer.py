import pytest
from minigpt.tokenizer import CharTokenizer


def test_empty_init_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        CharTokenizer("")


def test_vocab_size_and_deterministic_order():
    corpus = "hello world!"
    tok1 = CharTokenizer(corpus)
    tok2 = CharTokenizer(corpus)

    expected_chars = sorted(list(set(corpus)))
    assert tok1.vocab_size == len(expected_chars)
    assert tok1.vocabulary == expected_chars
    # Determinism across instances
    assert tok1.encode("hello") == tok2.encode("hello")


def test_roundtrip_invertibility():
    corpus = "The quick brown fox jumps over the lazy dog. 1234567890 \n\t!@#$%"
    tok = CharTokenizer(corpus)

    test_strings = [
        "The quick brown fox",
        "1234567890",
        "lazy dog.\n!@#$%",
        corpus,
    ]

    for s in test_strings:
        encoded = tok.encode(s)
        decoded = tok.decode(encoded)
        assert decoded == s
        assert isinstance(encoded, list)
        assert all(isinstance(x, int) for x in encoded)


def test_empty_string_roundtrip():
    tok = CharTokenizer("abc")
    assert tok.encode("") == []
    assert tok.decode([]) == ""


def test_unknown_character_raises_value_error():
    tok = CharTokenizer("abc")
    with pytest.raises(ValueError, match="not found in tokenizer vocabulary"):
        tok.encode("abcd")


def test_invalid_token_id_raises_value_error():
    tok = CharTokenizer("abc")  # vocab: 'a':0, 'b':1, 'c':2
    with pytest.raises(ValueError, match="outside vocabulary range"):
        tok.decode([0, 1, 99])

    with pytest.raises(ValueError, match="outside vocabulary range"):
        tok.decode([-1])
