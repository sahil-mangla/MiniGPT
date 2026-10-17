import pytest
from minigpt.tokenizer import CharTokenizer, BPETokenizer


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


def test_bpe_training_and_vocab_size():
    corpus = "To be or not to be, that is the question. Whether 'tis nobler in the mind to suffer." * 5
    bpe = BPETokenizer.train_from_text(corpus, target_vocab_size=275)
    assert bpe.vocab_size == 275
    assert len(bpe.merges) == 19  # 275 - 256 = 19 merges


def test_bpe_roundtrip_invertibility():
    corpus = "The quick brown fox jumps over the lazy dog! 1234567890\n\t"
    bpe = BPETokenizer.train_from_text(corpus, target_vocab_size=280)

    test_samples = [
        "The quick brown fox",
        "lazy dog!",
        "unknown word completely unseen in training",
        "Unicode test: café, résumé, Zürich, 漢字",
        "Emoji test: 🤖 🚀 🔥",
        corpus,
    ]

    for sample in test_samples:
        encoded = bpe.encode(sample)
        decoded = bpe.decode(encoded)
        assert decoded == sample, f"Fidelity failed for: {sample}"


def test_bpe_zero_out_of_vocabulary():
    """
    Byte-level BPE must never fail on unseen characters because any
    valid UTF-8 string decomposes into bytes [0..255].
    """
    corpus = "basic english text"
    bpe = BPETokenizer.train_from_text(corpus, target_vocab_size=260)

    # Completely unseen characters and symbols
    unseen_text = "✨ Special character: § ¶ ♠ ♣ 🚀"
    encoded = bpe.encode(unseen_text)
    assert isinstance(encoded, list)
    assert len(encoded) > 0
    assert bpe.decode(encoded) == unseen_text


def test_bpe_compression_ratio():
    corpus = "First Citizen:\nBefore we proceed any further, hear me speak.\n" * 10
    bpe = BPETokenizer.train_from_text(corpus, target_vocab_size=290)

    ratio = bpe.compression_ratio(corpus)
    # Merges must compress text: sequence length of tokens must be smaller than character count
    assert ratio > 1.2
    assert len(bpe.encode(corpus)) < len(corpus)


def test_bpe_empty_string():
    bpe = BPETokenizer()
    assert bpe.encode("") == []
    assert bpe.decode([]) == ""


def test_bpe_invalid_token_id():
    bpe = BPETokenizer()  # base vocab 256
    with pytest.raises(ValueError, match="outside BPE vocabulary range"):
        bpe.decode([9999])
