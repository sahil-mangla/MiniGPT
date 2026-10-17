import pytest
import torch
from minigpt.tokenizer import CharTokenizer
from minigpt.dataset import prepare_data, get_batch


def test_prepare_data_splits_and_types():
    text = "abcdefghijklmnopqrstuvwxyz" * 10  # 260 chars
    tokenizer = CharTokenizer(text)

    train_data, val_data = prepare_data(text, tokenizer, train_split=0.8)

    assert isinstance(train_data, torch.Tensor)
    assert isinstance(val_data, torch.Tensor)
    assert train_data.dtype == torch.long
    assert val_data.dtype == torch.long

    total_len = len(text)
    expected_train_len = int(total_len * 0.8)
    assert len(train_data) == expected_train_len
    assert len(val_data) == total_len - expected_train_len


def test_prepare_data_invalid_split():
    tokenizer = CharTokenizer("abc")
    with pytest.raises(ValueError):
        prepare_data("abc", tokenizer, train_split=1.5)
    with pytest.raises(ValueError):
        prepare_data("abc", tokenizer, train_split=-0.1)


def test_get_batch_shapes_and_alignment():
    # Deterministic sequence: 0, 1, 2, ..., 99
    data = torch.arange(100, dtype=torch.long)
    block_size = 8
    batch_size = 4

    g = torch.Generator().manual_seed(42)
    x, y = get_batch(data, block_size=block_size, batch_size=batch_size, generator=g)

    assert x.shape == (batch_size, block_size)
    assert y.shape == (batch_size, block_size)
    assert x.dtype == torch.long
    assert y.dtype == torch.long

    # Core causal invariant: target sequence is strictly offset by +1
    # For every batch item, y[b, :-1] must match x[b, 1:]
    assert torch.equal(x[:, 1:], y[:, :-1])

    # Also verify that y is indeed data[i+1 : i+block_size+1]
    for b in range(batch_size):
        first_input_token = x[b, 0].item()
        assert y[b, 0].item() == first_input_token + 1
        assert y[b, -1].item() == x[b, -1].item() + 1


def test_get_batch_too_short_raises():
    data = torch.tensor([1, 2, 3], dtype=torch.long)
    with pytest.raises(ValueError, match="strictly greater than block_size"):
        get_batch(data, block_size=4, batch_size=2)
