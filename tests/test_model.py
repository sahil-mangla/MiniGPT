import pytest
import torch
from minigpt.config import GPTConfig
from minigpt.model import MiniGPT, CausalSelfAttention


@pytest.fixture
def small_config():
    return GPTConfig(
        vocab_size=32,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        bias=False,
    )


def test_model_forward_shapes_and_loss(small_config):
    model = MiniGPT(small_config)
    model.eval()

    B, T = 4, 8
    idx = torch.randint(0, small_config.vocab_size, (B, T))
    targets = torch.randint(0, small_config.vocab_size, (B, T))

    # Without targets
    logits, loss = model(idx)
    assert logits.shape == (B, T, small_config.vocab_size)
    assert loss is None

    # With targets
    logits, loss = model(idx, targets=targets)
    assert logits.shape == (B, T, small_config.vocab_size)
    assert loss is not None
    assert loss.dim() == 0  # scalar
    assert not torch.isnan(loss)
    assert loss.item() > 0


def test_weight_tying(small_config):
    model = MiniGPT(small_config)
    # The weight tensors must refer to the exact same storage in memory
    assert model.transformer.wte.weight is model.lm_head.weight


def test_causal_masking_invariant(small_config):
    """
    Core causal invariant test:
    Changing future tokens must have ZERO effect on representations or logits
    of past or current tokens.
    """
    model = MiniGPT(small_config)
    model.eval()

    # Sequence 1: [5, 10, 15, 20]
    seq1 = torch.tensor([[5, 10, 15, 20]], dtype=torch.long)
    # Sequence 2: [5, 10, 99, 88] (positions 0 and 1 are identical, 2 and 3 differ)
    seq2 = torch.tensor([[5, 10, 25, 30]], dtype=torch.long)

    with torch.no_grad():
        logits1, _ = model(seq1)
        logits2, _ = model(seq2)

    # Logits at position 0 must be identical
    assert torch.allclose(logits1[:, 0, :], logits2[:, 0, :], atol=1e-5)
    # Logits at position 1 must be identical
    assert torch.allclose(logits1[:, 1, :], logits2[:, 1, :], atol=1e-5)
    # Logits at positions 2 and 3 should NOT be identical
    assert not torch.allclose(logits1[:, 2, :], logits2[:, 2, :], atol=1e-4)


def test_exceeding_block_size_raises(small_config):
    model = MiniGPT(small_config)
    idx = torch.randint(0, small_config.vocab_size, (1, small_config.block_size + 1))
    with pytest.raises(ValueError, match="Cannot forward sequence of length"):
        model(idx)


def test_autoregressive_generation(small_config):
    model = MiniGPT(small_config)
    model.eval()

    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)  # length 3
    max_new_tokens = 5

    out = model.generate(prompt, max_new_tokens=max_new_tokens, temperature=1.0)
    assert out.shape == (1, 3 + max_new_tokens)
    assert torch.equal(out[:, :3], prompt)  # prompt prefix is preserved

    # Greedy generation test (temperature=0.0)
    out_greedy_1 = model.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0)
    out_greedy_2 = model.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0)
    assert torch.equal(out_greedy_1, out_greedy_2)  # must be deterministic


def test_bidirectional_attention_allows_future_leakage():
    """
    Verify that when causal=False, future tokens DO influence past logits.
    """
    config = GPTConfig(
        vocab_size=32,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        causal=False,
    )
    model = MiniGPT(config)
    model.eval()

    seq1 = torch.tensor([[5, 10, 15, 20]], dtype=torch.long)
    seq2 = torch.tensor([[5, 10, 25, 30]], dtype=torch.long)

    with torch.no_grad():
        logits1, _ = model(seq1)
        logits2, _ = model(seq2)

    # With causal=False, position 0 logits MUST change because position 2 and 3 tokens changed
    assert not torch.allclose(logits1[:, 0, :], logits2[:, 0, :], atol=1e-4)


def test_sdpa_and_explicit_attention_equivalence():
    """
    Verify that fused SDPA and explicit matrix attention compute mathematically
    identical output representations when given identical weights.
    """
    torch.manual_seed(42)
    config_explicit = GPTConfig(
        vocab_size=32,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        causal=True,
        use_sdpa=False,
    )
    config_sdpa = GPTConfig(
        vocab_size=32,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        causal=True,
        use_sdpa=True,
    )

    model_explicit = MiniGPT(config_explicit)
    model_sdpa = MiniGPT(config_sdpa)

    # Copy identical weights
    model_sdpa.load_state_dict(model_explicit.state_dict())

    model_explicit.eval()
    model_sdpa.eval()

    idx = torch.randint(0, 32, (2, 10))

    with torch.no_grad():
        out_explicit, _ = model_explicit(idx)
        out_sdpa, _ = model_sdpa(idx)

    # Must be numerically identical within floating point tolerance
    assert torch.allclose(out_explicit, out_sdpa, atol=1e-5)
