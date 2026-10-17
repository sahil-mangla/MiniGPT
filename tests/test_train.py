import os
import pytest
import torch
from minigpt.config import GPTConfig, TrainConfig
from minigpt.tokenizer import CharTokenizer
from minigpt.dataset import prepare_data
from minigpt.model import MiniGPT
from minigpt.train import configure_optimizers, get_lr, train
from minigpt.generate import load_model_from_checkpoint, generate_text


def test_optimizer_parameter_groups():
    config = GPTConfig(vocab_size=16, block_size=8, n_layer=1, n_head=1, n_embd=16, bias=True)
    model = MiniGPT(config)
    optimizer = configure_optimizers(model, weight_decay=0.1, learning_rate=1e-3)

    decay_group = optimizer.param_groups[0]
    no_decay_group = optimizer.param_groups[1]

    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0

    # Ensure all decay params are >= 2D
    for p in decay_group["params"]:
        assert p.dim() >= 2

    # Ensure all no_decay params are 1D (biases, layer norms)
    for p in no_decay_group["params"]:
        assert p.dim() < 2


def test_lr_schedule():
    max_iters = 100
    warmup_iters = 10
    max_lr = 1e-3
    min_lr = 1e-4

    # Warmup should increase linearly
    lr_0 = get_lr(0, max_iters, warmup_iters, max_lr, min_lr)
    lr_warm = get_lr(warmup_iters - 1, max_iters, warmup_iters, max_lr, min_lr)
    assert lr_0 < lr_warm

    # At completion or past max_iters, lr equals min_lr
    lr_end = get_lr(max_iters + 5, max_iters, warmup_iters, max_lr, min_lr)
    assert lr_end == min_lr


def test_end_to_end_training_and_generation(tmp_path):
    corpus = "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer."
    tokenizer = CharTokenizer(corpus)
    train_data, val_data = prepare_data(corpus, tokenizer, train_split=0.8)

    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
    )
    model = MiniGPT(model_config)

    train_config = TrainConfig(
        batch_size=2,
        learning_rate=1e-3,
        max_iters=10,
        eval_interval=5,
        eval_iters=2,
        device="cpu",
    )

    checkpoint_dir = str(tmp_path / "checkpoints")
    history = train(
        model=model,
        train_data=train_data,
        val_data=val_data,
        tokenizer=tokenizer,
        model_config=model_config,
        train_config=train_config,
        checkpoint_dir=checkpoint_dir,
    )

    # Validate history
    assert len(history["step"]) >= 2
    assert os.path.exists(os.path.join(checkpoint_dir, "best_model.pt"))

    # Test loading and generation
    loaded_model, loaded_tokenizer = load_model_from_checkpoint(
        os.path.join(checkpoint_dir, "best_model.pt"), device="cpu"
    )
    generated = generate_text(
        loaded_model, loaded_tokenizer, prompt="To be", max_new_tokens=10, device="cpu"
    )

    assert isinstance(generated, str)
    assert generated.startswith("To be")
    assert len(generated) == len("To be") + 10
