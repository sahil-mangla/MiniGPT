import torch
from typing import Tuple
from minigpt.tokenizer import Tokenizer


def prepare_data(
    text: str, tokenizer: Tokenizer, train_split: float = 0.9
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode an entire raw text corpus into a 1D LongTensor and split into train/val sets.
    """
    if not (0.0 < train_split < 1.0):
        raise ValueError(f"train_split must be between 0.0 and 1.0, got {train_split}")

    encoded = tokenizer.encode(text)
    data = torch.tensor(encoded, dtype=torch.long)

    n_train = int(len(data) * train_split)
    train_data = data[:n_train]
    val_data = data[n_train:]

    return train_data, val_data


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str = "cpu",
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a random batch of inputs x and targets y for next-token prediction.

    Invariants:
      x shape: (batch_size, block_size)
      y shape: (batch_size, block_size)
      For every position t in [0, block_size-2]: y[b, t] == x[b, t+1]
      y[b, block_size-1] is the subsequent token right after the input sequence.
    """
    if len(data) <= block_size:
        raise ValueError(
            f"Dataset length ({len(data)}) must be strictly greater than block_size ({block_size})."
        )

    # Sample random start indices
    max_idx = len(data) - block_size
    ix = torch.randint(0, max_idx, (batch_size,), generator=generator)

    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

    if device != "cpu":
        x = x.to(device)
        y = y.to(device)

    return x, y
