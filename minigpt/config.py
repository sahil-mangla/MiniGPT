import torch
from dataclasses import dataclass


@dataclass(frozen=True)
class GPTConfig:
    """
    Architectural configuration for the MiniGPT model.
    """
    vocab_size: int
    block_size: int = 128      # Context window (sequence length T)
    n_layer: int = 4           # Number of transformer blocks
    n_head: int = 4            # Number of attention heads
    n_embd: int = 128          # Embedding dimension (d_model)
    dropout: float = 0.1       # Dropout probability
    bias: bool = False         # Whether to include bias terms in Linears & LayerNorms
    causal: bool = True        # Causal masking (True for GPT, False for unmasked/bidirectional)

    def __post_init__(self):
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})."
            )


@dataclass
class TrainConfig:
    """
    Hyperparameters and runtime settings for training.
    """
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_iters: int = 2000
    eval_interval: int = 200
    eval_iters: int = 50
    weight_decay: float = 1e-1
    seed: int = 1337
    device: str = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
