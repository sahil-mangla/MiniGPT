import math
import os
import time
import torch
from minigpt.config import GPTConfig, TrainConfig
from minigpt.dataset import get_batch, prepare_data
from minigpt.model import MiniGPT
from minigpt.tokenizer import Tokenizer, CharTokenizer, BPETokenizer


def configure_optimizers(
    model: MiniGPT, weight_decay: float, learning_rate: float, betas: tuple[float, float] = (0.9, 0.95)
) -> torch.optim.AdamW:
    """
    Separate parameters into those that decay and those that do not:
    - 2D parameters (weights of Linear and Embedding) decay.
    - 1D parameters (biases and LayerNorm scales/biases) do not decay.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    return optimizer


def get_lr(
    it: int,
    max_iters: int,
    warmup_iters: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Cosine learning rate decay with linear warmup."""
    if it < warmup_iters:
        return max_lr * (it + 1) / (warmup_iters + 1)
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / max(1, (max_iters - warmup_iters))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def estimate_loss(
    model: MiniGPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_iters: int,
    device: str,
) -> dict[str, float]:
    """Estimate average loss across multiple batches for train and val splits."""
    model.eval()
    out = {}
    for split_name, split_data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split_data, block_size, batch_size, device=device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split_name] = losses.mean().item()
    model.train()
    return out


def train(
    model: MiniGPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    tokenizer: CharTokenizer,
    model_config: GPTConfig,
    train_config: TrainConfig,
    checkpoint_dir: str = "checkpoints",
) -> dict[str, list]:
    """
    Execute full training loop with validation, learning rate scheduling,
    and checkpointing.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = train_config.device
    model = model.to(device)

    optimizer = configure_optimizers(
        model,
        weight_decay=train_config.weight_decay,
        learning_rate=train_config.learning_rate,
    )

    warmup_iters = int(train_config.max_iters * 0.1)
    min_lr = train_config.learning_rate * 0.1
    best_val_loss = float("inf")

    history = {"step": [], "train_loss": [], "val_loss": [], "lr": []}

    print(f"Training MiniGPT on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")

    t0 = time.time()
    for iter_num in range(train_config.max_iters):
        # Update learning rate per schedule
        lr = get_lr(
            iter_num,
            train_config.max_iters,
            warmup_iters,
            train_config.learning_rate,
            min_lr,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Periodic evaluation & checkpointing
        if iter_num % train_config.eval_interval == 0 or iter_num == train_config.max_iters - 1:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                model_config.block_size,
                train_config.batch_size,
                train_config.eval_iters,
                device,
            )
            elapsed = time.time() - t0
            print(
                f"step {iter_num:4d} | train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | lr {lr:.2e} | time {elapsed:.2f}s"
            )

            history["step"].append(iter_num)
            history["train_loss"].append(losses["train"])
            history["val_loss"].append(losses["val"])
            history["lr"].append(lr)

            # Checkpoint the best model
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
                is_bpe = isinstance(tokenizer, BPETokenizer)
                torch.save(
                    {
                        "step": iter_num,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "model_config": model_config,
                        "train_config": train_config,
                        "val_loss": best_val_loss,
                        "tokenizer_type": "bpe" if is_bpe else "char",
                        "merges": tokenizer.merges if is_bpe else None,
                        "chars": tokenizer.vocabulary if not is_bpe else None,
                    },
                    checkpoint_path,
                )

        # Sample a batch and perform forward & backward pass
        x, y = get_batch(
            train_data,
            model_config.block_size,
            train_config.batch_size,
            device=device,
        )

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    print(f"Training completed. Best val loss: {best_val_loss:.4f}")
    return history
