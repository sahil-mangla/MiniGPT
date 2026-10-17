# MiniGPT — A GPT Implementation & AI Experimentation Lab

A compact, modular, autoregressive Transformer language model built from scratch in PyTorch, designed for architectural understanding, controlled experiments, and mechanical interpretability.

## Architecture & Design

MiniGPT implements the standard decoder-only Transformer architecture (GPT-2 style) with clean module boundaries:

- **Token & Positional Embeddings**: Learnable token lookup matrix tied with the final language modeling head (`wte.weight is lm_head.weight`), plus absolute learned positional embeddings (`wpe`).
- **Multi-Head Causal Self-Attention**: Batched $Q, K, V$ projections with explicit lower-triangular causal masking ($M_{ij} = -\infty$ for $j > i$) to prevent future information leakage.
- **Transformer Decoder Blocks**: Pre-LayerNorm formulation ($x + \text{Attn}(\text{LN}(x))$, $x + \text{MLP}(\text{LN}(x))$) with GELU non-linearities and residual scaling.
- **Decoupled Weight Decay**: Weight decay applied strictly to 2D matrix weights (Linears, Embeddings), while 1D tensors (LayerNorm scales and biases) have zero weight decay.
- **Apple Silicon Acceleration**: Native support for Metal Performance Shaders (`mps`) backend on Apple Silicon M-series chips.

## Directory Structure

```text
MiniGPT/
├── minigpt/
│   ├── __init__.py
│   ├── config.py         # Strongly-typed configuration (GPTConfig, TrainConfig)
│   ├── tokenizer.py      # Character tokenizer with deterministic vocabularies
│   ├── dataset.py        # Causal sequence slicing and batch sampling
│   ├── model.py          # CausalSelfAttention, MLP, Block, MiniGPT
│   ├── train.py          # Training loop, AdamW optimizer, cosine schedule, checkpointing
│   └── generate.py       # Autoregressive generation (greedy, temperature, top-k)
├── tests/                # Automated pytest suite (18 unit & integration tests)
│   ├── test_tokenizer.py
│   ├── test_dataset.py
│   ├── test_model.py
│   └── test_train.py
├── data/                 # Training text corpora (e.g. input.txt)
├── checkpoints/          # Saved model weights & training metadata
├── pyproject.toml        # Editable package configuration
└── run.py                # Unified CLI runner for training and generation
```

## Setup

```bash
# Clone the repository
git clone https://github.com/sahil-mangla/MiniGPT.git
cd MiniGPT

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies and editable package
pip install -e ".[dev]"
```

## Running Tests

All mathematical invariants (invertibility, deterministic vocabulary ordering, causal masking, target offset alignment) are verified via `pytest`:

```bash
python -m pytest tests/ -v
```

## Training

Train MiniGPT on any text corpus (e.g. `data/input.txt`):

```bash
# Train on Apple Silicon GPU (MPS) or CPU
python run.py train \
    --data data/input.txt \
    --block_size 64 \
    --n_layer 4 \
    --n_head 4 \
    --n_embd 128 \
    --max_iters 500 \
    --lr 1e-3 \
    --batch_size 16
```

## Text Generation

Generate text from a saved checkpoint:

```bash
python run.py generate \
    --checkpoint checkpoints/best_model.pt \
    --prompt "First Citizen:\n" \
    --max_new_tokens 150 \
    --temperature 0.8 \
    --top_k 40
```
