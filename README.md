# MiniGPT — A Transformer Architecture & AI Experimentation Lab

A compact, modular, autoregressive Transformer language model engineered from first principles in PyTorch. 

MiniGPT is built for **mechanical interpretability, architectural understanding, and controlled empirical experimentation**, serving as the foundational milestone in a 4-project AI deep-dive track:
1. **MiniGPT** (Autoregressive Transformer, BPE, and training dynamics)
2. **AI Game Lab** (Reinforcement learning, reward functions, and policy optimization)
3. **RAG Engine** (Information retrieval, dense embeddings, vector indexing, and reranking)
4. **ORBIT** (Modular agent experimentation harness and evaluation benchmark)

---

## 1. Architectural Decisions & Engineering Principles

Rather than treating the Transformer as a black box or relying on high-level abstractions, MiniGPT implements every component from scratch with explicit design choices:

```text
Input Text  -->  Byte-Level BPE / Char Tokenizer  -->  Token IDs (B, T)
                                                             │
┌────────────────────────── MiniGPT Block ───────────────────┴────────────────────────┐
│                                                                                     │
│  Token Embeddings (wte)  +  Learned Positional Embeddings (wpe)                     │
│  [ Tied Weight Storage: wte.weight is lm_head.weight ]                              │
│                                                                                     │
│  ┌─────────────────────── Transformer Block (x N) ───────────────────────────────┐  │
│  │                                                                               │  │
│  │   x  ──>  LayerNorm (Pre-LN)  ──>  Causal Self-Attention  ──(+)──> x1         │  │
│  │                                    [ Explicit Mask or Fused SDPA ]            │  │
│  │                                                                               │  │
│  │   x1 ──>  LayerNorm (Pre-LN)  ──>  MLP (4x Expansion, GELU) ──(+)──> x2        │  │
│  │                                                                               │  │
│  └───────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                     │
│  Final LayerNorm (ln_f)  ──>  lm_head Linear Projection  ──>  Logits (B, T, V)      │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Key Engineering Decisions
* **Pre-LayerNorm Residual Formulation**: Normalization is applied *before* the attention and feed-forward operations ($x + \text{Attn}(\text{LN}(x))$), providing a direct identity path throughout the residual stream. This eliminates the severe vanishing gradient issues of Post-LN architectures and removes the need for warm-up hacks.
* **Tied Embeddings**: Input token embeddings and final linear projection weights share the exact same underlying tensor memory (`wte.weight is lm_head.weight`). This cuts parameter count significantly and forces representation consistency between token inputs and next-token prediction logits (Press & Wolf, 2017).
* **Decoupled Weight Decay**: Follows modern AdamW best practice: weight decay ($0.1$) is strictly applied to 2D matrix weights (Linear layers, Embeddings), while 1D tensors (LayerNorm scales and biases) have zero weight decay to prevent distortion of normalization scales.
* **Byte-Level Byte-Pair Encoding (BPE)**: Implemented from scratch starting from raw UTF-8 bytes (0..255). This eliminates `<UNK>` out-of-vocabulary exceptions entirely and delivers a **2.4x context compression factor**, allowing 2–3 complete sentences to fit inside a 64-token window.
* **Dual Attention Path (Explicit vs. Fused SDPA)**: Supports both explicit matrix attention for weight inspection and PyTorch fused Scaled Dot-Product Attention (`use_sdpa=True`) for $O(T)$ memory scaling on Apple Silicon Metal (`mps`).

---

## 2. Empirical Laboratory: 5 Controlled Experiments

Every experiment in MiniGPT is designed to answer a specific hypothesis, record controlled metrics, and analyze failure modes.

### Experiment 01: Decoding Dynamics & Sampling Landscape
* **Research Question**: How do temperature ($T$) and top-$k$ truncation affect entropy, vocabulary diversity, and structural degeneration?
* **Key Finding**: At $T \le 0.4$, the model falls into an infinite attractor loop (`athe athe athe...`), collapsing bigram diversity (Distinct-2: 0.17). At $T \ge 1.4$ without top-$k$, entropy approaches theoretical maximum ($H \to \ln(54) \approx 3.99$), producing unpronounceable character noise. Setting $T = 0.7-1.0$ with top-$k = 20$ hits the "Goldilocks zone" of coherent dialogue formatting and lexical diversity.
* **Detailed Report**: [experiments/reports/exp01_sampling_dynamics.md](file:///Users/sahilmangla/MiniGPT/experiments/reports/exp01_sampling_dynamics.md)

### Experiment 02: Causal Mask Ablation & Target Leakage
* **Research Question**: What happens when the lower-triangular causal attention mask is disabled during training and autoregressive generation?
* **Key Finding**: In the non-causal model, **96.9% of attention mass jumped to future tokens ($t+1$)**, learning a trivial lookahead copy operator. At test time, when future tokens do not exist in the context window, the model broke down completely into stuttering subword echo loops (`zezezetinozezezeneze`).
* **Diagnostic Value**: Demonstrates why unnatural loss collapse during early training accompanied by test-time gibberish is the hallmark signature of causal mask inversion or data loader off-by-one indexing.
* **Detailed Report**: [experiments/reports/exp02_causal_ablation.md](file:///Users/sahilmangla/MiniGPT/experiments/reports/exp02_causal_ablation.md)

### Experiment 03: Model Capacity & The Overfitting Frontier
* **Research Question**: At what parameter scale does MiniGPT transition from underfitting to peak generalization, and where does memorization begin?
* **Key Finding**: Swept 4 capacity tiers (57K to 4.76M parameters). While training loss decreased monotonically, the Generalization Gap ($\mathcal{L}_{\text{val}} - \mathcal{L}_{\text{train}}$) exploded from $+0.17$ at Tier 2 up to **$+0.77$ at Tier 4 (948 parameters per training token)**, empirically demonstrating the Chinchilla scaling principle: parameter capacity without corresponding data volume yields memorization rather than generalization.
* **Detailed Report**: [experiments/reports/exp03_capacity_overfitting.md](file:///Users/sahilmangla/MiniGPT/experiments/reports/exp03_capacity_overfitting.md)

### Experiment 04: Subword Tokenization via Byte-Pair Encoding (BPE)
* **Research Question**: How does subword BPE tokenization affect sequence length compression and effective context window capacity?
* **Key Finding**: Training BPE from raw UTF-8 bytes to vocabulary size 500 compressed raw text from 5,569 tokens down to 2,327 tokens (**2.39x compression**). Because self-attention complexity scales as $O(T^2)$, cutting sequence length by $2.39\times$ reduces attention compute and memory footprint by **over 5.7x** for the exact same text passage.
* **Detailed Report**: [experiments/reports/exp04_bpe_compression.md](file:///Users/sahilmangla/MiniGPT/experiments/reports/exp04_bpe_compression.md)

### Experiment 05: Attention Kernel Profiling & Fused SDPA
* **Research Question**: How does PyTorch fused SDPA compare against explicit matrix attention in latency and throughput across context lengths $T \in [64, 512]$ on Apple Silicon?
* **Key Finding**: Both kernels sustain ~70,000–88,000 tokens/sec on Apple Silicon M4 GPU. Crucially, Fused SDPA maintains **$O(T)$ auxiliary memory complexity** instead of $O(T^2)$ by computing attention tiles in fast SRAM/registers without ever materializing the $T \times T$ attention matrix in global unified memory.
* **Detailed Report**: [experiments/reports/exp05_sdpa_profiling.md](file:///Users/sahilmangla/MiniGPT/experiments/reports/exp05_sdpa_profiling.md)

---

## 3. Codebase Structure

```text
MiniGPT/
├── minigpt/
│   ├── __init__.py       # Package exports
│   ├── config.py         # GPTConfig & TrainConfig (dataclasses)
│   ├── tokenizer.py      # Tokenizer ABC, CharTokenizer, BPETokenizer
│   ├── dataset.py        # prepare_data & causal batch generator
│   ├── model.py          # CausalSelfAttention, MLP, Block, MiniGPT
│   ├── train.py          # Training loop, AdamW optimizer, cosine schedule
│   └── generate.py       # Autoregressive generation with temperature & top-k
├── experiments/          # Reproducible experiment drivers and reports
│   ├── exp01_sampling_dynamics.py
│   ├── exp02_causal_ablation.py
│   ├── exp03_capacity_overfitting.py
│   ├── exp04_bpe_compression.py
│   ├── exp05_sdpa_profiling.py
│   └── reports/          # Markdown experimental write-ups
├── scripts/              # Data utilities (e.g. TinyStories corpus generator)
├── tests/                # Automated pytest suite (26 unit and integration tests)
├── data/                 # Training corpora (tinystories.txt, input.txt)
├── checkpoints/          # Serialized model weights and tokenizer metadata
├── pyproject.toml        # Editable package configuration
└── run.py                # Unified CLI runner for training and generation
```

---

## 4. Setup & Verification

### Installation
```bash
git clone https://github.com/sahil-mangla/MiniGPT.git
cd MiniGPT

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install editable package with dev dependencies
pip install -e ".[dev]"
```

### Running the Test Suite
All architectural and mathematical contracts are continuously verified via `pytest`:
```bash
python -m pytest tests/ -v
```
**26 passing tests** validating:
* Invertibility: `decode(encode(text)) == text` across ASCII, code, and emojis.
* Zero Out-of-Vocabulary: Unseen UTF-8 characters cleanly decompose to base bytes.
* Causal Isolation: Perturbing future tokens causes zero change in past logits.
* Weight Tying: `wte.weight is lm_head.weight` points to the exact same tensor storage.
* Numerical Equivalence: Fused SDPA and explicit attention compute identical values within $1\times 10^{-5}$.

---

## 5. Usage: Training & Generation

### 1. Train on TinyStories with Byte-Pair Encoding (BPE)
Train a 4-layer, BPE-backed model with Fused SDPA on Apple Silicon GPU (`mps`):
```bash
python run.py train \
    --data data/tinystories.txt \
    --tokenizer bpe \
    --bpe_vocab_size 400 \
    --use_sdpa \
    --block_size 64 \
    --n_layer 4 \
    --n_head 4 \
    --n_embd 128 \
    --max_iters 500 \
    --lr 1e-3 \
    --batch_size 16 \
    --checkpoint_dir checkpoints/tinystories
```

### 2. Autoregressive Story Generation
Generate text from the trained checkpoint:
```bash
python run.py generate \
    --checkpoint checkpoints/tinystories/best_model.pt \
    --prompt "Once upon a time, there was a little girl named Lily. " \
    --max_new_tokens 150 \
    --temperature 0.7 \
    --top_k 25
```

### 3. Run Experiments Locally
Every experiment can be executed and verified independently:
```bash
PYTHONPATH=. python experiments/exp01_sampling_dynamics.py
PYTHONPATH=. python experiments/exp02_causal_ablation.py
PYTHONPATH=. python experiments/exp03_capacity_overfitting.py
PYTHONPATH=. python experiments/exp04_bpe_compression.py
PYTHONPATH=. python experiments/exp05_sdpa_profiling.py
```
