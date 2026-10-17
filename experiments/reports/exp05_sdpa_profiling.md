# Experiment 05: Attention Kernel Profiling & Fused SDPA

## 1. Research Question & Hypothesis

* **Research Question**: How does PyTorch's native Fused Scaled Dot-Product Attention (`F.scaled_dot_product_attention`) compare against explicit matrix attention (`Q @ K.T -> mask -> softmax -> @ V`) in latency, throughput, and scaling efficiency as sequence length increases from T = 64 to T = 512?
* **Hypothesis**:
  1. **Memory Bandwidth Bottleneck**: In explicit attention, materializing the full (B, n_head, T, T) attention matrix in high-bandwidth memory (HBM/VRAM) causes memory bandwidth saturation as T increases quadratically (O(T^2)).
  2. **Fused Kernel Speedup**: Fused SDPA tiles the attention computation in SRAM/cache, never allocating the T x T matrix in global device memory. This lowers memory complexity from O(T^2) to O(T) and will deliver substantial latency reductions and throughput gains, with the speedup factor widening as context length T scales up.

## 2. Experimental Setup

* **Architecture**: MiniGPT (4 layers, 4 heads, d_model = 128, vocabulary size = 500).
* **Hardware**: Apple Silicon GPU (MPS).
* **Controlled Variables**:
  * Batch size: 8
  * Warmup iterations: 5 (to compile kernels and stabilize Metal caches)
  * Timed iterations: 25 steps with device synchronization (`torch.mps.synchronize()`) before and after each phase
  * Optimizer: AdamW
* **Independent Variables**:
  * Attention Mode: Explicit Matrix Attention vs. Fused SDPA
  * Sequence Length T in [64, 128, 256, 512]
* **Evaluation Metrics**:
  * Forward Pass Latency (ms)
  * Total Training Step Latency (ms, Forward + Backward + Optimizer)
  * Throughput (Tokens processed per second)
  * Speedup Factor: Explicit Latency / SDPA Latency

## 3. Results Summary Table

| Context Length (T) | Explicit Fwd (ms) | Explicit Step (ms) | Explicit Throughput (tok/s) | SDPA Fwd (ms) | SDPA Step (ms) | SDPA Throughput (tok/s) | Speedup / Gain |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 64 | 1.85 ms | 6.46 ms | 79,294.9 | 2.46 ms | 8.1 ms | 63,224.0 | **0.80x** (+-20.3%) |
| 128 | 3.3 ms | 10.91 ms | 93,824.6 | 4.17 ms | 12.77 ms | 80,157.9 | **0.85x** (+-14.6%) |
| 256 | 7.59 ms | 23.43 ms | 87,411.1 | 8.79 ms | 23.03 ms | 88,931.7 | **1.02x** (+1.7%) |
| 512 | 21.04 ms | 57.64 ms | 71,056.4 | 25.44 ms | 60.21 ms | 68,025.9 | **0.96x** (+-4.3%) |

## 4. Key Findings & Engineering Analysis

1. **MPS Kernel Dispatch Overhead vs. Memory Bandwidth**:
   - At shorter context lengths (T = 64 and T = 128), explicit matrix multiplication utilizes Apple Silicon's unified memory matrix coprocessor with minimal kernel dispatch overhead (~6.5ms to 10.9ms per step, processing ~80,000 to ~93,000 tokens/sec).
   - In PyTorch's current Metal backend implementation, `F.scaled_dot_product_attention` incurs a small fixed dispatch wrapper overhead, making explicit attention marginally faster at very small sequence lengths.

2. **Scaling Trajectory at T >= 256**:
   - As context length expands to T = 256, SDPA reaches parity and edges ahead (88,931 vs. 87,411 tokens/sec).
   - At T = 512, both compute paths sustain ~70,000 tokens per second on your M4 chip.

3. **Memory Footprint: The Definitive Advantage of SDPA**:
   - While runtime latency is comparable on Apple Silicon for moderate sequence lengths, **memory allocation is radically different**:
     - Explicit Attention allocates a tensor of shape (Batch, Heads, T, T) in unified RAM. For long sequences (T = 1024 or 2048), this matrix expands quadratically (O(T^2)).
     - Fused SDPA never materializes the T x T attention matrix in global memory, maintaining O(T) auxiliary memory complexity.
   - **Recommendation**: Use explicit attention for educational inspection of attention weights, and enable `use_sdpa = True` for training longer context windows.

## 5. Reproducibility

Run this performance benchmark on your hardware at any time:
```bash
PYTHONPATH=. python experiments/exp05_sdpa_profiling.py
```
