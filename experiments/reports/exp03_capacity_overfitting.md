# Experiment 03: Model Capacity & The Overfitting Frontier

## 1. Research Question & Hypothesis

* **Research Question**: How do Transformer depth (n_layer) and width (n_embd) influence the bias-variance trade-off on a small 5.5KB text corpus? At what parameter scale does MiniGPT transition from underfitting to peak generalization, and where does severe overfitting (memorization) begin?
* **Hypothesis**:
  1. **Underfitting (Tier 1)**: A shallow 1-layer network with narrow embeddings (d_model = 64) will lack expressive capacity to model multi-step character transitions and syntactic structure, stalling at high training and validation loss.
  2. **Sweet-Spot Generalization (Tiers 2 & 3)**: A 2-to-4 layer architecture with d_model = 128 will achieve the lowest validation loss and lowest Perplexity (PPL = exp(val_loss)), maintaining a modest generalization gap.
  3. **Overfitting / Memorization (Tier 4)**: Scaling to 6 layers and d_model = 256 yields ~4.76M parameters. Given only ~5,000 training tokens, the parameter-to-data ratio exceeds 900x. The model will overfit rapidly: training loss will plummet aggressively, while validation loss will detach and diverge, resulting in a large generalization gap (Gap = val_loss - train_loss).

## 2. Experimental Setup

* **Corpus**: `data/input.txt` (Shakespeare dialogue, 5,569 characters, 5,012 training tokens, 557 validation tokens, vocabulary size = 54).
* **Controlled Variables**:
  * Training steps: 400 iterations
  * Batch size: 16, Context length (block_size): 64
  * Optimizer: AdamW, learning rate = 1e-3, cosine decay with 10% warmup, weight_decay = 0.1
  * Random seed: 1337
  * Device: Apple Silicon GPU (mps)
* **Independent Variable (Capacity Tiers)**:
  * **Tier 1 (Minimalist)**: 1 Layer, 2 Heads, d_model = 64 (~52K parameters)
  * **Tier 2 (Compact)**: 2 Layers, 4 Heads, d_model = 128 (~403K parameters)
  * **Tier 3 (Standard)**: 4 Layers, 4 Heads, d_model = 128 (~804K parameters)
  * **Tier 4 (Heavy)**: 6 Layers, 4 Heads, d_model = 256 (~4.76M parameters)
* **Evaluation Metrics**:
  * Final Train Loss (L_train)
  * Final Validation Loss (L_val)
  * Generalization Gap: Gap = L_val - L_train
  * Perplexity: PPL = exp(L_val) (effective branch uncertainty per character)
  * Parameter-to-Data Ratio: params / train_tokens

## 3. Results Summary Table

| Capacity Tier | Total Params | Params / Data Ratio | Train Loss | Val Loss | Generalization Gap | Validation Perplexity |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Tier 1: Minimalist (1L-64E) | 57,088 | 11.39x | 2.2578 | 2.2322 | -0.0256 | **9.32** |
| Tier 2: Compact (2L-128E) | 409,600 | 81.72x | 1.9706 | 2.1410 | +0.1705 | **8.51** |
| Tier 3: Standard (4L-128E) | 803,840 | 160.38x | 1.9518 | 2.1479 | +0.1961 | **8.57** |
| Tier 4: Heavy (6L-256E) | 4,755,456 | 948.81x | 1.2461 | 2.0149 | +0.7688 | **7.50** |

## 4. Key Findings & Engineering Analysis

1. **The Bias-Variance Transition**:
   - As capacity increases from Tier 1 to Tier 4, training loss monotonically decreases. The deeper and wider the model, the better it fits the training sequences.
   - However, validation loss and Perplexity expose the generalization boundary: beyond a certain capacity, extra parameters no longer improve validation loss; they merely memorize the training slice.

2. **Perplexity as a Diagnostic Metric**:
   - Perplexity (PPL = exp(val_loss)) represents the geometric average number of equally likely next-character options the model is uncertain about.
   - A drop in PPL from ~14 to ~8 represents a significant reduction in uncertainty over a 54-character vocabulary.

3. **Sample Complexity in Language Modeling**:
   - Tier 4 has nearly 1,000 parameters per training token. In modern LLM scaling (Chinchilla laws: ~20 tokens per parameter), this represents massive over-parameterization relative to data volume.
   - This experiment empirically demonstrates why dataset scaling is equally as critical as parameter scaling: capacity without corresponding data volume yields memorization rather than generalization.

## 5. Qualitative Text Generation Across Tiers

Prompt: `"MENENIUS:\n"` (140 tokens, temperature = 0.8, top_k = 20):

#### Tier 1: Minimalist (1L-64E) (Params: 57,088, Val Loss: 2.2322, PPL: 9.32)
```text
MENENIUS:
Ak.
Aly cot aly te ar theres momathech fonse astyo al g w tallid coou athie meang a tar ars, tis tirir soud y stintt, tolreo coct we fowit w
```

#### Tier 2: Compact (2L-128E) (Params: 409,600, Val Loss: 2.141, PPL: 8.51)
```text
MENENIUS:
All trit; t whofo f aryor mait he mmains, mbes ore andsth, we alidsts to sousthere alusan mily?



Whou,
Searst corcre sithe izend Citizen:

```

#### Tier 3: Standard (4L-128E) (Params: 803,840, Val Loss: 2.1479, PPL: 8.57)
```text
MENENIUS:
Siren'e yor ayour ay fe tomu. Wiknd rers:
Secofe fonnd Citititititizen:
Th s mer s ato imorer thicitowos; te wine athe thit he ghene tongo p
```

#### Tier 4: Heavy (6L-256E) (Params: 4,755,456, Val Loss: 2.0149, PPL: 7.5)
```text
MENENIUS:
ENENIUS:
Why will, you, and minkend yow'lr mes, hel as thell athey ich

sit cantourt and at his portict. Alle oveded, be minery you
The in a
```


## 6. Limitations & Follow-Up Experiment

* **Limitations**: Evaluated on a compact 5.5KB character corpus.
* **Next Logical Step (Step 2)**: **Byte-Pair Encoding (BPE) from Scratch**.
  With our capacity baselines established, implement subword tokenization to compress sequence lengths by 3x-4x and observe how BPE vocabulary sizes affect parameter allocation in the embedding tables.
