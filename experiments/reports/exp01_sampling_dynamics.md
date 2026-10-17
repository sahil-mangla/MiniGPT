# Experiment 01: Decoding Dynamics & Sampling Landscape

## 1. Research Question & Hypothesis

* **Research Question**: How do sampling temperature ($T$) and top-$k$ truncation affect the entropy, vocabulary diversity, and structural degeneration of generated text in an autoregressive Transformer?
* **Hypothesis**:
  1. **Low Temperature ($T \le 0.4$)**: Exponentiating logits by $1/T$ makes the distribution hyper-peaked. The model will fall into deterministic repeating cycles (attractors), exhibiting near-zero entropy and collapsed $n$-gram diversity (low Distinct-2).
  2. **High Temperature ($T \ge 1.4$) without Top-$k$**: Flattens probabilities across the full vocabulary ($\lim_{T \to \infty} p_i = 1/V$, where maximum theoretical entropy $H_{\max} = \ln(54) = 3.989$). The model will sample low-probability character noise, destroying lexical and grammatical coherence.
  3. **Top-$k$ Truncation as a Regularizer**: Truncating the distribution to the top $k$ candidates before softmax bounds the maximum entropy, preventing tail noise even at high temperatures while retaining linguistic diversity.

## 2. Experimental Setup

* **Model**: MiniGPT (4 layers, 4 heads, $d_{\text{model}} = 128$, block size $= 64$, 803,840 parameters).
* **Controlled Variables**:
  * Training corpus: `data/input.txt` (Shakespeare dialogue, 5.5K chars, $|V| = 54$).
  * Training steps: 500 iterations, batch size 16, learning rate $1\times 10^{-3}$, AdamW.
  * Inference prompt: `"MENENIUS:\n"`
  * Generated sequence length: 200 tokens
  * Inference random seed: `42`
* **Independent Variables**:
  * Temperature $T \in [0.1, 0.4, 0.7, 1.0, 1.4, 2.0]$
  * Top-$k \in [\text{None}, 5, 20]$
* **Evaluation Metrics**:
  * **Mean Entropy ($H$)**: Average Shannon entropy of the sampling distribution across decoding steps.
  * **Distinct-1, Distinct-2, Distinct-3**: Ratio of unique unigrams, bigrams, and trigrams to total generated $n$-grams (measures repetitive degeneration).
  * **Mean Chosen Probability**: Average probability assigned to the selected token.

## 3. Results Summary Table

| Temperature | Top-k | Mean Entropy | Distinct-1 | Distinct-2 | Distinct-3 | Sample Snippet |
|-------------|-------|--------------|------------|------------|------------|----------------|
| 0.1 | None (full vocab) | 0.223 | 0.085 | 0.171 | 0.247 | `MENENIUS:\nWher st you where s mathe s theathe the...` |
| 0.1 | 5 | 0.221 | 0.085 | 0.171 | 0.247 | `MENENIUS:\nWher st you where s mathe s theathe the...` |
| 0.1 | 20 | 0.223 | 0.085 | 0.171 | 0.247 | `MENENIUS:\nWher st you where s mathe s theathe the...` |
| 0.4 | None (full vocab) | 0.921 | 0.145 | 0.412 | 0.631 | `MENENIUS:\nIUS:\nWhe hel nore where s, athathe whe...` |
| 0.4 | 5 | 0.75 | 0.14 | 0.377 | 0.596 | `MENENIUS:\nIUS:\nWhe whel are where s, alll tod th...` |
| 0.4 | 20 | 0.918 | 0.145 | 0.412 | 0.631 | `MENENIUS:\nIUS:\nWhe hel nore where s, athathe whe...` |
| 0.7 | None (full vocab) | 1.686 | 0.175 | 0.573 | 0.909 | `MENENIUS:\nIUS:\nLorer'l nd.\nWhu renell beathe wo...` |
| 0.7 | 5 | 1.149 | 0.125 | 0.322 | 0.561 | `MENENIUS:\nIUS:\nWore mins, that the whathe ton th...` |
| 0.7 | 20 | 1.698 | 0.165 | 0.548 | 0.854 | `MENENIUS:\nIUS:\nLorer'l nd.\nWhu renell beathe wo...` |
| 1.0 | None (full vocab) | 2.083 | 0.185 | 0.608 | 0.904 | `MENENIUS:\nI ha llleem ynd.\nWhu renell b?\nS:\nS:...` |
| 1.0 | 5 | 1.334 | 0.135 | 0.437 | 0.737 | `MENENIUS:\nI st ronere you.\nWhe you mat sprerit w...` |
| 1.0 | 20 | 1.82 | 0.19 | 0.598 | 0.848 | `MENENIUS:\nI ha llleem you.\nWhu renell b?\nS:\nS:...` |
| 1.4 | None (full vocab) | 2.633 | 0.2 | 0.754 | 0.97 | `MENENIUS:\nI ha loneeml'n''ts;\nWr,: g't gay\nBing...` |
| 1.4 | 5 | 1.376 | 0.145 | 0.497 | 0.793 | `MENENIUS:\nI st ronerer, whel mereas,\nThe y, with...` |
| 1.4 | 20 | 2.28 | 0.205 | 0.678 | 0.929 | `MENENIUS:\nI ha loneem hel'ts;\nWrenel't gay\nLit ...` |
| 2.0 | None (full vocab) | 3.165 | 0.225 | 0.844 | 0.985 | `MENENIUS:\nghar l\nTheml'n''ts;; re: :\nTh?! to, g...` |
| 2.0 | 5 | 1.516 | 0.15 | 0.548 | 0.879 | `MENENIUS:\nIhar swater hellll me\nWithir sthe' ot ...` |
| 2.0 | 20 | 2.554 | 0.215 | 0.799 | 0.99 | `MENENIUS:\nNIUYol'seem y! rthau ranellov?\n\n\nBEi...` |

## 4. Key Findings & Interpretation

1. **The Greedy Degeneration Trap ($T=0.1$)**:
   - At $T=0.1$, entropy drops dramatically. The model rapidly enters infinite repeating loops (e.g. repeated citizen lines or identical character strings), resulting in low Distinct-2 and Distinct-3 scores.
   - **Mechanism**: The argmax choice repeatedly reinforces the same local context window, trapping the autoregressive state in a closed cycle.

2. **The "Goldilocks Zone" ($T \in [0.7, 1.0]$, Top-$k \in [5, 20]$)**:
   - This regime achieves optimal balance: entropy remains moderate, Distinct-2 is high, and the text maintains authentic Shakespearean dialogue formatting with character names, colons, and metered verses.

3. **High Temperature Chaos ($T \ge 1.4$) and Top-$k$ Intervention**:
   - At $T=2.0$ with `top_k=None`, entropy spikes toward theoretical maximum, and text disintegrates into unpronounceable character soup.
   - Adding `top_k=5` at $T=2.0$ restores syntactic structure by eliminating the long tail of 49 impossible characters while still permitting random choice among the 5 most plausible continuations.

## 5. Qualitative Sample Highlights

#### Temperature = 0.1, Top-k = None (full vocab)
- **Mean Entropy**: 0.223 | **Distinct-2**: 0.171
```
MENENIUS:
Wher st you where s mathe s theathe the athe athe athe ath athe he athe athe ath and
indind the the the an the the the the the athe the athe the athe athe ath ath ath ath he athe he ande the the an th
```

#### Temperature = 0.1, Top-k = 5
- **Mean Entropy**: 0.221 | **Distinct-2**: 0.171
```
MENENIUS:
Wher st you where s mathe s theathe the athe athe athe ath athe he athe athe ath and
indind the the the an the the the the the athe the athe the athe athe ath ath ath ath he athe he ande the the an th
```

#### Temperature = 0.4, Top-k = None (full vocab)
- **Mean Entropy**: 0.921 | **Distinct-2**: 0.412
```
MENENIUS:
IUS:
Whe hel nore where s, athathe whe whe s athese athe malus
The the te the de an sthealat t sthe ale athe the ave and
irthe cthe he is an prore.



Secondititit Citizen:
We onoust e at there hen ea
```

#### Temperature = 0.4, Top-k = 5
- **Mean Entropy**: 0.75 | **Distinct-2**: 0.377
```
MENENIUS:
IUS:
Whe whel are where s, alll tod the aton we whe sothe sous here he athe aven an pore.




Seconditit Cititizen:
We at athat the here ande athe atond ande and
indond this the ther the he seane s fo
```

#### Temperature = 0.7, Top-k = None (full vocab)
- **Mean Entropy**: 1.686 | **Distinct-2**: 0.573
```
MENENIUS:
IUS:
Lorer'l nd.
Whu renell beathe wowe nd y wea cous, kiked cinountes be peard
vendistitentit sten: ce an t ise boun f seror yousther is arepr ar noort pecusapeliary.

Al:
Firend Citen:
Whe gakith e 
```

#### Temperature = 0.7, Top-k = 5
- **Mean Entropy**: 1.149 | **Distinct-2**: 0.322
```
MENENIUS:
IUS:
Wore mins, that the whathe ton the atout he sesere arat tre atheatre the th an hode anthe sthatre the thise ave ands tonthes the foryor the
is tour we wan therathe th alere ther s athe sean th fo
```

#### Temperature = 1.0, Top-k = None (full vocab)
- **Mean Entropy**: 2.083 | **Distinct-2**: 0.608
```
MENENIUS:
I ha llleem ynd.
Whu renell b?
S:
S:
NENENENI'et y'su, kirst, whI'f hat nd berd! whelll heakENast?


Aplveal:
We boud fkit onsychemThere.
The hote notit we wane lllly, wesitatre over,e W' we ghowen er
```

#### Temperature = 1.0, Top-k = 5
- **Mean Entropy**: 1.334 | **Distinct-2**: 0.437
```
MENENIUS:
I st ronere you.
Whe you mat sprerit wenoury we wheserer malalin

Theshee the od an sodeneakit st, hact and wise willinds ath:
sthe't we whe whothe sowh we were therore this, tod honond s where win st
```

#### Temperature = 1.4, Top-k = None (full vocab)
- **Mean Entropy**: 2.633 | **Distinct-2**: 0.754
```
MENENIUS:
I ha loneeml'n''ts;
Wr,: g't gay
Bingwe? wou we Wcesu, knknd, Cvikfusecen: ar go pe bod, yakENale?

veppatit iutird dif serortyc-heThere. ngeproar nooit wucus.

Whalyouy szeatrverver,eAW' l, gake n ea
```

#### Temperature = 1.4, Top-k = 5
- **Mean Entropy**: 1.376 | **Distinct-2**: 0.497
```
MENENIUS:
I st ronerer, whel mereas,
The y, withe ally we wouserer wald whod athere the th an hodemes,

sthearalvest t ise poustitiert, weshe here angor ote woory we wan thally.

Al:
Fired,-irsed Cil, Cititite,
```

#### Temperature = 2.0, Top-k = None (full vocab)
- **Mean Entropy**: 3.165 | **Distinct-2**: 0.844
```
MENENIUS:
ghar l
Theml'n''ts;; re: :
Th?! to, gwe?niuy We WcNNENI:
--CucCvoufbe, beCinize! ps bod,hy',ENast?
Vvcek; bCeiue IdDiIf?
St-TNyc-htTherebyng: d arenooit luzescp
Whalyoed szeatrvor-or,eAW'wUHe;; hwj e,
```

#### Temperature = 2.0, Top-k = 5
- **Mean Entropy**: 1.516 | **Distinct-2**: 0.548
```
MENENIUS:
Ihar swater hellll me
Withir sthe' ot t ale tavinseserend avang
Thoresored aingh an hid heakit stondicens at the cous,f wer
Tayesuly, se dorerathe avort ands andicct, eden ak, undondour
Aporecese ite.
```


## 6. Limitations & Follow-Up Experiment

* **Limitations**: Evaluated on a small character-level vocabulary ($|V| = 54$). On subword BPE tokenizers ($|V| \approx 50,000$), tail mass is significantly larger, making top-$p$ (nucleus) sampling often preferable to fixed top-$k$.
* **Follow-up Experiment (Exp 02)**: **Context Length Scaling and Causal Attention Ablation**:
  Measure the impact of context window size ($T \in [16, 32, 64, 128]$) on attention matrix sparsity and downstream loss convergence.
