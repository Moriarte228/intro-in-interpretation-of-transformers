# ARENA interpretability notebooks

My work through chapters of the [ARENA](https://www.arena.education/) curriculum on transformer interpretability, adapted to run for free on Kaggle GPUs (T4) instead of Colab / paid APIs. Each notebook contains the course exercises plus my own adaptations and extra experiments.

My main interpretability project, truth probes on Gemma-2-9B, is in a separate repository: [truth-probes-transfer](https://github.com/Moriarte228/truth-probes-transfer).

## Notebooks

| Notebook | Topic | Models |
|---|---|---|
| `1.3.1-linear-probes-llama2.ipynb` | Linear probes for truth: probe training, causal interventions, deception probes, attention probes | LLaMA-2-13B, LLaMA-3.1-8B-Instruct |
| `1.3.2-function-vectors.ipynb` | Function vectors and steering vectors with `nnsight` | GPT-J-6B, GPT2-XL |
| `4.4-persona-vectors.ipynb` | Persona vectors: contrastive extraction, steering, LLM-judge evaluation | Gemma-2-9B, Qwen2.5-7B-Instruct |

### 1.3.2 Function vectors

Kaggle adaptation: local execution instead of NDIF (`REMOTE = False`), float16 instead of bfloat16 (the T4 has no native bf16), GPT-J loaded from its float16 revision (~12 GB instead of ~24 GB). Sections 3–4 (function vectors, steering in GPT2-XL) are executed; sections 1–2 are mostly code without saved outputs.

### 4.4 Persona vectors: my additions (sections 3–4, Qwen2.5-7B-Instruct)

- **Honest evaluation of the contrastive sycophancy direction.** The data is split by question rather than by pair, so the same question never appears in both train and test. Held-out separation: d′ = 5.17, AUROC = 1.000. Shuffled-label control: held-out AUROC 0.469.
- **Steering works and is graded.** Mean LLM-judge sycophancy score (0–100) goes from ≈2 without steering to ≈17 at coefficient +1 and ≈95–98 at +3…+5. Negative coefficients stay at ≈0.
- **Logit lens shows nothing interpretable.** The logit-lens difference between the mean sycophantic and mean honest states is compared with a random direction of the same norm. Its top tokens are fragments and unrelated words rather than agreement or praise words. So the behavioural effect of the direction is not visible in the direct vocabulary readout.
- **Free-tier setup:** generation runs on the local model, and Gemini (free tier) acts as the judge, with request batching and caching to fit the quota.

Sections 1–2 (persona space and the assistant axis on Gemma-2-9B) are mostly code without saved outputs in this version.

## Running

Open in Kaggle with a GPU. Add API keys as Kaggle Secrets (`HF_TOKEN` for gated models, `GEMINI_API_KEY` for the judge). Nothing is hardcoded.
