# autoresearch - 8-Agent Swarm Results

![progress](progress.png)

This is a fork of [@karpathy's autoresearch](https://github.com/karpathy/autoresearch), where we ran an **8-agent swarm** across **8 H100 GPUs** to autonomously optimize a small LLM training setup. The swarm consisted of 4 Claude Code agents and 4 Codex agents (later replaced with 4 more Claude agents), coordinated via [ClawTeam](https://github.com/a9logic/clawteam).

## Results

| Metric | Baseline | Final | Change |
|--------|----------|-------|--------|
| **val_bpb** | 1.044 | **0.977** | **-6.4%** |
| Total experiments | - | 2430+ | - |
| GPU-hours | - | ~240 | 8 GPUs x ~30h |
| Agent-hours | - | ~240 | 8 agents x ~30h |

## Key Optimizations Discovered

The agents independently discovered and validated the following improvements through systematic experimentation:

**Architecture (biggest wins):**
- `DEPTH=12` (from 8) - deeper model with 768-dim embeddings
- Norm-before-RoPE - apply QK normalization before rotary embeddings
- `q *= 1.25` - scale queries before attention for sharper attention patterns
- `short_window=512` - fixed sliding window size
- `RoPE base=60000` - higher frequency base for positional encoding

**Optimization (second biggest wins):**
- `TOTAL_BATCH_SIZE=2^17` (from 2^19) - 4x smaller batch = 4x more gradient steps in 5 minutes
- `MATRIX_LR=0.028` - Muon learning rate tuned for depth-12
- `UNEMBEDDING_LR=0.009` - higher LR for output projection
- `SCALAR_LR=0.25` - lower scalar LR
- `WEIGHT_DECAY=0.17` with linear decay schedule
- `MUON_BETA2=0.83`, momentum warmup `0.90 -> 0.95`

**Regularization:**
- `EMBEDDING_WEIGHT_DECAY=0.008` with linear decay to 0
- Dynamic unembedding weight decay, also decaying to 0

**Schedule:**
- `WARMDOWN_RATIO=0.88` - longer learning rate cooldown
- `FINAL_LR_FRAC=0.03` - small residual learning rate

## Optimization Timeline

The experiment progressed in four distinct phases:

1. **Architecture discovery** (experiments 1-12): Agents quickly found that `DEPTH=12` with `batch=2^17` was far better than the default, dropping val_bpb from 1.044 to 0.991.

2. **Deep tuning** (experiments 12-170): Claude agents ran 200+ experiments each, discovering norm-before-RoPE, weight decay decay schedules, embedding regularization, and RoPE base tuning. val_bpb reached 0.982.

3. **Cross-pollination** (experiments 170-280): Codex agents received the best configs and tried combining findings from different Claude agents. Discovered that norm-before-rotary + q-scaling + momentum tuning stack well. val_bpb reached 0.978.

4. **Fine-tuning** (experiments 280-430+): Fresh Claude agents started from the global best commit and ran systematic hyperparameter sweeps, finding the final improvements in UNEMBEDDING_LR and MATRIX_LR+MUON_BETA2. val_bpb reached 0.977.

## Agent Contributions

| Agent | GPU | Experiments | Best val_bpb | Key Discoveries |
|-------|-----|-------------|-------------|----------------|
| gpu0-claude | 0 | 408 | 0.982 | depth=12, norm-before-RoPE, q_scale, Muon momentum |
| gpu1-claude | 1 | 424 | 0.987 | GELU-squared, MLP 5x, embedding WD differentiation |
| gpu2-claude | 2 | 434 | 0.981 | LR/WD systematic tuning, init scale, MLP Muon LR |
| gpu3-claude | 3 | 432 | 0.981 | Embedding WD decay, RoPE 60k, short_window=512 |
| gpu4-claude2 | 4 | 175 | 0.977 | UNEMBEDDING_LR=0.010, SCALAR_LR=0.22, EMBEDDING_LR=0.65 |
| gpu5-claude2 | 5 | 180 | **0.977** | WARMDOWN=0.88, MATRIX_LR=0.028 + MUON_BETA2=0.83 |
| gpu6-claude2 | 6 | 173 | 0.979 | Embedding LR, warmdown ratio, short_window tuning |
| gpu7-claude2 | 7 | 174 | 0.977 | scalar_lr=0.35, Muon beta2, wte init tuning |
| Codex R1-R7 | 4-7 | ~60 | 0.980 | Attention patterns, ablations, architecture validation |

## What Didn't Work

The agents also tried many things that failed, providing useful negative results:
- MoE (Mixture of Experts) - significant regression
- GQA / MQA - regression, full MHA is better at this scale
- ALiBi / NoPE positional encodings - RoPE is clearly better
- Label smoothing - catastrophic regression
- Gradient clipping - regression
- GELU / SiLU activations - ReLU-squared remains best
- Deeper models (14-24 layers) - too slow, fewer steps in budget
- Wider models (ASPECT_RATIO > 64) - worse at depth=12

## How to Reproduce

```bash
# Setup
uv sync
uv run prepare.py

# Run the optimized model (5 minutes)
uv run train.py
# Expected: val_bpb ≈ 0.977
```

## Multi-Agent Setup

The swarm was coordinated using [ClawTeam](https://github.com/a9logic/clawteam) with git worktrees for isolation:

```bash
# Install ClawTeam
pip install -e ~/ClawTeam/

# Create team
clawteam team spawn-team autoresearch -d "8-agent LLM optimization"

# Spawn agents (each gets its own git worktree)
CUDA_VISIBLE_DEVICES=0 clawteam spawn tmux claude \
  --team autoresearch --agent-name gpu0 --workspace --skip-permissions \
  --task "Read program.md and start experimenting"
```

See `program.md` for the full experiment protocol.

## Original README

See [README-old.md](README-old.md) for the original autoresearch documentation by @karpathy.

## License

MIT
