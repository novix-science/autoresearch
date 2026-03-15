"""Generate progress plot for the AutoResearch 8-agent swarm experiment."""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

# Global best val_bpb progression (chronological order of discoveries)
# Each entry: (experiment_number_approx, val_bpb, label)
global_best_timeline = [
    # Phase 1: Architecture discovery (gpu0/gpu3, first ~20 experiments)
    (1, 1.044, "baseline (depth=8)"),
    (3, 1.022, "depth=10"),
    (5, 1.016, "depth=12"),
    (8, 0.997, "batch=2^18"),
    (10, 0.991, "batch=2^17"),
    # Phase 2: Claude deep optimization (~50-200 experiments)
    (15, 0.989, "warmdown=0.8"),
    (20, 0.987, "WD=0.1"),
    (25, 0.986, "final_lr=0.05"),
    (30, 0.986, "x0_lambda=0.2"),
    (35, 0.985, "norm-before-RoPE"),
    (50, 0.984, "quadratic WD decay"),
    (70, 0.984, "Muon beta2=0.92"),
    (90, 0.983, "Muon momentum 0.80->0.93"),
    (100, 0.983, "emb WD decay"),
    (120, 0.982, "RoPE base=60k"),
    (140, 0.982, "unembedding WD decay"),
    (160, 0.982, "short_window=512"),
    # Phase 3: Codex rounds + cross-pollination (~R1-R5)
    (180, 0.981, "combine gpu3 findings"),
    (200, 0.980, "+ norm-before-rotary"),
    (220, 0.980, "+ WD=0.17, MATRIX_LR=0.025"),
    (240, 0.979, "+ WARMDOWN=0.75, q*1.25"),
    (260, 0.978, "+ momentum 0.88->0.95"),
    # Phase 4: Claude2 fine-tuning (~R2 agents, 300+ experiments)
    (300, 0.978, "UNEMBEDDING_LR=0.010"),
    (340, 0.977, "UNEMBEDDING_LR=0.009"),
    (370, 0.977, "SCALAR_LR=0.22"),
    (400, 0.977, "MUON_MOMENTUM_END=0.93"),
    (430, 0.977, "MATRIX_LR=0.028 + MUON_BETA2=0.83"),
]

exps = [x[0] for x in global_best_timeline]
vals = [x[1] for x in global_best_timeline]

# Per-agent best trajectories (keeps only, running minimum)
agent_data = {
    "gpu0-claude": [1.045, 1.022, 1.016, 0.997, 0.991, 0.989, 0.989, 0.988, 0.986, 0.986, 0.985, 0.984, 0.984, 0.984, 0.983, 0.983, 0.983, 0.983, 0.982, 0.982, 0.982, 0.982, 0.982, 0.982],
    "gpu1-claude": [1.045, 1.026, 1.011, 0.995, 0.994, 0.994, 0.993, 0.992, 0.992, 0.991, 0.990, 0.991, 0.991, 0.990, 0.990, 0.989, 0.988, 0.988, 0.988, 0.988, 0.988, 0.987, 0.987, 0.987],
    "gpu2-claude": [1.044, 1.021, 1.018, 0.997, 0.992, 0.990, 0.988, 0.988, 0.988, 0.988, 0.988, 0.988, 0.987, 0.986, 0.986, 0.985, 0.985, 0.985, 0.985, 0.985, 0.985, 0.984, 0.984, 0.984, 0.984, 0.984, 0.984, 0.983, 0.983, 0.983, 0.982, 0.982],
    "gpu3-claude": [1.043, 1.030, 1.028, 0.992, 0.992, 0.991, 0.990, 0.990, 0.989, 0.989, 0.987, 0.988, 0.988, 0.988, 0.988, 0.986, 0.985, 0.985, 0.984, 0.983, 0.983, 0.983, 0.983, 0.982, 0.982, 0.981, 0.981, 0.981],
    "gpu4-claude2": [0.979, 0.978, 0.977, 0.977, 0.977, 0.977],
    "gpu5-claude2": [0.979, 0.979, 0.979, 0.979, 0.979, 0.979, 0.979, 0.979, 0.977, 0.977, 0.977],
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left plot: Global best progression
ax1.plot(exps, vals, 'o-', color='#2563eb', linewidth=2.5, markersize=4, zorder=5)
ax1.fill_between(exps, vals, max(vals), alpha=0.08, color='#2563eb')

# Annotate key milestones
annotations = [
    (1, 1.044, "baseline\n1.044", (-30, 15)),
    (5, 1.016, "depth=12", (10, 10)),
    (10, 0.991, "batch=2^17", (10, 10)),
    (100, 0.983, "emb WD\ndecay", (10, 10)),
    (200, 0.980, "norm-before\n-rotary", (-60, -20)),
    (430, 0.977, "final best\n0.9767", (10, -15)),
]
for x, y, text, offset in annotations:
    ax1.annotate(text, (x, y), textcoords="offset points", xytext=offset,
                fontsize=8, color='#1e40af', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#93c5fd', lw=1))

# Phase backgrounds
ax1.axvspan(0, 12, alpha=0.05, color='red', label='Phase 1: Architecture')
ax1.axvspan(12, 170, alpha=0.05, color='blue', label='Phase 2: Claude deep tuning')
ax1.axvspan(170, 280, alpha=0.05, color='green', label='Phase 3: Cross-pollination')
ax1.axvspan(280, 450, alpha=0.05, color='purple', label='Phase 4: Fine-tuning')

ax1.set_xlabel('Cumulative Experiments (approx)', fontsize=11)
ax1.set_ylabel('val_bpb (lower is better)', fontsize=11)
ax1.set_title('Global Best val_bpb Progression', fontsize=13, fontweight='bold')
ax1.legend(fontsize=8, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0.974, 1.050)

# Right plot: Per-agent trajectories
colors = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#3b82f6', '#8b5cf6', '#ec4899', '#06b6d4']
agent_names = list(agent_data.keys())
for i, (name, data) in enumerate(agent_data.items()):
    x = list(range(1, len(data)+1))
    # Running minimum
    running_min = []
    cur_min = float('inf')
    for v in data:
        cur_min = min(cur_min, v)
        running_min.append(cur_min)
    ax2.plot(x, running_min, '-', color=colors[i], linewidth=1.8, label=name, alpha=0.8)

ax2.set_xlabel('Agent Experiment # (keeps only)', fontsize=11)
ax2.set_ylabel('Running Best val_bpb', fontsize=11)
ax2.set_title('Per-Agent Optimization Trajectories', fontsize=13, fontweight='bold')
ax2.legend(fontsize=7, loc='upper right', ncol=2)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.974, 1.050)

# Add summary text
fig.text(0.5, 0.01,
    '8 agents (4 Claude Code + 4 Codex/Claude2) on 8x H100 GPUs | 2430+ experiments | baseline 1.044 → best 0.9767 (6.4% improvement)',
    ha='center', fontsize=10, style='italic', color='#475569')

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig('progress.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved progress.png")
