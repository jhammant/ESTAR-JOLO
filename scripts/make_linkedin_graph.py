#!/usr/bin/env python3
"""Generate a polished LinkedIn-ready bar chart for ESTAR-LITE results."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Load results
data = json.loads(Path("results/estar_summary.json").read_text())
problems = data["problems"]

# Sort by full thinking tokens (descending) — biggest problems first
problems_sorted = sorted(problems, key=lambda p: p["thinking_tokens"], reverse=True)

names = [p["short"] for p in problems_sorted]
full_tokens = [p["thinking_tokens"] for p in problems_sorted]
estar_tokens = [p["estar_tokens"] for p in problems_sorted]
correct = [p["correct"] for p in problems_sorted]

# ── Create the figure ──
fig, (ax_main, ax_stats) = plt.subplots(
    1, 2, figsize=(16, 7),
    gridspec_kw={"width_ratios": [3, 1]},
)
fig.patch.set_facecolor("#FAFAFA")

# Colors
FULL_COLOR = "#3B82F6"      # blue
ESTAR_COLOR = "#F59E0B"     # amber
SAVED_COLOR = "#10B981"     # green
ACCENT_RED = "#EF4444"
BG_COLOR = "#FAFAFA"

# ── Main plot: grouped bar chart ──
ax_main.set_facecolor(BG_COLOR)
x = np.arange(len(names))
bar_width = 0.35

# Full reasoning bars
bars_full = ax_main.bar(
    x - bar_width / 2, full_tokens, bar_width,
    label="Full reasoning", color=FULL_COLOR,
    edgecolor="white", linewidth=0.8, zorder=3,
)

# ESTAR bars
bars_estar = ax_main.bar(
    x + bar_width / 2, estar_tokens, bar_width,
    label="ESTAR-LITE (early stop)", color=ESTAR_COLOR,
    edgecolor="white", linewidth=0.8, zorder=3,
)

# Annotate savings percentage above each pair
for i, (ft, et) in enumerate(zip(full_tokens, estar_tokens)):
    saved = ft - et
    if saved > 0:
        pct = saved / ft * 100
        y_pos = ft + 20
        ax_main.text(
            i, y_pos, f"-{pct:.0f}%",
            fontsize=10, fontweight="bold", color=SAVED_COLOR,
            ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=SAVED_COLOR, alpha=0.9),
            zorder=4,
        )

# Mark incorrect answers with red border on the ESTAR bar
for i, c in enumerate(correct):
    if not c:
        bars_estar[i].set_edgecolor(ACCENT_RED)
        bars_estar[i].set_linewidth(2.5)
        ax_main.text(
            i + bar_width / 2, estar_tokens[i] + 10, "X",
            fontsize=12, fontweight="bold", color=ACCENT_RED,
            ha="center", va="bottom", zorder=5,
        )

ax_main.set_xticks(x)
ax_main.set_xticklabels(names, rotation=30, ha="right", fontsize=10)
ax_main.set_ylabel("Thinking Tokens", fontsize=12, fontweight="bold")
ax_main.set_title(
    "Full Reasoning vs ESTAR-LITE Early Stopping",
    fontsize=15, fontweight="bold", pad=15,
)
ax_main.legend(fontsize=11, loc="upper right", framealpha=0.9)
ax_main.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
ax_main.set_xlim(-0.5, len(names) - 0.5)
ax_main.spines["top"].set_visible(False)
ax_main.spines["right"].set_visible(False)

# ── Right panel: summary stats ──
ax_stats.set_facecolor(BG_COLOR)
ax_stats.axis("off")

total_full = sum(full_tokens)
total_estar = sum(estar_tokens)
total_saved = total_full - total_estar
reduction = total_full / max(total_estar, 1)
accuracy = sum(1 for c in correct if c) / len(correct) * 100
mean_savings = total_saved / total_full * 100

# Big number: reduction ratio
ax_stats.text(
    0.5, 0.92, f"{reduction:.1f}x",
    transform=ax_stats.transAxes,
    fontsize=48, fontweight="bold", color=ESTAR_COLOR,
    ha="center", va="top",
)
ax_stats.text(
    0.5, 0.78, "token reduction",
    transform=ax_stats.transAxes,
    fontsize=13, color="#666666",
    ha="center", va="top",
)

# Stats boxes
stats = [
    ("Accuracy", f"{accuracy:.0f}%", FULL_COLOR),
    ("Mean Savings", f"{mean_savings:.0f}%", SAVED_COLOR),
    ("Full Tokens", f"{total_full:,}", "#666666"),
    ("ESTAR Tokens", f"{total_estar:,}", ESTAR_COLOR),
    ("Tokens Saved", f"{total_saved:,}", SAVED_COLOR),
]

for i, (label, value, color) in enumerate(stats):
    y = 0.62 - i * 0.12
    ax_stats.text(
        0.5, y, value,
        transform=ax_stats.transAxes,
        fontsize=18, fontweight="bold", color=color,
        ha="center", va="center",
    )
    ax_stats.text(
        0.5, y - 0.04, label,
        transform=ax_stats.transAxes,
        fontsize=9, color="#999999",
        ha="center", va="center",
    )

# Footer
ax_stats.text(
    0.5, 0.04,
    "DeepSeek-R1 1.5B\nOllama (local Mac)\nReal logprobs + threshold tuning",
    transform=ax_stats.transAxes,
    fontsize=9, color="#AAAAAA",
    ha="center", va="bottom",
    style="italic",
)

# Subtitle
fig.text(
    0.02, 0.01,
    "JOLO: Real Ollama logprobs, threshold/patience tuning, adaptive stopping  |  github.com/jhammant/JOLO",
    fontsize=9, color="#999999", style="italic",
)

plt.tight_layout(rect=[0, 0.03, 1, 1])

output = "results/estar_linkedin_graph.png"
plt.savefig(output, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {output}")
