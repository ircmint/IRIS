"""
draw_tagnet_v2_architecture.py

Recreates the exact V2 architecture diagram (Figure 2, TAG-Net-VLM final)
the user supplied -- same boxes, same colors, same formulas, same layout.
Not a redesign: this reproduces what was given so it can be embedded in a
docx (the original was pasted as an image, not saved as a file).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(16, 10.2))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10.2)
ax.axis("off")

C_STD = "#CFE0F5"       # Standard
C_ADAPTER = "#FBE2B9"   # Novel: adapter
C_AMG = "#D9EAD3"       # Novel: AMG
C_CGPA = "#F4CCCC"      # Novel: CGPA
C_KG = "#E5D6F0"        # Retrieval / KG
C_NEUTRAL = "#EDEDED"
C_BORDER = "#1A1A1A"
TXT = "#111111"


def box(x, y, w, h, text, color, fontsize=9.3, weight="normal", edge=C_BORDER, lw=1.4, style="normal"):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                           linewidth=lw, edgecolor=edge, facecolor=color, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=TXT, weight=weight, style=style, zorder=3, linespacing=1.4)


def arrow(x1, y1, x2, y2, lw=1.3, connectionstyle="arc3,rad=0.0"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                         linewidth=lw, color=C_BORDER, zorder=1, connectionstyle=connectionstyle)
    ax.add_patch(a)


# ---- Legend row ----
legend_items = [("Standard", C_STD), ("Novel: adapter", C_ADAPTER), ("Novel: AMG", C_AMG),
                 ("Novel: CGPA", C_CGPA), ("Retrieval / KG", C_KG)]
lx = 0.3
for label, color in legend_items:
    box(lx, 9.65, 0.45, 0.32, "", color, lw=1.2)
    ax.text(lx + 0.58, 9.81, label, ha="left", va="center", fontsize=9.5)
    lx += 0.45 + 0.13 + len(label) * 0.095 + 0.35

# ---- Row 1: inputs ----
box(0.3, 8.35, 2.9, 0.75, "Video\nN frames", C_STD)
box(3.5, 8.35, 2.9, 0.75, "Telemetry summary", C_STD)
box(6.7, 8.35, 2.9, 0.75, "GPS / context", C_STD)
box(10.1, 8.35, 5.6, 0.75, "Retrieved IRC clauses (top-k, per fold)", C_KG)

# ---- Row 2: encoders / adapters ----
box(0.3, 6.95, 2.9, 1.1, "Vision Encoder\n(ViT, LoRA)", C_STD)
box(3.5, 6.95, 2.9, 1.1, "Telemetry Adapter\nMLP(4→64)+GELU → ℝ¹⁵³⁶", C_ADAPTER, fontsize=8.8)
box(6.7, 6.95, 2.9, 1.1, "Context Adapter\nMLP(6→64)+GELU → ℝ¹⁵³⁶", C_ADAPTER, fontsize=8.8)
box(10.1, 6.95, 5.6, 1.1, "Clause Encoder\ntext → k+1 embeddings (incl. 'no clause')", C_KG, fontsize=8.9)

arrow(1.75, 8.35, 1.75, 8.05)
arrow(4.95, 8.35, 4.95, 8.05)
arrow(8.15, 8.35, 8.15, 8.05)
arrow(12.9, 8.35, 12.9, 8.05)

# ---- Row 3: AMG + CGPA (CGPA sits directly under Clause Encoder, no overlap) ----
box(3.5, 5.55, 6.1, 0.85,
    "Adaptive Modality Gating (AMG):   (g_tel, g_ctx) = softmax(MLP([v̄, t̄, c̄]))",
    C_AMG, fontsize=9.3, weight="bold")
box(10.1, 4.35, 5.6, 2.6,
    "Clause-Grounded\nPointer Attention  (n_heads = 4)\n\n"
    "query: decoder hidden state\nat <CITE>\n\n"
    "keys / values: k+1\nclause embeddings\n\n"
    "α = softmax(QKᵀ / √d_head)\n\n"
    "cited clause = argmax(α)\n(copied, never generated)",
    C_CGPA, fontsize=9.0, weight="bold")

arrow(4.95, 6.95, 4.95, 6.4)
arrow(8.15, 6.95, 8.15, 6.4)

# ---- Row 4: fused sequence ----
box(0.3, 4.35, 9.3, 0.75,
    "Fused sequence:  visual tokens ⊕ g_tel·telemetry token ⊕ g_ctx·context token ⊕ prompt",
    C_NEUTRAL, fontsize=9, edge="#777")
arrow(1.75, 6.95, 1.9, 5.1, connectionstyle="arc3,rad=0.15")
arrow(4.95, 5.55, 4.95, 5.1)
arrow(8.15, 5.55, 6.8, 5.1, connectionstyle="arc3,rad=-0.15")

# ---- Row 5: decoder ----
box(0.3, 2.95, 9.3, 1.15,
    "Decoder (Qwen2-VL-2B) — QLoRA / LoRA, r = 16, α = 32, p = 0.05\n"
    "FOLD 1 free text  +  hidden state at <CITE> for FOLD 2 / FOLD 3",
    C_STD, fontsize=9.2, weight="bold")
arrow(4.95, 4.35, 4.95, 4.1)
ax.text(9.75, 3.7, "query", ha="left", va="center", fontsize=8.6, style="italic")
arrow(9.6, 3.55, 10.1, 5.3, connectionstyle="arc3,rad=-0.25")
arrow(9.6, 3.4, 12.9, 3.4, connectionstyle="arc3,rad=0.0")

# ---- Row 6: fold outputs ----
box(0.3, 1.55, 2.9, 1.05, "FOLD 1\nEnvironment\n(free text)", C_STD, fontsize=9)
box(3.5, 1.55, 3.1, 1.05, "FOLD 2\nRoad Surface (IRC 35)\ncite ← CGPA", C_STD, fontsize=9)
box(6.9, 1.55, 3.1, 1.05, "FOLD 3\nInfrastructure (IRC 67)\ncite ← CGPA", C_STD, fontsize=9)
arrow(1.75, 2.95, 1.75, 2.6)
arrow(5.0, 2.95, 5.0, 2.6)
arrow(8.4, 2.95, 8.4, 2.6)

# ---- Row 7: metrics footer ----
box(0.3, 0.3, 15.4, 0.95,
    "Metrics:   Accuracy / Macro-F1  ·  CPA  ·  structural CHR = 0  ·  No-Clause Recall  ·  "
    "pointer entropy → conf.  ·  params / latency (APMP, LNA)",
    C_NEUTRAL, fontsize=9, edge="#777")

ax.text(0.15, -0.35,
        "Figure 2. TAG-Net-VLM (final): Qwen2-VL-2B backbone with QLoRA/LoRA, Adaptive Modality Gating (AMG) over\n"
        "telemetry/context tokens, and Clause-Grounded Pointer Attention (CGPA) for citation resolution. Total added\n"
        "parameters beyond the backbone: adapters + AMG + CGPA ≈ 12.7M ( < 1% of the 2B backbone).",
        fontsize=8.6, color="#333")

plt.tight_layout()
plt.savefig("tagnet_v2_architecture.png", dpi=220, bbox_inches="tight", facecolor="white")
print("Saved tagnet_v2_architecture.png")
