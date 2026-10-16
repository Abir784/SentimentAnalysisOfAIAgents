"""Regenerate the method pipeline figure with the validated workflow.

Same visual language as the original (rounded pastel boxes, left-to-right
arrows): 7 stages, validation inserted before analysis and feeding forward.
Output: data/figures/method_pipeline_validated.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data/figures/method_pipeline_validated.png"

BOXES = [
    ("Data Collection",
     "1,296 MoltBook comments\n(55 posts, 548 authors)\nCollected Jan 2026",
     "#D6E4F0"),
    ("Preprocessing",
     "langdetect, lowercasing,\nURL/emoji removal,\nNLTK tokenisation,\nlemmatisation\n(1,219 English)",
     "#D9EAD3"),
    ("Candidate Scorers",
     "VADER + SentiWordNet\n\u2192 ensemble (0.05/0.02,\nagree-or-neutral backoff)\n+ 3 transformers",
     "#FCE5CD"),
    ("Human Validation",
     "250-comment gold,\ndouble-annotated (\u03ba),\nbenchmark 6 scorers\n\u2192 empirical selection",
     "#CFE8E4"),
    ("Network Analysis",
     "Sequential-proxy reply\nnetwork (fallback),\nPageRank, Reciprocity,\nLouvain, Gini index",
     "#FFF2CC"),
    ("Dynamics",
     "Shares + CIs; K-W H\nwith \u03b5\u00b2 (post=thread once);\n\u03c7\u00b2(4), V, permutation;\ntrend test",
     "#EAD1DC"),
    ("Robustness",
     "Preprocessing variants;\nscorer sensitivity;\ndirect-edge confirmation\n(pending)",
     "#D9D2E9"),
]


def main() -> None:
    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots(figsize=(20, 4.4), dpi=200)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    n = len(BOXES)
    gap = 0.32
    w = (20 - gap * (n + 1)) / n
    h = 3.2
    y = 0.45
    prev_center = None
    for i, (title, body, color) in enumerate(BOXES):
        x = gap + i * (w + gap)
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.18",
                             facecolor=color, edgecolor="#444444", linewidth=1.0)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h - 0.40, title, ha="center", va="center",
                fontsize=15, fontweight="bold", color="black")
        ax.text(x + w / 2, y + h / 2 - 0.34, body, ha="center", va="center",
                fontsize=13, color="black", linespacing=1.4)
        if prev_center is not None:
            arr = FancyArrowPatch((prev_center, y + h / 2), (x, y + h / 2),
                                  arrowstyle="-|>", mutation_scale=10,
                                  linewidth=1.0, color="#444444")
            ax.add_patch(arr)
        prev_center = x + w

    fig.tight_layout(pad=0.4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
