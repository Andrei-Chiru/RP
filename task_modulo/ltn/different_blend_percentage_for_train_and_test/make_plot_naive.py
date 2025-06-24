"""
Plot benign accuracy and attack-success rate (ASR) versus training epoch,
with the legend placed **inside** the axes and rendered at a smaller size.
"""

from pathlib import Path

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Matplotlib global style tweaks
# ------------------------------------------------------------------
mpl.rcParams.update({
    "font.size": 24,          # base font size
    "axes.titlesize": 24,     # title font size
    "axes.labelsize": 20,     # x / y label size
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,    # default legend size (overridden below)
    "legend.title_fontsize": 20
})

# ------------------------------------------------------------------
# Load the CSV
# ------------------------------------------------------------------
PARENT_DIR = Path(__file__).resolve().parent
df = pd.read_csv(PARENT_DIR / "naive.csv")

# A concise identifier for each (blend %, test-blend %) pair
df["label"] = df.apply(
    lambda r: f"blend={r['blend_percentage']}, test blend={r['test_blend_percentage']}",
    axis=1
)

# ------------------------------------------------------------------
# Plot: benign accuracy + ASR
# ------------------------------------------------------------------
plt.figure(figsize=(14, 10))

# Benign accuracy curves
for label, grp in df.groupby("label"):
    grp = grp.sort_values("epoch")
    plt.plot(
        grp["epoch"],
        grp["clean_acc"],
        label=f"{label} – Benign",
        linewidth=2
    )

# ASR curves (dashed)
for label, grp in df.groupby("label"):
    grp = grp.sort_values("epoch")
    plt.plot(
        grp["epoch"],
        grp["asr"],
        linestyle="--",
        label=f"{label} – ASR",
        linewidth=2
    )

plt.title("Benign Accuracy & ASR over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True, alpha=0.3)

# ------------------------------------------------------------------
# Legend: inside the axes, smaller font
# ------------------------------------------------------------------
legend = plt.legend(
    title="Blend/Test Blend Pair",
    loc="upper right",           # inside plot area
    bbox_to_anchor=(0.98, 0.98), # slight inset from corner
    fontsize=20,                 # legend entry size
    title_fontsize=24,           # legend title size
    framealpha=0.8,              # translucent background
    ncol=1                       # single column
)
legend.get_frame().set_linewidth(0.5)  # thin border

plt.tight_layout()

# ------------------------------------------------------------------
# Save fig
# ------------------------------------------------------------------
plt.savefig(PARENT_DIR / "naive.png", dpi=150)
plt.close()
