import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl

# ──────────────────────────────────────────────────────────────
# Global Matplotlib style tweaks
# ──────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.size": 24,
    "axes.titlesize": 24,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 24,        # legend entries
    "legend.title_fontsize": 28   # legend title
})

# ──────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────
PARENT_DIR = Path(__file__).resolve().parent
df = pd.read_csv(PARENT_DIR / "comparison.csv")

run_id_map = {
    "20250622T191947": "Addition PGD Targeted",
    "20250622T193815": "Addition PGD Targeted Dirty Label",
    "20250622T194814": "Modulo PGD Targeted",
    "20250622T185300": "Modulo PGD Targeted Dirty Label"
}

df["run_label"] = df["run_id"].map(run_id_map)
df = df.sort_values(["run_id", "epoch"])

# ──────────────────────────────────────────────────────────────
# Plot benign accuracy + ASR
# ──────────────────────────────────────────────────────────────
plt.figure(figsize=(18, 12))

for i, (label, group) in enumerate(df.groupby("run_label")):
    group = group.sort_values("epoch")
    color = f"C{i % 10}"
    plt.plot(group["epoch"], group["clean_acc"],
             label=f"{label} – Benign", color=color, linewidth=2)
    plt.plot(group["epoch"], group["asr"],
             label=f"{label} – ASR",    color=color, linestyle="--")

plt.title("Benign Accuracy and ASR per Model over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True, alpha=0.3)

# ──────────────────────────────────────────────────────────────
# Legend: inside plot, font 20 / 24
# ──────────────────────────────────────────────────────────────
legend = plt.legend(
    title="Model Variant",
    loc="upper right",            # inside the axes
    bbox_to_anchor=(0.98, 0.98),  # slight inset from the edge
    ncol=1,
    framealpha=0.8
)
legend.get_frame().set_linewidth(0.5)

plt.tight_layout()
plt.savefig(PARENT_DIR / "comparison_dirty.png", dpi=150)
plt.close()
