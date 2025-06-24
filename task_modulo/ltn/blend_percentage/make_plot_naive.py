import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl

# ──────────────────────────────────────────────────────────────
# Global Matplotlib style tweaks
# ──────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.size": 24,            # base font size
    "axes.titlesize": 24,       # title font size
    "axes.labelsize": 20,       # x / y label size
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 24,      # ← bigger legend entries
    "legend.title_fontsize": 26 # ← bigger legend title
})

# ──────────────────────────────────────────────────────────────
# Load the data
# ──────────────────────────────────────────────────────────────
PARENT_DIR = Path(__file__).resolve().parent
df = pd.read_csv(PARENT_DIR / "naive.csv")

# Label each run by blend percentage
df["run_label"] = df.apply(lambda r: f"blend={r['blend_percentage']}", axis=1)
df = df.sort_values(["run_id", "epoch"])

# ──────────────────────────────────────────────────────────────
# Plot benign accuracy + ASR
# ──────────────────────────────────────────────────────────────
plt.figure(figsize=(14, 12))

for i, (run_label, group) in enumerate(df.groupby("run_label")):
    group = group.sort_values("epoch")
    colour = f"C{i % 10}"  # cycle through default colour set

    plt.plot(
        group["epoch"], group["clean_acc"],
        label=f"{run_label} – Benign",
        color=colour,
        linewidth=2
    )
    plt.plot(
        group["epoch"], group["asr"],
        label=f"{run_label} – ASR",
        color=colour,
        linestyle="--"
    )

plt.title("Benign Accuracy and ASR over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True, alpha=0.3)

# ──────────────────────────────────────────────────────────────
# Legend: inside plot, larger font
# ──────────────────────────────────────────────────────────────
legend = plt.legend(
    title="Blend %",
    loc="upper right",            # inside the axes
    bbox_to_anchor=(0.98, 0.98),  # small inset from the corner
    ncol=1,                       # single column
    framealpha=0.8                # light translucent box
)
legend.get_frame().set_linewidth(0.5)  # thin border

plt.tight_layout()

# ──────────────────────────────────────────────────────────────
# Save the figure
# ──────────────────────────────────────────────────────────────
plt.savefig(PARENT_DIR / "blend_percentage_ltn.png", dpi=150)
plt.close()
