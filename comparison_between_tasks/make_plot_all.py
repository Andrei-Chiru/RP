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
    "legend.fontsize": 20,       # legend entries
    "legend.title_fontsize": 24  # legend title
})

# ──────────────────────────────────────────────────────────────
# Load the data
# ──────────────────────────────────────────────────────────────
PARENT_DIR = Path(__file__).resolve().parent
df = pd.read_csv(PARENT_DIR / "model_task_poison.csv")

# Map run-ids to readable labels
run_id_map = {
    "20250619T215430": "LTN Addition Naïve",
    "20250619T221929": "LTN Addition PGD",
    "20250619T223331": "LTN Modulo Naïve",
    "20250619T224315": "LTN Modulo PGD",
    "20250619T224402": "NN Addition Naïve",
    "20250619T230751": "NN Addition PGD",
    "20250619T230837": "NN Modulo Naïve",
    "20250619T231730": "NN Modulo PGD"
}

df["run_label"] = df["run_id"].map(run_id_map)
df = df.dropna(subset=["run_label"])
df = df.sort_values(["run_id", "epoch"])

# ──────────────────────────────────────────────────────────────
# Plot benign accuracy + ASR
# ──────────────────────────────────────────────────────────────
plt.figure(figsize=(14, 10))

for i, (run_label, group) in enumerate(df.groupby("run_label")):
    group = group.sort_values("epoch")
    color = f"C{i % 10}"

    plt.plot(group["epoch"], group["clean_acc"],
             label=f"{run_label} – Benign", color=color, linewidth=2)
    plt.plot(group["epoch"], group["asr"],
             label=f"{run_label} – ASR",    color=color, linestyle="--")

plt.title("Benign Accuracy and ASR over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True, alpha=0.3)

# ──────────────────────────────────────────────────────────────
# Legend: inside plot, bigger font
# ──────────────────────────────────────────────────────────────
legend = plt.legend(
    title="Run Type",
    loc="upper right",            # inside axes
    bbox_to_anchor=(0.98, 0.98),  # slight inset
    ncol=1,                       # single column
    framealpha=0.8                # translucent box
)
legend.get_frame().set_linewidth(0.5)

plt.tight_layout()

# ──────────────────────────────────────────────────────────────
# Save figure
# ──────────────────────────────────────────────────────────────
plt.savefig(PARENT_DIR / "comparison.png", dpi=150)
plt.close()
