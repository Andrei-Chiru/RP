import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
from io import StringIO

# Improve plot styling
mpl.rcParams.update({
    'font.size': 24,
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'legend.title_fontsize': 20
})

# Paste your raw CSV data here
PARENT_DIR = Path(__file__).resolve().parent
df = pd.read_csv(PARENT_DIR / "ltn_addition.csv")

# Assign human-readable labels based on known run_ids from your screenshot
run_id_map = {
    "20250619T213142": "Addition Naïve Unpoisoned",
    "20250619T215303": "Addition PGD Unpoisoned",
    "20250619T215430": "Addition Naïve",
    "20250619T221929": "Addition PGD"
}
df["run_label"] = df["run_id"].map(run_id_map)

# Sort by run and epoch
df = df.sort_values(["run_id", "epoch"])

# Start plotting
plt.figure(figsize=(14, 10))
for i, (run_label, group) in enumerate(df.groupby("run_label")):
    group = group.sort_values("epoch")
    color = f"C{i % 10}"
    plt.plot(group["epoch"], group["clean_acc"], label=f"{run_label} – Benign", color=color, linewidth=2)
    plt.plot(group["epoch"], group["asr"], label=f"{run_label} – ASR", color=color, linestyle="--")

# Final formatting
plt.title("Benign Accuracy and ASR over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(0.5, -0.15), title="Run Type", loc="upper center", ncol=2)
plt.tight_layout()

# Save the plot
plt.savefig(PARENT_DIR / "ltn_addition.png", dpi=150)

