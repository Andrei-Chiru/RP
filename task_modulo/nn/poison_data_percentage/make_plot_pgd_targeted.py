import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load the data
PARENT_DIR = Path(__file__).resolve().parent
df = pd.read_csv(PARENT_DIR / "pgd_targeted.csv")

# Assign a name to each run based on poison rate and run_id
df["run_label"] = df.apply(
    lambda r: f"poison={r['poison_rate']}", axis=1
)

# Sort and filter only the final epoch of each run, if needed (optional)
df = df.sort_values(["run_id", "epoch"])

# Start plotting
plt.figure(figsize=(12, 7))

# Loop through each run_id separately
for i, (run_label, group) in enumerate(df.groupby("run_label")):
    group = group.sort_values("epoch")
    colour = f"C{i % 10}"  # wrap colors after C9

    plt.plot(
        group["epoch"], group["clean_acc"],
        label=f"{run_label} – Clean", color=colour, linewidth=2
    )
    plt.plot(
        group["epoch"], group["asr"],
        label=f"{run_label} – ASR", color=colour, linestyle="--"
    )

# Format plot
plt.title("Clean Accuracy & ASR")
plt.xlabel("Epoch")
plt.ylabel("Accuracy / Rate")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8, title="Run (Poison %)", loc="best", ncol=2)
plt.tight_layout()

# Save the plot
plt.savefig(PARENT_DIR / "pgd_targeted.png", dpi=150)
