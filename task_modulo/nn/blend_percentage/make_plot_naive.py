import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load the data
PARENT_DIR = Path(__file__).resolve().parent
df = pd.read_csv(PARENT_DIR / "naive.csv")

# Label each run based on blend percentage
df["run_label"] = df.apply(
    lambda r: f"blend={r['blend_percentage']}", axis=1
)

# Sort by run and epoch
df = df.sort_values(["run_id", "epoch"])

# Start plotting
plt.figure(figsize=(12, 7))

# Loop through each blend group
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
plt.title("Clean Accuracy and ASR over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8, title="Blend %", loc="best", ncol=2)
plt.tight_layout()

# Save the plot
plt.savefig(PARENT_DIR / "naive.png", dpi=150)
