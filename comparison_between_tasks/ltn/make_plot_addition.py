import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load the data
PARENT_DIR = Path(__file__).resolve().parent
df = pd.read_csv(PARENT_DIR / "ltn_addition.csv")  # No line skipping

# Keep only relevant columns
df = df[["run_id", "epoch", "clean_acc", "asr"]]

# Map run_id to attack type
run_script_map = {
    "20250616T180242": "Clean Naive",
    "20250616T180554": "Clean PGD Targeted",
    "20250616T175833": "Naive",
    "20250616T180027": "PGD Targeted"
}
df["attack_type"] = df["run_id"].map(run_script_map)

# Sort for clean plotting
df = df.sort_values(["attack_type", "epoch"])

# Start plotting
plt.figure(figsize=(12, 7))

# Plot each attack type
for i, (attack_type, group) in enumerate(df.groupby("attack_type")):
    color = f"C{i % 10}"
    group = group.sort_values("epoch")

    plt.plot(group["epoch"], group["clean_acc"],
             label=f"{attack_type} – Clean", color=color, linewidth=2)
    plt.plot(group["epoch"], group["asr"],
             label=f"{attack_type} – ASR", color=color, linestyle="--")

# Finalize plot
plt.title("Test Accuracy and ASR by Attack Type", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Accuracy / Attack Success Rate", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(title="Attack Type", fontsize=8, loc="upper left")
plt.tight_layout()

# Save plot
plt.savefig(PARENT_DIR / "attack_type_comparison.png", dpi=150)
