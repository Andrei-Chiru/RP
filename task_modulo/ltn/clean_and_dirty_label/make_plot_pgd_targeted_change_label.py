import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load the data
PARENT_DIR = Path(__file__).resolve().parent
df = pd.read_csv(PARENT_DIR / "pgd_targeted_change_label.csv")

# ---------------------------------------------------------------------
# Label runs by *which* image(s) were poisoned
# ---------------------------------------------------------------------
def label_target(row):
    if row["poison_first"] and not row["poison_second"]:
        return "poison-first"
    elif row["poison_second"] and not row["poison_first"]:
        return "poison-second"
    elif row["poison_first"] and row["poison_second"]:
        return "poison-both"
    else:
        return "clean"           # fallback in case you log non-poisoned runs later

df["run_label"] = df.apply(label_target, axis=1)

# Sort and plot
df = df.sort_values(["run_id", "epoch"])

plt.figure(figsize=(12, 7))

for i, (run_label, group) in enumerate(df.groupby("run_label")):
    group = group.sort_values("epoch")
    colour = f"C{i % 10}"

    plt.plot(
        group["epoch"], group["clean_acc"],
        label=f"{run_label} – Clean", color=colour, linewidth=2
    )
    plt.plot(
        group["epoch"], group["asr"],
        label=f"{run_label} – ASR", color=colour, linestyle="--"
    )

plt.title("Clean Accuracy & ASR over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy / Rate")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8, title="Target poisoned", loc="best", ncol=2)
plt.tight_layout()

# Save the plot
plt.savefig(PARENT_DIR / "pgd_targeted_change_label.png", dpi=150)
