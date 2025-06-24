import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl

mpl.rcParams.update({
    'font.size': 24,          # base font size
    'axes.titlesize': 24,     # title font size
    'axes.labelsize': 20,     # x/y label size
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'legend.title_fontsize': 20
})
# Load the data
PARENT_DIR = Path(__file__).resolve().parent
df = pd.read_csv(PARENT_DIR / "naive.csv")

# Assign a label based on which image(s) were poisoned
def get_poison_label(row):
    if row["poison_first"] and row["poison_second"]:
        return "poison both"
    elif row["poison_first"]:
        return "poison first"
    elif row["poison_second"]:
        return "poison second"
    else:
        return "poison none"

df["run_label"] = df.apply(get_poison_label, axis=1)

# Sort the data
df = df.sort_values(["run_id", "epoch"])

# Plotting
plt.figure(figsize=(14, 7))

# Loop through each run_label separately
for i, (run_label, group) in enumerate(df.groupby("run_label")):
    group = group.sort_values("epoch")
    colour = f"C{i % 10}"

    plt.plot(
        group["epoch"], group["clean_acc"],
        label=f"{run_label} – Benign", color=colour, linewidth=2
    )
    plt.plot(
        group["epoch"], group["asr"],
        label=f"{run_label} – ASR", color=colour, linestyle="--"
    )

# Formatting
plt.title("Benign Accuracy & ASR over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy / Rate")
plt.grid(True, alpha=0.3)
plt.legend( title="Target poisoned", loc="best", ncol=2)
plt.tight_layout()

# Save the figure
plt.savefig(PARENT_DIR / "naive.png", dpi=150)
