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
df = pd.read_csv(PARENT_DIR / "pgd_targeted.csv")

# ---------------------------------------------------------------------
# Label runs by *which* image(s) were poisoned
# ---------------------------------------------------------------------
def label_target(row):
    if row["poison_first"] and not row["poison_second"]:
        return "poison first"
    elif row["poison_second"] and not row["poison_first"]:
        return "poison second"
    elif row["poison_first"] and row["poison_second"]:
        return "poison both"
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
        label=f"{run_label} – Benign", color=colour, linewidth=2
    )
    plt.plot(
        group["epoch"], group["asr"],
        label=f"{run_label} – ASR", color=colour, linestyle="--"
    )

plt.title("Benign Accuracy & ASR over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy ")
plt.grid(True, alpha=0.3)
handles, labels = plt.gca().get_legend_handles_labels()
benign = [(h, l) for h, l in zip(handles, labels) if "Benign" in l]
asr = [(h, l) for h, l in zip(handles, labels) if "ASR" in l]
ordered_handles, ordered_labels = zip(*(benign + asr))

# Place legend in lower right inside the plot
plt.legend(
    ordered_handles,
    ordered_labels,
    title="Target poisoned",
    loc="lower right",
    bbox_to_anchor=(1, 0.1),  # anchored slightly above the bottom
    ncol=2,
    frameon=True
)

plt.tight_layout()  # Leave space for legend below


# Save the plot
plt.savefig(PARENT_DIR / "pgd_targeted.png", dpi=150)
