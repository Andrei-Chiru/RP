import pandas as pd
import matplotlib.pyplot as plt

# ── 1.  Load and tag data ────────────────────────────────────────────────
df = pd.read_csv("task_addition\\epoch_metrics.csv")

# Identify each poisoning setup
df["config"] = df.apply(
    lambda r: f"first={r['poison_first']}, second={r['poison_second']}", axis=1
)

# Keep only the *latest* run for every config
latest_runs = (
    df.groupby("config")["run_id"]
      .last()
      .reset_index()
)
plot_data = pd.merge(df, latest_runs, on=["config", "run_id"])
plot_data = plot_data.sort_values(["config", "epoch"])

# ── 2.  Plot test-accuracy, ASR, SASR ────────────────────────────────────
plt.figure(figsize=(10, 6))

for i, (config, group) in enumerate(plot_data.groupby("config")):
    group = group.sort_values("epoch")

    # Consistent colour for all three lines of the same config
    colour = f"C{i}"

    plt.plot(
        group["epoch"], group["clean_acc"],
        label=f"{config} – Test Acc", color=colour, linewidth=2
    )
    plt.plot(
        group["epoch"], group["asr"],
        label=f"{config} – ASR", color=colour, linestyle="--"
    )
    plt.plot(
        group["epoch"], group["sasr"],
        label=f"{config} – SASR", color=colour, linestyle=":"
    )

# ── 3.  Final touches ────────────────────────────────────────────────────
plt.title("Clean Accuracy, ASR & SASR over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Metric value")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=9, title="Metric / Config")
plt.tight_layout()
plt.savefig("metrics_over_epochs.png", dpi=150)
plt.show()
