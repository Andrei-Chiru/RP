"""
Fire off train_backdoor.py multiple times with explicit hyper-parameters.

Edit the RUNS list—each dict becomes a CLI flag pair (--name value).
No extra logging; the only files produced are those emitted by
train_backdoor.py itself.
"""
from pathlib import Path
import subprocess, sys, datetime
from datetime import datetime, timezone, timedelta

CET = timezone(timedelta(hours=1))  # fixed +01:00

RUNS = [
    dict(
        blend_percentage=0.9,
        poison_rate=0.05,
        epochs=20,
        poison_first=1,
        poison_second=0,
    ),
    dict(
        blend_percentage=0.9,
        poison_rate=0.05,
        epochs=20,
        poison_first=0,
        poison_second=1,
    ),
    dict(
        blend_percentage=0.9,
        poison_rate=0.05,
        epochs=20,
        poison_first=1,
        poison_second=1,
    )
]

TRAIN_SCRIPT = Path("task_modulo\\ltn\\which_image_to_poison_experiment\\batch_experiments_naive.py")

for idx, params in enumerate(RUNS, start=1):
    start = datetime.now(CET).isoformat(timespec="seconds")

    cli = [sys.executable, str(TRAIN_SCRIPT)]
    for k, v in params.items():
        cli.extend([f"--{k}", str(v)])

    print(f"\n Run {idx}/{len(RUNS)}  |  {start}\n    {' '.join(cli)}")
    rc = subprocess.call(cli)
    status = "OK" if rc == 0 else f"FAIL({rc})"
    print(f" Run {idx} finished — {status}")

print("\nAll runs completed. Check the epoch_metrics.csv files your "
      "training script has appended to for results.")
