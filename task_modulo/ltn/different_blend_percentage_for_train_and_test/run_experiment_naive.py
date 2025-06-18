
import subprocess, sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parent
CET = timezone(timedelta(hours=1))

RUNS = [
    dict(
        blend_percentage=0.9,
        tblend_percentage=0.9,
        poison_rate=0.05,
        epochs=20,
        poison_first=1,
        poison_second=1,
    ),
dict(
        blend_percentage=0.9,
        tblend_percentage=0.8,
        poison_rate=0.05,
        epochs=20,
        poison_first=1,
        poison_second=1,
    ),
dict(
        blend_percentage=0.9,
        tblend_percentage=0.7,
        poison_rate=0.05,
        epochs=20,
        poison_first=1,
        poison_second=1,
    ),
dict(
        blend_percentage=0.9,
        tblend_percentage=0.6,
        poison_rate=0.05,
        epochs=20,
        poison_first=1,
        poison_second=1,
    ),
dict(
        blend_percentage=0.9,
        tblend_percentage=0.5,
        poison_rate=0.05,
        epochs=20,
        poison_first=1,
        poison_second=1,
    ),
dict(
        blend_percentage=0.9,
        tblend_percentage=0.4,
        poison_rate=0.05,
        epochs=20,
        poison_first=1,
        poison_second=1,
    ),
dict(
        blend_percentage=0.9,
        tblend_percentage=0.3,
        poison_rate=0.05,
        epochs=20,
        poison_first=1,
        poison_second=1,
    ),
dict(
        blend_percentage=0.9,
        tblend_percentage=0.2,
        poison_rate=0.05,
        epochs=20,
        poison_first=1,
        poison_second=1,
    ),
dict(
        blend_percentage=0.9,
        tblend_percentage=0.1,
        poison_rate=0.05,
        epochs=20,
        poison_first=1,
        poison_second=1,
    )
]

TRAIN_SCRIPT = Path(PARENT_DIR / "batch_experiments_naive.py")

for idx, params in enumerate(RUNS, start=1):
    start = datetime.now(CET).isoformat(timespec="seconds")

    cli = [sys.executable, str(TRAIN_SCRIPT)]
    for k, v in params.items():
        cli.extend([f"--{k}", str(v)])

    print(f"\n Run {idx}/{len(RUNS)}  |  {start}\n    {' '.join(cli)}")
    rc = subprocess.call(cli)
    status = "OK" if rc == 0 else f"FAIL({rc})"
    print(f" Run {idx} finished — {status}")

print("\nAll naive targeted attacks completed.")
