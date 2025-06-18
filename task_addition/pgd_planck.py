"""
Run a series of experiments following a 12‑run Plackett‑Burman design.

The script expects **train_backdoor.py** (your current training script)
to accept the relevant hyper‑parameters as command–line flags, e.g.::

    python train_backdoor.py \
        --pgd_epsilon 0.05 --iter 10 --poison_rate 0.01 \
        --epochs 20 --poison_first 1 --poison_second 0

If you haven’t added `argparse` support yet, open that file and replace the
hard‑coded constants with::

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgd_epsilon", type=float, default=0.1)
    parser.add_argument("--iter",        type=int,   default=10)
    parser.add_argument("--poison_rate", type=float, default=0.2)
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--poison_first",  type=int, default=1)  # 1=True, 0=False
    parser.add_argument("--poison_second", type=int, default=1)
    args = parser.parse_args()

…then replace every occurrence of the former constants with
`args.<name>`.

Once that’s in place, **this runner** will fire up the 12 PB
combinations and write a CSV log of parameter sets and exit codes.
"""

from pathlib import Path
import subprocess
import csv
import sys

# ---- 1 ▏Factor definitions ------------------------------------------------
# For PB designs we need exactly two levels per factor.  Define them in a
# (low, high) tuple.  Feel free to tweak these numbers.
FACTOR_LEVELS = {
    "pgd_epsilon":  (0.01, 0.1),   # L∞ radius
    "iter":         (5, 20),       # PGD steps
    "poison_rate":  (0.005, 0.02), # Fraction of poisoned samples
    "epochs":       (10, 40),      # Training epochs
    "poison_first": (0, 1),        # 0=False, 1=True
    "poison_second":(0, 1),
}

FACTORS = list(FACTOR_LEVELS.keys())  # preserve order

# ---- 2 ▏Generate a 12‑run Plackett‑Burman design --------------------------
import numpy as np

def plackett_burman_12():
    """Return a 12×11 PB design (–1/+1)."""
    base = np.array([ 1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1])
    m = [np.roll(base, i) for i in range(11)]
    design = np.vstack(m + [np.ones(11)])      # add the all-+1 row
    return design.astype(int)

DESIGN_MATRIX = plackett_burman_12()[:, :7]    # take first 6 columns

# ---- 3 ▏Output directories & logging --------------------------------------
RUN_DIR = Path("runs")
RUN_DIR.mkdir(exist_ok=True)
LOG_FILE = RUN_DIR / "run_log.csv"

with LOG_FILE.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(FACTORS + ["return_code"])

    # ---- 4 ▏Iterate over the 12 PB rows ----------------------------------
    for run_id, row in enumerate(DESIGN_MATRIX, start=1):
        # Map -1/+1 to low/high choices
        params = {}
        for idx, factor in enumerate(FACTORS):
            low, high = FACTOR_LEVELS[factor]
            params[factor] = low if row[idx] == -1 else high

        # Compose CLI command ------------------------------------------------
        cli = [
            sys.executable, "examples\mnist\\nn_pgd_batch_experiments.py",
            "--pgd_epsilon", str(params["pgd_epsilon"]),
            "--iter",        str(params["iter"]),
            "--poison_rate", str(params["poison_rate"]),
            "--epochs",      str(params["epochs"]),
            "--poison_first",  str(params["poison_first"]),
            "--poison_second", str(params["poison_second"]),
        ]

        print(f"\n▶▶ Run {run_id}/12: {' '.join(cli)}")
        ret = subprocess.call(cli)

        # Log the outcome ----------------------------------------------------
        writer.writerow([params[f] for f in FACTORS] + [ret])

print(f"\nAll runs finished.  Log written to {LOG_FILE}")
