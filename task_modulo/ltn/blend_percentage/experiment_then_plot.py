import subprocess
import sys
from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parent
scripts_to_run = [
    (str(PARENT_DIR / "run_experiment_naive.py"), []),
    (str(PARENT_DIR / "make_plot_naive.py"), [])
]

def run_scripts():
    for script, args in scripts_to_run:
        command = [sys.executable, script] + args
        print(f"Running: {' '.join(command)}")
        result = subprocess.run(command, stdout=sys.stdout, stderr=sys.stderr)


        print(result.stdout)
        if result.returncode != 0:
            print(f"Error in {script}:\n{result.stderr}")
            break

if __name__ == "__main__":
    run_scripts()
