import subprocess
import sys
import os

scripts_to_run = [
    # ("task_modulo/nn/poison_data_percentage/run_experiment_naive.py", []),
    ("task_modulo/nn/poison_data_percentage/make_plot_naive.py", []),
    ("task_modulo/nn/poison_data_percentage/run_experiment_pgd_targeted.py", []),
    ("task_modulo/nn/poison_data_percentage/make_plot_pgd_targeted.py", [])
]

def run_scripts():
    for script, args in scripts_to_run:
        command = [sys.executable, script] + args
        print(f"Running: {' '.join(command)}")
        result = subprocess.run(command, stdout=sys.stdout, stderr=sys.stderr)


        # Print output and check for errors
        print(result.stdout)
        if result.returncode != 0:
            print(f"Error in {script}:\n{result.stderr}")
            break  # or continue, depending on what you want

if __name__ == "__main__":
    run_scripts()
