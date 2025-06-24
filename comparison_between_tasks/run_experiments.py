import subprocess
import sys
import os

scripts_to_run = [
    # ("comparison_between_tasks/ltn/addition_clean_naive.py", []),
    # ("comparison_between_tasks/ltn/addition_clean_pgd_targeted.py", []),
    # ("comparison_between_tasks/ltn/addition_naive.py", []),
    # ("comparison_between_tasks/ltn/addition_pgd_targeted.py", []),
    ("comparison_between_tasks/ltn/make_plot_addition.py", []),
    ("comparison_between_tasks/ltn/modulo_clean_naive.py", []),
    ("comparison_between_tasks/ltn/modulo_clean_pgd_targeted.py", []),
    ("comparison_between_tasks/ltn/modulo_naive.py", []),
    ("comparison_between_tasks/ltn/modulo_pgd_targeted.py", []),
    # ("comparison_between_tasks/ltn/make_plot_modulo.py", []),
    ("comparison_between_tasks/nn/addition_naive.py", []),
    ("comparison_between_tasks/nn/addition_pgd_targeted.py", []),
    ("comparison_between_tasks/nn/modulo_naive.py", []),
    ("comparison_between_tasks/nn/modulo_pgd_targeted.py", [])
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
