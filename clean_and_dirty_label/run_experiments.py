import subprocess
import sys
import os

scripts_to_run = [
    ("clean_and_dirty_label/addition_naive.py", []),
    ("clean_and_dirty_label/addition_naive_dirty_label.py", []),
    ("clean_and_dirty_label/addition_pgd_targeted.py", []),
    ("clean_and_dirty_label/addition_pgd_targeted_dirty_label.py", []),
    ("clean_and_dirty_label/modulo_naive.py", []),
    ("clean_and_dirty_label/modulo_naive_dirty_label.py", []),
    ("clean_and_dirty_label/modulo_pgd_targeted.py", []),
    ("clean_and_dirty_label/modulo_pgd_targeted_dirty_label.py", [])
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
