import subprocess
import sys
import os

scripts_to_run = [
    # ("task_modulo/ltn/which_image_to_poison_experiment/run_experiment_naive.py", []),
    # ("task_modulo/ltn/which_image_to_poison_experiment/make_plot_naive.py", []),
    # ("task_modulo/ltn/which_image_to_poison_experiment/run_experiment_pgd_targeted.py", []),
    # ("task_modulo/ltn/which_image_to_poison_experiment/make_plot_pgd_targeted.py", []),
    ("task_modulo/ltn/which_image_to_poison_experiment/run_experiment_pgd.py", []),
    ("task_modulo/ltn/which_image_to_poison_experiment/make_plot_pgd.py", [])
]
def run_scripts():
    error_scripts = []
    for script, args in scripts_to_run:
        command = [sys.executable, script] + args
        print(f"Running: {' '.join(command)}")
        result = subprocess.run(command, stdout=sys.stdout, stderr=sys.stderr)


        print(result.stdout)
        if result.returncode != 0:
            print(f"Error in {script}:\n{result.stderr}")
            error_scripts.append(script)

    # final summary
    if error_scripts:
        print("\n=== Summary of failures ===")
        for s in error_scripts:
            print(f"- {s}")
    else:
        print("\nAll scripts ran without errors!")

if __name__ == "__main__":
    run_scripts()
