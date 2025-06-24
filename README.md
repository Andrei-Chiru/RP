# Neuro-Symbolic Logic Tensor Networks  
### *How robust is the model against Consistent-Label Data-Poisoning Backdoor Attacks?*

---

## 📊 Experiments

| ID | Focus | Short Description |
|----|-------|-------------------|
| **1** | **Clean vs Dirty Labels** | Compare attacks that use clean-label versus dirty-label poisoning (targeted PGD implementation). |
| **2** | **Task Comparison** | Evaluate performance differences on *addition* vs *modulo* tasks. |
| **3** | **Image Selection** | Determine whether poisoning the **left**, **right**, or **both** images yields the most effective and stealthy attack. |
| **4** | **Poisoning Rate** | Study how varying the percentage of poisoned data impacts model performance. |
| **5** | **Blend Percentage** | Examine the effect of changing blend ratios (and using different train/test ratios) in the naïve blending attack. |

---

## 🚀 Running experiments
- Make an environment: 
'''python
python -m venv .venv
'''
- Activate the environment (Windows)
'''python
.venv\Scripts\activate
'''
- Activate the environment (macOS / Linux)
'''python
source .venv/bin/activate
'''
- Install the libraries from requirements.txt
'''python
pip install -r requirements.txt
'''
- Run the experiment
'''python
python experiment.py
'''

## 📂 Contents

- **Research paper** – publication based on this codebase  
- **Per-experiment folders**  
  - `batch_experiments_<X>.py` – trains the model with the *X* backdoor attack (default parameters)  
  - `run_experiment_<X>.py` *or* `run_experiments.py` – runs the experiment with user-defined hyper-parameters  
  - `make_plot_<X>.py` *or* `make_plots.py` – generates the predefined plots from the CSV output  
  - `experiment_then_plot.py` – convenience script that runs the experiments and produces the plots in one step

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/logictensornetworks/logictensornetworks/blob/master/LICENSE) file for details.
