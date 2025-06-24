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
## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/logictensornetworks/logictensornetworks/blob/master/LICENSE) file for details.
