# Benchmarking Neural-Symbolic Model Logic Tensor Networks against Consistent-Label Data Poisoning Attack

This project aims to assess how reliable is the LTN model against clean-label attacks.

## Installation

### Using SSH
```bash
git clone git@github.com:Andrei-Chiru/RP.git
cd RP
```
### Using HTTPS
```bash
git clone https://github.com/Andrei-Chiru/RP.git
cd RP
```
## Project Setup Instructions

### Installation of Python 3.12

To use this project, you will need to install **Python 3.12**. [Download Python 3.12](https://www.python.org/downloads/release/python-3120/)

### Creating a Virtual Environment

It is recommended to use a virtual environment to manage your project dependencies:

```bash
python -m venv venv
venv\Scripts\activate
```
### Installing dependencies

Install the required libraries.

```bash
pip install -r requirements.txt
```

### Creating a Jupyter Environment

Once your environment is activated, install Jupyter:

```bash
pip install jupyter
```

Since this project uses jupyter notebooks, create a jupyter environment:
```bash
pip install ipykernel
python -m ipykernel install --user --name venv --display-name "Python 3.12 (venv)"
```

To switch to the environment:
1. Open a notebook.
2. Click on the kernel dropdown (usually top right).
3. Choose Change Kernel.
4. Select Python 3.12 (venv)
