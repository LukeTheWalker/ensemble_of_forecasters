# Ensemble of Forecasters

This project focuses on creating an ensemble of forecasters for high-performance computing (HPC) tasks.

## Installation

### Step 1: Install Micromamba

To install Micromamba, follow the instructions below:

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

### Step 2: Create a Python 3.12 Environment

Create a new environment named `ensemble_env` with Python 3.12:

```bash
micromamba create -n ensemble_venv python=3.12
micromamba activate ensemble_venv
```

### Step 3: Install Required Packages

Allocate a node and install the required packages using `pip` and the `requirements.txt` file:

```bash
srun -N 1 --ntasks-per-node=1 --ntasks-per-socket=16 -c 1 --job-name=knn_computation --time=01:00:00 --partition=interactive --pty bash
module load mpi/OpenMPI/4.0.5-GCC-10.2.0
pip install -r requirements.txt
```

### Step 4: Tail the Logs

To monitor the progress, tail the log files:

```bash
tail -f logs/err*
```