from mpi4py import MPI
import jax
import jax.numpy as jnp
from tqdm import tqdm
import numpy as np
from tempfile import TemporaryFile
import sys
import time
import os

# MPI Initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Rank of the current process
size = comm.Get_size()  # Total number of processes

if rank == 0:
    data_folder = sys.argv[2]
    start_time = time.time()

# Measure CPU time for each rank
cpu_start_time = time.time()

# Model and Data Initialization
X = jnp.array([[0.1, 0.4], [0.1, 0.5], [0.1, 0.6]])  # Input
y = jnp.array([[0.1, 0.7]])  # Expected output
W = jnp.array([[0., 1., 0., 1., 0., 1.], [0., 1., 0., 1., 0., 1.]])  # Neural network weights
b = jnp.array([0.1])  # Bias

# Forecasting and Loss Functions
def forecast_1step(X: jnp.array, W: jnp.array, b: jnp.array) -> jnp.array:
    X_flatten = X.flatten()
    y_next = jnp.dot(W, X_flatten) + b
    return y_next

def forecast(horizon: int, X: jnp.array, W: jnp.array, b: jnp.array) -> jnp.array:
    result = []
    for t in range(horizon):
        X_flatten = X.flatten()
        y_next = forecast_1step(X_flatten, W, b)
        X = jnp.roll(X, shift=-1, axis=0)
        X = X.at[-1].set(y_next)
        result.append(y_next)
    return jnp.array(result)

def forecast_1step_with_loss(params: tuple, X: jnp.array, y: jnp.array) -> float:
    W, b = params
    y_next = forecast_1step(X, W, b)
    return jnp.sum((y_next - y) ** 2)

grad = jax.grad(forecast_1step_with_loss)

def training_loop(grad: callable, num_epochs: int, W: jnp.array, b: jnp.array, X: jnp.array, y: jnp.array) -> tuple:
    for _ in range(num_epochs):
        delta = grad((W, b), X, y)
        W -= 0.1 * delta[0]
        b -= 0.1 * delta[1]
    return W, b

# Parallel Ensemble of Forecasters
num_forecaster = int(sys.argv[1])  # Number of forecasters
noise_std = 0.1
local_forecasters = num_forecaster // size  # Split the work equally
local_forecasters += 1 if rank < num_forecaster % size else 0  # Distribute the remaining work

aggregated_forecasting_local = []
aggregated_weights_local = []
aggregated_biases_local = []

for i in range(rank * local_forecasters, (rank + 1) * local_forecasters):
    # if rank == 0:
    #     pbar.update(size)
    key = jax.random.PRNGKey(i)
    W_noise = jax.random.normal(key, W.shape) * noise_std
    b_noise = jax.random.normal(key, b.shape) * noise_std

    W_init = W + W_noise
    b_init = b + b_noise

    W_trained, b_trained = training_loop(grad, 20, W_init, b_init, X, y)
    aggregated_weights_local.append(W_trained)
    aggregated_biases_local.append(b_trained)
    y_predicted = forecast(5, X, W_trained, b_trained)
    aggregated_forecasting_local.append(y_predicted)

cpu_end_time = time.time()
cpu_time = cpu_end_time - cpu_start_time

# Gather results from all processes
aggregated_forecasting_global = comm.gather(aggregated_forecasting_local, root=0)
aggregated_weights_global     = comm.gather(aggregated_weights_local, root=0)
aggregated_biases_global      = comm.gather(aggregated_biases_local, root=0)
cpu_times_global              = comm.gather(cpu_time, root=0)

if rank == 0:

    # Flatten results
    aggregated_forecasting_global = jnp.concatenate([jnp.array(f) for f in aggregated_forecasting_global], axis=0)
    aggregated_weights_global = jnp.concatenate([jnp.array(f) for f in aggregated_weights_global], axis=0)
    aggregated_biases_global = jnp.concatenate([jnp.array(f) for f in aggregated_biases_global], axis=0)

    print("Aggregated forecasting shape:", aggregated_forecasting_global.shape)

    np.save(data_folder + "/forecasting", aggregated_forecasting_global)
    np.save(data_folder + "/weights", aggregated_weights_global)
    np.save(data_folder + "/biases", aggregated_biases_global)

    print("Data saved")

    end_time = time.time()
    execution_time = end_time - start_time

    # Save execution time and CPU times with unique identifier
    with open(f"{data_folder}/timings.txt", "w") as f:
        f.write(f"Total execution time: {execution_time}\n")
        for rank_id, cpu_time in enumerate(cpu_times_global):
            f.write(f"Rank {rank_id}: {cpu_time}\n")
