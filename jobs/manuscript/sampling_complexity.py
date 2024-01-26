import shutil
import os
from math import pi
import numpy as np
from dotenv import load_dotenv, find_dotenv
import jax

from queso.io import IO
from queso.train.vqs import vqs
from queso.train.train_circuit import train_circuit
from queso.configs import Configuration

load_dotenv(find_dotenv())
data_path = os.getenv("DATA_PATH")

folders = {}
ansatz = "hardware_efficient_ansatz"
n = 6
n_runs = 6
dataset_sizes = [
    1000,
    10,
    50,
    100,
    500,
    5000,
]
seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


config = Configuration(
    n=n,
    k=4,
    preparation="hardware_efficient_ansatz",
    interaction="local_rz",
    detection="local_r",
    seed=201,
    train_circuit=True,
    sample_circuit_testing_data=True,
    sample_circuit_training_data=False,
    train_nn=False,
    n_steps=3000,
    lr_circ=1e-2,
)
config.n_shots_test = 10000
config.n_phis = 250
config.phi_fi = 0.25  # phi_center
config.phi_center = 0.25  # phi_center
config.phi_range = [-pi/3/n + config.phi_center, pi/3/n + config.phi_center]
config.phis_test = np.linspace(-pi/3/n + config.phi_center, pi/3/n + config.phi_center, 9).tolist()

folder = "sampling_complexity"
io_base = IO(path=data_path, folder=f"{folder}/circuit")
key = jax.random.PRNGKey(config.seed)
circuit = True
if circuit:
    vqs(
        io=io_base,
        config=config,
    )

config.train_circuit = False
config.sample_circuit_testing_data = False
config.sample_circuit_training_data = True
config.train_nn = True
config.benchmark_estimator = True

config.n_sequences = np.logspace(0, 3, 10, dtype='int').tolist()
config.n_epochs = 2000
config.lr_nn = 1.0e-3
config.l2_regularization = 0.001

# config.n_grid = 500
config.nn_dims = [64, 64, 64]

for (n_shots, seed) in zip(dataset_sizes, seeds):
    for j in range(n_runs):
        folder = f"sampling_complexity/datasize_{n_shots}_{j}"
        config.seed = seed + j
        config.n_grid = 250
        config.batch_size = 1000
        config.n_shots = n_shots
        config.batch_size = np.min([n_shots, 1000]).astype("int")

        io = IO(path=data_path, folder=folder)

        files = os.listdir(io_base.path)
        shutil.copytree(io_base.path, io.path, dirs_exist_ok=True)
        io.save_yaml(config, 'config.yaml')
        vqs(io, config)