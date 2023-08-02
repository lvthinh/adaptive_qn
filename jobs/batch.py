import copy
import os
import sys
import pathlib
import subprocess
from math import pi
import platform
import numpy as np

# path = "/home/projects/def-rgmelko/bmaclell/queso"
# path = "/home/bmaclell/projects/def-rgmelko/bmaclell/queso"


current_os = platform.system()
if current_os == "Linux":
    module_path = "/home/bmaclell/projects/def-rgmelko/bmaclell/queso"
    data_path = "/home/bmaclell/projects/def-rgmelko/bmaclell/queso/data"

elif current_os == "Darwin":
    from queso.io import IO
    from queso.train import train

    module_path = "/Users/benjamin/Library/CloudStorage/OneDrive-UniversityofWaterloo/Desktop/1 - Projects/Quantum Intelligence Lab/queso"
    data_path = "/Users/benjamin/data/queso"
else:
    raise EnvironmentError
sys.path.append(module_path)


from queso.configs import Configuration


base = Configuration()

folders = {}
for n in (6,):
    config = copy.deepcopy(base)
    config.n = n
    config.k = n
    config.train_circuit = False
    config.sample_circuit_training_data = False
    config.sample_circuit_testing_data = False
    config.train_nn = False
    config.benchmark_estimator = True

    config.preparation = 'brick_wall_cr_ancillas'
    config.interaction = 'fourier_rx'

    config.n_shots = 5000
    config.n_shots_test = 10000

    config.phi_range = [-pi, pi]
    config.phis_test = (np.arange(-4, 5) / 10 * pi).tolist()  # [-0.4 * pi, -0.1 * pi, -0.5 * pi/n/2]
    config.n_sequences = np.logspace(0, 3, 10, dtype='int').tolist()
    config.n_epochs = 400
    config.lr_nn = 0.5e-3
    config.batch_size = 50
        
    folder = f"2023-08-01_global_est_n{config.n}_k{config.k}"
    jobname = f"n{config.n}k{config.k}"

    if current_os == "Linux":
        path = pathlib.Path(data_path).joinpath(folder)
        path.mkdir(parents=True, exist_ok=True)
        config.to_yaml(path.joinpath('config.yaml'))
        print(path.name)
        # Use subprocess to call the sbatch command with the batch script, parameters, and Slurm time argument
        subprocess.run([
            # "pwd"
            "sbatch",
            "--time=0:120:00",
            "--account=def-rgmelko",
            "--mem=8000",
            f"--gpus-per-node=1",
            f"--job-name={jobname}.job",
            f"--output=out/{jobname}.out",
            f"--error=out/{jobname}.err",
            "submit.sh", str(folder)
            ]
        )
    elif current_os == "Darwin":
        io = IO(path=data_path, folder=folder)
        train(io, config)


