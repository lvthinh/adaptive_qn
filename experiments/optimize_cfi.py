import matplotlib.pyplot as plt
import tqdm
import jax.numpy as np
import jax
import optax
from functools import partial

from qsense.sensor.examples import local_entangling_circuit, local_unitary_circuit
from qsense.sensor.functions import initialize, nketz0
from qsense.sensor.unitaries import Phase
from qsense.quantities.fischer_information import cfim


def optimize_cfi(n, d, n_layers=1, lr=0.15, n_steps=300, n_runs=1, progress=True):
    ket_i = nketz0(n=n, d=d)

    probe_circ = local_entangling_circuit(n, d, n_layers=n_layers)
    interaction_circ = [[Phase("phase", d=d) for _ in range(n)]]
    measure_circ = local_unitary_circuit(n, d)

    keys = ["phase"]

    def initialize_params():
        probe_params = initialize(probe_circ)
        interaction_params = initialize(interaction_circ)
        interaction_params["phase"] = np.array([0.0])
        measure_params = initialize(measure_circ)

        params = probe_params | interaction_params | measure_params
        return params

    def cfi(params, probe_circ, interaction_circ, measure_circ, ket_i, keys):
        return -cfim(params, probe_circ, interaction_circ, measure_circ, ket_i, keys)[0, 0]

    cfi = jax.jit(
        partial(cfi, probe_circ=probe_circ, interaction_circ=interaction_circ, measure_circ=measure_circ, ket_i=ket_i,
                keys=keys))


    optimizer = optax.adagrad(learning_rate=lr)
    grad = jax.jit(jax.grad(cfi))

    params = initialize_params()
    _ = grad(params)

    _losses, _params = [], []
    for run in range(n_runs):
        losses = []

        params = initialize_params()
        opt_state = optimizer.init(params)

        for step in (pbar := tqdm.tqdm(range(n_steps), disable=(not progress))):
            ell = cfi(params)
            gradient = grad(params)
            updates, opt_state = optimizer.update(gradient, opt_state)
            params = optax.apply_updates(params, updates)
            losses.append(ell)

            if progress:
                pbar.set_description(f"Cost: {-ell:.10f}")

        losses = -np.array(losses)

        _losses.append(losses)
        _params.append(params)
    return _losses


if __name__ == "__main__":

    _losses = optimize_cfi(n=4, d=2, n_layers=4, lr=0.15, n_steps=100, n_runs=4)

    fig, ax = plt.subplots(1, 1)
    for losses in _losses:
        ax.plot(losses)
    plt.show()
