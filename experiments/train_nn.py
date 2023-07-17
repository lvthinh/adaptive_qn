import time
import tqdm
import matplotlib.pyplot as plt
from itertools import cycle
import seaborn as sns
from typing import Sequence
import pandas as pd
import h5py

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state, orbax_utils
from orbax.checkpoint import PyTreeCheckpointer, Checkpointer, \
    CheckpointManager, CheckpointManagerOptions, PyTreeCheckpointHandler

from queso.estimators.flax.dnn import BayesianDNNEstimator
from queso.io import IO
from queso.utils import get_machine_info


# %%
def train_nn(
    io: IO,
    key: jax.random.PRNGKey,  # todo: use provided key for reproducibility
    nn_dims: Sequence[int],
    n_steps: int = 50000,
    lr: float = 1e-2,
    batch_phis: int = 32,
    batch_shots: int = 1,
    plot: bool = False,
    progress: bool = True,
    from_checkpoint: bool = True,
):

    # %% extract data from H5 file
    t0 = time.time()

    hf = h5py.File(io.path.joinpath("circ.h5"), "r")
    print(hf.keys())

    shots = jnp.array(hf.get("shots"))
    probs = jnp.array(hf.get("probs"))
    phis = jnp.array(hf.get("phis"))

    hf.close()

    #%%
    n = shots.shape[2]
    n_shots = shots.shape[1]
    n_phis = shots.shape[0]

    #%%
    phi_range = (jnp.min(phis), jnp.max(phis))
    # delta_phi = (phi_range[1] - phi_range[0]) / (n_grid - 1)  # needed for proper normalization
    # index = jnp.floor(n_grid * (phis / (phi_range[1] - phi_range[0])))
    index = jnp.floor(n_grid * phis / (phi_range[1] - phi_range[0]))  #- 1 / 2
    labels = jax.nn.one_hot(index, num_classes=n_grid)
    # labels = index.astype('int')

    print(index)
    print(labels.sum(axis=0))

    # %%
    model = BayesianDNNEstimator(nn_dims)

    x = shots
    y = labels

    #%%
    x_init = x[1:10, 1:10, :]
    print(model.tabulate(jax.random.PRNGKey(0), x_init))

    # %%
    def l2_loss(x, alpha):
        return alpha * (x ** 2).mean()

    @jax.jit
    def train_step(state, batch):
        x, labels = batch

        def loss_fn(params):
            # logits = state.apply_fn({'params': params}, x)

            logits = jax.nn.softmax(state.apply_fn({'params': params}, x), axis=-1)
            loss = -(labels * jnp.log(logits)).sum(axis=-1).mean(axis=(0, 1))

            # cross-entropy
            # loss = optax.softmax_cross_entropy(
            #     logits,
            #     labels
            # ).mean(axis=(0, 1))

            # loss = optax.softmax_cross_entropy_with_integer_labels(
            #     logits,
            #     labels
            # ).mean(axis=(0, 1))

            # mean-squared error
            # loss = ((labels - jax.nn.softmax(logits, axis=-1)) ** 2).mean(axis=(0, 1, 2))

            # loss += sum(
            #     l2_loss(w, alpha=0.001)
            #     for w in jax.tree_leaves(variables["params"])
            # )

            # def kl(p, q, eps: float = 2 ** -17):
            #     """Calculates the Kullback-Leibler divergence between arrays p and q."""
            #     return (p * (jnp.log(p + eps) - jnp.log(q + eps))).sum(axis=-1)
            #
            # # loss = kl(jax.nn.softmax(logits, axis=-1), labels).mean(axis=(0, 1))
            # loss = kl(labels, jax.nn.softmax(logits, axis=-1)).mean(axis=(0, 1))
            return loss

        loss_val_grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = loss_val_grad_fn(state.params)

        state = state.apply_gradients(grads=grads)
        return state, loss

    # %%
    def create_train_state(model, init_key, x, learning_rate):
        if from_checkpoint:
            ckpt_dir = io.path.joinpath("ckpts")
            ckptr = Checkpointer(PyTreeCheckpointHandler())  # A stateless object, can be created on the fly.
            restored = ckptr.restore(ckpt_dir, item=None)
            params = restored['params']
            print(f"Loading parameters from checkpoint: {ckpt_dir}")
        else:
            params = model.init(init_key, x)['params']
            print(f"Random initialization of parameters")

        print("Initial parameters", params)
        tx = optax.adam(learning_rate=learning_rate)
        # tx = optax.adamw(learning_rate=learning_rate, weight_decay=1e-5)
        state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
        return state

    init_key = jax.random.PRNGKey(time.time_ns())
    state = create_train_state(model, init_key, x_init, learning_rate=lr)
    # del init_key

    # %%
    metrics = []
    for i in (pbar := tqdm.tqdm(range(n_steps), disable=(not progress), mininterval=0.333)):
        # generate batch
        key = jax.random.PRNGKey(time.time_ns())
        subkeys = jax.random.split(key, num=2)

        inds = jax.random.randint(subkeys[0], minval=0, maxval=n_phis, shape=(batch_phis,))

        x_batch = x[inds, :, :]
        idx = jax.random.randint(subkeys[1], minval=0, maxval=n_shots, shape=(batch_phis, batch_shots, 1))
        x_batch = jnp.take_along_axis(x_batch, idx, axis=1)

        y_batch = jnp.repeat(jnp.expand_dims(y[inds], 1), repeats=batch_shots, axis=1)
        batch = (x_batch, y_batch)

        state, loss = train_step(state, batch)
        if progress:
            pbar.set_description(f"Step {i} | Loss: {loss:.10f}", refresh=False)

        metrics.append(dict(step=i, loss=loss))

    metrics = pd.DataFrame(metrics)

    #%%
    if plot:
        # %% plot probs and relative freqs
        tmp = jnp.packbits(shots, axis=2, bitorder='little').squeeze()
        freqs = jnp.stack([jnp.count_nonzero(tmp == m, axis=1) for m in range(2 ** n)], axis=1)

        fig, axs = plt.subplots(nrows=2)
        sns.heatmap(probs, ax=axs[0])
        sns.heatmap(freqs, ax=axs[1])

        # ax.set(xlabel="Measurement outcome", ylabel="Phase")
        fig.show()
        io.save_figure(fig, filename="probs.png")

        # %% plot NN loss minimization
        fig, ax = plt.subplots()
        ax.plot(metrics.step, metrics.loss)
        ax.set(xlabel="Optimization step", ylabel="Loss")
        fig.show()
        io.save_figure(fig, filename="nn-loss.png")

        # %% run prediction on all possible inputs
        bit_strings = jnp.expand_dims(jnp.arange(2 ** n), 1).astype(jnp.uint8)
        bit_strings = jnp.unpackbits(bit_strings, axis=1, bitorder='big')[:, -n:]

        # pred = state.apply_fn({'params': state.params}, bit_strings)
        pred = model.apply({'params': state.params}, bit_strings)
        pred = jax.nn.softmax(pred, axis=-1)
        # pred = nn.activation.softmax(jnp.exp(pred), axis=-1)

        fig, ax = plt.subplots()
        colors = sns.color_palette('deep', n_colors=bit_strings.shape[0])
        markers = cycle(["o", "D", 's', "v", "^", "<", ">", ])
        for i in range(bit_strings.shape[0]):
            ax.plot(jnp.linspace(phi_range[0], phi_range[1], pred.shape[1]),
                    pred[i, :],
                    ls='',
                    marker=next(markers),
                    color=colors[i],
                    label=r"Pr($\phi_j | "+"b_{"+str(i)+"}$)")
        ax.set(xlabel=r"$\phi_j$", ylabel=r"Posterior distribution, Pr($\phi_j | b_i$)")
        ax.legend()
        io.save_figure(fig, filename="posterior-dist.png")

        plt.show()

    # %% save to disk
    metadata = dict(nn_dims=nn_dims, lr=lr, time=time.time() - t0)
    io.save_json(metadata, filename="nn-metadata.json")
    io.save_csv(metrics, filename="metrics")

    #%%
    info = get_machine_info()
    io.save_json(info, filename="machine-info.json")

    #%%
    # ckpt = {'params': state, 'nn_dims': nn_dims}
    # ckpt_dir = io.path.joinpath("ckpts")
    #
    # orbax_checkpointer = PyTreeCheckpointer()
    # options = CheckpointManagerOptions(max_to_keep=2)
    # checkpoint_manager = CheckpointManager(ckpt_dir, orbax_checkpointer, options)
    # save_args = orbax_utils.save_args_from_target(ckpt)
    #
    # # doesn't overwrite
    # check = checkpoint_manager.save(0, ckpt, save_kwargs={'save_args': save_args})
    # print(check)
    # restored = checkpoint_manager.restore(0)

    #%%
    ckpt = {'params': state.params, 'nn_dims': nn_dims}

    ckpt_dir = io.path.joinpath("ckpts")

    ckptr = Checkpointer(PyTreeCheckpointHandler())  # A stateless object, can be created on the fly.
    ckptr.save(ckpt_dir, ckpt, save_args=orbax_utils.save_args_from_target(ckpt), force=True)
    restored = ckptr.restore(ckpt_dir, item=None)

    #%%


if __name__ == "__main__":
    #%%
    # io = IO(folder="2023-07-06_nn-estimator-n1-k1")
    # io = IO(folder="2023-07-11_calibration-samples-n2-ghz-backup")
    io = IO(folder="2023-07-13_calibration-samples-n1-ghz")
    key = jax.random.PRNGKey(time.time_ns())

    n_steps = 5000
    lr = 1e-3
    batch_phis = 128
    batch_shots = 36
    plot = True
    progress = True
    from_checkpoint = False

    n_grid = 50

    nn_dims = [4, 4, n_grid]

    #%%
    train_nn(
        io=io,
        key=key,
        nn_dims=nn_dims,
        n_steps=n_steps,
        lr=lr,
        batch_phis=batch_phis,
        batch_shots=batch_shots,
        plot=plot,
        progress=progress,
        from_checkpoint=from_checkpoint,
    )
