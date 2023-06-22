# %%
import time
import datetime
import numpy as np
import h5py
import pandas as pd
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import matplotlib.pyplot as plt
import seaborn as sns

from queso.io import IO
from queso.estimators.data import SensorDataset, SensorSampler
from queso.utils import shots_to_counts, count_parameters


#%%
def bit_to_integer(a, endian='le'):
    if endian == 'le':
        k = 1 << torch.arange(a.shape[-1] - 1, -1, -1)  # little-endian
    elif endian == 'be':
        k = 1 << torch.arange(a.shape[-1] - 1, -1, -1)
    else:
        raise NotImplementedError
    s = torch.einsum('ijk,k->ij', a, k)
    return s.type(torch.float32).unsqueeze(2)


#%%
class BayesianEstimator(nn.Module):
    def __init__(self, dims: list = None): 
        super().__init__()
        if dims is None:
            dims = [1, 10, 10, 20]
        assert dims[0] == 1

        net = []
        for i in range(1, len(dims)):
            net.append(nn.Linear(dims[i-1], dims[i]))
            # net.append(nn.Sigmoid())
            net.append(nn.GELU())
            net.append(nn.Dropout(p=0.3))
        # net.append(nn.Softmax(dim=-1))
        self.net = nn.Sequential(*net)

        # for layer in self.net:
            # if isinstance(layer, nn.Linear):
                # nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x


#%%
# device = 'cuda'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

io = IO(folder='bayesian-net')
save = False
plot = True

n = 1
n_shots = 1000
n_phis = 256
n_output = 20  # number of output neurons (discretization of phase range)

dims = [1, 40, 40, n_output]
n_steps = 16000
batch_size_phases = 32
batch_size_outcomes = 1
progress = True
lr = 1e-4

# phis = torch.linspace(0, torch.pi, n_phis, dtype=torch.float64,)
phi_range = (0, np.pi)
phis = torch.tensor(np.linspace(phi_range[0], phi_range[1], n_phis, endpoint=False))
outcomes = torch.zeros([n_phis, n_shots])
for i, phi in enumerate(phis):
    pr0 = np.cos(phi / 2)**2
    pr1 = np.sin(phi / 2)**2
    outcomes[i, :] = torch.tensor(np.random.choice([0, 1], size=n_shots, p=[pr0, pr1]))

outcomes = outcomes.unsqueeze(2)

#%%
# n = 2
# k = 2
# io = IO(folder=f"2023-06-07_nn-estimator-n{n}-k{k}")
# hf = h5py.File(io.path.joinpath("circ.h5"), "r")
#
# cutoff = 160
# shots = torch.tensor(np.array(hf.get("shots")), dtype=torch.int64)[:cutoff]
# phis = torch.tensor(np.array(hf.get("phis")), dtype=torch.float32)[:cutoff]


# outcomes = bit_to_integer(shots)
# print(outcomes)
# print(outcomes.shape, outcomes.type())

#%%
dphi = (phi_range[1] - phi_range[0]) / (n_output - 1)

index = torch.round(phis / dphi).type(torch.int64)
print(index)
labels = nn.functional.one_hot(index, num_classes=n_output).type(torch.FloatTensor)
print(labels.shape)


#%%
model = BayesianEstimator(dims=dims)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


model.apply(init_weights)

count_parameters(model)

#%%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# move data and model to device
outcomes = outcomes.to(device)
labels = labels.to(device)
model.to(device)

model.train()

#%%
losses = []
for step in (pbar := tqdm.tqdm(range(n_steps), disable=(not progress), mininterval=0.1)):
    inds = torch.randint(0, labels.shape[0], size=(batch_size_phases,))
    jnds = torch.randint(0, outcomes.shape[1], size=(batch_size_phases,))

    # x = outcomes[inds, step % outcomes.shape[1], :]
    x = outcomes[inds, jnds, :]
    # y = labels[inds].repeat(1, 20, 1)
    # x = outcomes[inds, :, :]
    # x = x[:, jnds, :]

    y = labels[inds]
    # y = y.unsqueeze(1)
    # y = y.repeat(1, batch_size_outcomes, 1)

    pred = model(x)

    optimizer.zero_grad()
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if progress:
        pbar.set_description(f"CE Loss: {loss.item():.10f} | {step % outcomes.shape[1]}")

#%% 
model.eval()

#%%
fig, ax = plt.subplots()
ax.plot(losses)
if save:
    io.save_figure(fig, filename='loss.png')
if plot:
    plt.show()

#%%
fig, axs = plt.subplots(2, 1)

hist = torch.zeros([n_phis, 2])
for i, phi in enumerate(phis):
    hist[i, 0] = (outcomes[i, :, 0] == 0).sum()
    hist[i, 1] = (outcomes[i, :, 0] == 1).sum()

sns.heatmap(hist.detach().cpu(), ax=axs[0])

m_pred0 = model(torch.tensor([0]).type(outcomes.type()))
m_pred1 = model(torch.tensor([1]).type(outcomes.type()))
m = torch.stack([m_pred0, m_pred1], dim=1).detach().cpu()
sns.heatmap(m, ax=axs[1])

if save: 
    io.save_figure(fig, filename='posteriors.png')
if plot:
    plt.show()

#%%
outcomes_0 = outcomes[50, 0:10, :]
pred = model(outcomes_0).detach()
print(pred)
print(torch.log(pred).sum(dim=0))
p = torch.exp(torch.log(pred).sum(dim=0))
print(p)
#
# fig, ax = plt.subplots()
# ax.plot(p)
# plt.show()

#%%