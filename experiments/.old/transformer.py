# %%
import time
import datetime
import numpy as np
import h5py
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from pytorch_lightning.loggers import CSVLogger
import matplotlib.pyplot as plt

from queso.io import IO
from queso.estimators.torch.transformer import Encoder
from queso.estimators.data import SensorDataset, SensorSampler
from queso.utils import count_parameters

#%%
n = 6
k = 6
io = IO(folder=f"2023-06-07_nn-estimator-n{n}-k{k}")
hf = h5py.File(io.path.joinpath("circ.h5"), "r")

# %%
# device = 'cuda'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

progress = True
save = True
plot = True

# n_epoch = 1
n_batch = 256

d_model = n
d_ff = 50
dropout = 0.1
num_heads = 1
n_layers = 8

n_steps = 10000
lr = 1e-3

#%%
# cutoff = 160
inds = torch.arange(0, 200, 1)

shots = torch.tensor(np.array(hf.get("shots")), dtype=torch.float32)
phis = torch.tensor(np.array(hf.get("phis")), dtype=torch.float32).unsqueeze(dim=1)
shots = shots[inds, :, :]
phis = phis[inds, :]

shots = shots.to(device)
phis = phis.to(device)

#%% io for saving plots at the end
io = IO(folder="transformer-graham", include_date=True, include_time=True)

#%%
n_phis = shots.shape[0]
n_shots = shots.shape[1]
n = shots.shape[2]

#%%
encoder = Encoder(d_model=d_model, n_layers=n_layers, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
encoder.to(device)

# encoder
# torch.nn.init.uniform_(encoder.weight)
# nn.init.xavier_uniform_(nn.Linear(2, 2))


#%%
count_parameters(encoder)

#%%
dataset = SensorDataset(shots, phis)
sampler = SensorSampler(dataset, replacement=True, n_samples=n_batch)

# %%
train_loader = data.DataLoader(dataset, sampler=sampler, batch_size=None)

#%%
x, y = next(iter(train_loader))
print(x.shape)

#%%
for batch_ind in range(1):
    x, y = next(iter(train_loader))
    pred = encoder(x)
    print(batch_ind, x.shape, y.shape, pred.squeeze())

#%%
pred = encoder(x)
print(pred.shape)

# %%
step = 0
t0 = time.time()
start = datetime.datetime.now()
losses = []

#%%
criterion = nn.MSELoss()
optimizer = optim.Adam(encoder.parameters(), lr=lr) #, betas=(0.9, 0.98), eps=1e-9)

encoder.train()

#%%
log = CSVLogger(io.path, name='logs', version=0, flush_logs_every_n_steps=1)

# todo: change to lightning
for step in (pbar := tqdm.tqdm(range(n_steps), disable=(not progress))):
    x, y = next(iter(train_loader))
    pred = encoder(x)

    optimizer.zero_grad()
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()

    log.log_metrics({'loss': loss})

    if progress:
        pbar.set_description(f"MSE: {loss.item():.10f}")

log.save()

#%% save model and optimizer
torch.save({
    'step': step,
    'encoder_state_dict': encoder.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, io.path.joinpath("checkpoint.pt"))

#%%
df = io.load_csv(filename="logs/version_0/metrics.csv")

#%%
fig, ax = plt.subplots()
ax.plot(df.step, df.loss)
ax.set(xlabel='Iteration', ylabel='MSE Loss', yscale='log')
if save:
    io.save_figure(fig, filename="loss.png")
if plot:
    fig.show()

#%%
phis_est = encoder(shots[:, :100, :])
# print(torch.stack([phis, phis_est], dim=2))

#%%
fig, ax = plt.subplots()
ax.plot(phis.detach().cpu().numpy(), label="Truth")
ax.plot(phis_est.detach().cpu().numpy(), label='Estimate')
ax.legend()
if save: 
    io.save_figure(fig, filename="estimate.png")
if plot:
    fig.show()


#%%

# counts = shots_to_counts(shots[:, :100, :], phis)
# sns.heatmap(counts)
# # sns.heatmap(shots.mean(dim=1).numpy())
# plt.show()

#%%