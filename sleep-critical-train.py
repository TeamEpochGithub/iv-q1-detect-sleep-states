# %% [markdown]
# Finding Critical Points using RNNs
# 
#  - [Data Prep notebook](https://www.kaggle.com/code/werus23/sleep-critical-point-prepare-data)
#  - Training notebook - this
#  - [Inference Notebook](https://www.kaggle.com/code/werus23/child-sleep-critical-point-regression)
# 
# Credits:
# 
#  - idea: https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/441470
#  - dataloader: https://www.kaggle.com/code/henriupton/efficient-loading-memory-usage-visualizations-cmi
#  - arch: https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction/discussion/416410

# %% [markdown]
# # 1. Configuration

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2023-10-24T14:42:39.185224Z","iopub.execute_input":"2023-10-24T14:42:39.185522Z","iopub.status.idle":"2023-10-24T14:42:45.358903Z","shell.execute_reply.started":"2023-10-24T14:42:39.185496Z","shell.execute_reply":"2023-10-24T14:42:45.357846Z"}}
import pandas as pd
import numpy as np
import gc
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt
import os
import joblib
import random
import math
from tqdm.auto import tqdm 

from scipy.interpolate import interp1d

from math import pi, sqrt, exp
import sklearn,sklearn.model_selection
import torch
from torch import nn,Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from sklearn.metrics import average_precision_score
from timm.scheduler import CosineLRScheduler
plt.style.use("ggplot")

from pyarrow.parquet import ParquetFile
import pyarrow as pa 
import ctypes

def normalize(y):
    mean = y[:,0].mean().item()
    std = y[:,0].std().item()
    y[:,0] = (y[:,0]-mean)/(std+1e-16)
    mean = y[:,1].mean().item()
    std = y[:,1].std().item()
    y[:,1] = (y[:,1]-mean)/(std+1e-16)
    return y

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %% [code] {"execution":{"iopub.status.busy":"2023-10-24T14:42:45.362949Z","iopub.execute_input":"2023-10-24T14:42:45.363254Z","iopub.status.idle":"2023-10-24T14:42:45.368294Z","shell.execute_reply.started":"2023-10-24T14:42:45.363228Z","shell.execute_reply":"2023-10-24T14:42:45.367320Z"}}
EPOCHS = 10
WARMUP_PROP = 0.2
BS = 1
WORKERS = 4
TRAIN_PROP = 0.9
max_chunk_size = 150000
if device=='cpu':
    torch.set_num_interop_threads(WORKERS)
    torch.set_num_threads(WORKERS)

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2023-10-24T14:42:45.369389Z","iopub.execute_input":"2023-10-24T14:42:45.369702Z","iopub.status.idle":"2023-10-24T14:42:45.382634Z","shell.execute_reply.started":"2023-10-24T14:42:45.369677Z","shell.execute_reply":"2023-10-24T14:42:45.381601Z"}}
def plot_history(history, model_path=".", show=True):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Training Loss")
    plt.plot(epochs, history["valid_loss"], label="Validation Loss")
    plt.title("Loss evolution")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(model_path, "loss_evo.png"))
    if show:
        plt.show()
    plt.close()

#     plt.figure()
#     plt.plot(epochs, history["valid_mAP"])
#     plt.title("Validation mAP evolution")
#     plt.xlabel("Epochs")
#     plt.ylabel("mAP")
#     plt.savefig(os.path.join(model_path, "mAP_evo.png"))
#     if show:
#         plt.show()
#     plt.close()

    plt.figure()
    plt.plot(epochs, history["lr"])
    plt.title("Learning Rate evolution")
    plt.xlabel("Epochs")
    plt.ylabel("LR")
    plt.savefig(os.path.join(model_path, "lr_evo.png"))
    if show:
        plt.show()
    plt.close()

# %% [code] {"_kg_hide-output":true,"_kg_hide-input":true,"execution":{"iopub.status.busy":"2023-10-24T14:42:45.384819Z","iopub.execute_input":"2023-10-24T14:42:45.385125Z","iopub.status.idle":"2023-10-24T14:42:45.399947Z","shell.execute_reply.started":"2023-10-24T14:42:45.385088Z","shell.execute_reply":"2023-10-24T14:42:45.398949Z"}}
class ResidualBiGRU(nn.Module):
    def __init__(self, hidden_size, n_layers=1, bidir=True):
        super(ResidualBiGRU, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            n_layers,
            batch_first=True,
            bidirectional=bidir,
        )
        dir_factor = 2 if bidir else 1
        self.fc1 = nn.Linear(
            hidden_size * dir_factor, hidden_size * dir_factor * 2
        )
        self.ln1 = nn.LayerNorm(hidden_size * dir_factor * 2)
        self.fc2 = nn.Linear(hidden_size * dir_factor * 2, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x, h=None):
        res, new_h = self.gru(x, h)
        # res.shape = (batch_size, sequence_size, 2*hidden_size)

        res = self.fc1(res)
        res = self.ln1(res)
        res = nn.functional.relu(res)

        res = self.fc2(res)
        res = self.ln2(res)
        res = nn.functional.relu(res)

        # skip connection
        res = res + x

        return res, new_h

class MultiResidualBiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, n_layers, bidir=True):
        super(MultiResidualBiGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.n_layers = n_layers

        self.fc_in = nn.Linear(input_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.res_bigrus = nn.ModuleList(
            [
                ResidualBiGRU(hidden_size, n_layers=1, bidir=bidir)
                for _ in range(n_layers)
            ]
        )
        self.fc_out = nn.Linear(hidden_size, out_size)

    def forward(self, x, h=None):
        # if we are at the beginning of a sequence (no hidden state)
        if h is None:
            # (re)initialize the hidden state
            h = [None for _ in range(self.n_layers)]

        x = self.fc_in(x)
        x = self.ln(x)
        x = nn.functional.relu(x)

        new_h = []
        for i, res_bigru in enumerate(self.res_bigrus):
            x, new_hi = res_bigru(x, h[i])
            new_h.append(new_hi)

        x = self.fc_out(x)
#         x = F.normalize(x,dim=0)
        return x, new_h  # log probabilities + hidden states

# %% [markdown]
# # Define Dataset

# %% [code] {"execution":{"iopub.status.busy":"2023-10-24T14:42:45.401271Z","iopub.execute_input":"2023-10-24T14:42:45.401897Z","iopub.status.idle":"2023-10-24T14:42:53.146435Z","shell.execute_reply.started":"2023-10-24T14:42:45.401870Z","shell.execute_reply":"2023-10-24T14:42:53.145670Z"}}
SIGMA = 720 #average length of day is 24*60*12 = 17280 for comparison
SAMPLE_FREQ = 12 # 1 obs per minute
class SleepDataset(Dataset):
    def __init__(
        self,
        file
    ):
        self.targets,self.data,self.ids = joblib.load(file)
            
    def downsample_seq_generate_features(self,feat, downsample_factor = SAMPLE_FREQ):
        # downsample data and generate features
        if len(feat)%SAMPLE_FREQ==0:
            feat = np.concatenate([feat,np.zeros(SAMPLE_FREQ-((len(feat))%SAMPLE_FREQ))+feat[-1]])
        feat = np.reshape(feat, (-1,SAMPLE_FREQ))
        feat_mean = np.mean(feat,1)
        feat_std = np.std(feat,1)
        feat_median = np.median(feat,1)
        feat_max = np.max(feat,1)
        feat_min = np.min(feat,1)

        return np.dstack([feat_mean,feat_std,feat_median,feat_max,feat_min])[0]
    def downsample_seq(self,feat, downsample_factor = SAMPLE_FREQ):
        # downsample data
        if len(feat)%SAMPLE_FREQ==0:
            feat = np.concatenate([feat,np.zeros(SAMPLE_FREQ-((len(feat))%SAMPLE_FREQ))+feat[-1]])
        feat = np.reshape(feat, (-1,SAMPLE_FREQ))
        feat_mean = np.mean(feat,1)
        return feat_mean
    
    def gauss(self,n=SIGMA,sigma=SIGMA*0.15):
        # guassian distribution function
        r = range(-int(n/2),int(n/2)+1)
        return [exp(-float(x)**2/(2*sigma**2)) for x in r]
    
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        X = self.data[index][['anglez','enmo']]
        y = self.targets[index]
        
        # turn target inds into array
        target_guassian = np.zeros((len(X),2))
        for s,e in y:
            st1,st2 = max(0,s-SIGMA//2),s+SIGMA//2+1
            ed1,ed2 = e-SIGMA//2,min(len(X),e+SIGMA//2+1)
            target_guassian[st1:st2,0] = self.gauss()[st1-(s-SIGMA//2):]
            target_guassian[ed1:ed2,1] = self.gauss()[:SIGMA+1-((e+SIGMA//2+1)-ed2)]
            # gc.collect()
        y = target_guassian
        # gc.collect()
        X = np.concatenate([self.downsample_seq_generate_features(X.values[:,i],SAMPLE_FREQ) for i in range(X.shape[1])],-1)
        # gc.collect()
        y = np.dstack([self.downsample_seq(y[:,i],SAMPLE_FREQ) for i in range(y.shape[1])])[0]
        # gc.collect()
        y = normalize(torch.from_numpy(y))
        X = torch.from_numpy(X)
        return X, y
train_ds = SleepDataset('train_data.pkl')

# %% [markdown]
# # Train and Eval

# %% [code] {"execution":{"iopub.status.busy":"2023-10-24T14:42:53.147867Z","iopub.execute_input":"2023-10-24T14:42:53.148577Z","iopub.status.idle":"2023-10-24T14:42:58.108601Z","shell.execute_reply.started":"2023-10-24T14:42:53.148524Z","shell.execute_reply":"2023-10-24T14:42:58.107610Z"}}
train_size = int(TRAIN_PROP * len(train_ds))
valid_size = len(train_ds) - train_size
indices = torch.randperm(len(train_ds))
train_sampler = SubsetRandomSampler(indices[:train_size])
valid_sampler = SubsetRandomSampler(
    indices[train_size : train_size + valid_size]
)
steps = train_size*EPOCHS
warmup_steps = int(steps*WARMUP_PROP)
model = MultiResidualBiGRU(input_size=10,hidden_size=64,out_size=2,n_layers=5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay = 0)
scheduler = CosineLRScheduler(optimizer,t_initial= steps,warmup_t=warmup_steps, warmup_lr_init=1e-6,lr_min=2e-8,)
dt = time.time()
model_path = ''
history = {
    "train_loss": [],
    "valid_loss": [],
    "valid_mAP": [],
    "lr": [],
}
best_valid_loss = np.inf
criterion = torch.nn.MSELoss()

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2023-10-24T14:42:58.109931Z","iopub.execute_input":"2023-10-24T14:42:58.110696Z","iopub.status.idle":"2023-10-24T14:42:58.120637Z","shell.execute_reply.started":"2023-10-24T14:42:58.110660Z","shell.execute_reply":"2023-10-24T14:42:58.119591Z"}}
def evaluate(
    model: nn.Module, max_chunk_size: int, loader: DataLoader, device, criterion
):
    model.eval()
    valid_loss = 0.0
    y_true_full = torch.FloatTensor([]).half()
    y_pred_full = torch.FloatTensor([]).half()
    for X_batch, y_batch in tqdm(loader, desc="Eval", unit="batch"):
        # as some of the sequences we are dealing with are pretty long, we
        # use a chunk-based approach
        y_batch = y_batch.to(device, non_blocking=True)
        pred = torch.zeros(y_batch.shape).to(device, non_blocking=True).half()

        # (re)initialize model's hidden state(s)
        h = None

        # number of chunks for this sequence (we assume batch size = 1)
        seq_len = X_batch.shape[1]
        for i in range(0, seq_len, max_chunk_size):
            X_chunk = X_batch[:, i : i + max_chunk_size].float().to(device, non_blocking=True)

            y_pred, h = model(X_chunk, h)
            h = [hi.detach() for hi in h]
            pred[:, i : i + max_chunk_size] = y_pred.half()
            del X_chunk
            # gc.collect()
        loss = criterion(
            pred.float(),
            y_batch.float(),
        )
        valid_loss += loss.item()
        del pred,loss
        # gc.collect()

    valid_loss /= len(loader)

    y_true_full = y_true_full.squeeze(0)
    y_pred_full = y_pred_full.squeeze(0)
    # gc.collect()
    return valid_loss

# %% [code] {"execution":{"iopub.status.busy":"2023-10-24T14:42:58.122256Z","iopub.execute_input":"2023-10-24T14:42:58.122607Z"}}
train_loader = DataLoader(
    train_ds,
    batch_size=BS,
    sampler=train_sampler,
    pin_memory=True,
    num_workers=0,
)
valid_loader = DataLoader(
    train_ds,
    batch_size=1,
    sampler=valid_sampler,
    pin_memory=True,
    num_workers=0,
)
for epoch in range(1, EPOCHS + 1):
    train_loss = 0.0
    n_tot_chunks = 0
    model.train()
    pbar = tqdm(
        train_loader, desc="Training", unit="batch"
    )
    for step,(X_batch, y_batch) in enumerate(pbar):
        y_batch = y_batch.to(device, non_blocking=True)
        pred = torch.zeros(y_batch.shape).to(device, non_blocking=True)
        optimizer.zero_grad()
        scheduler.step(step+train_size*epoch)
        h = None

        seq_len = X_batch.shape[1]
        for i in range(0, seq_len, max_chunk_size):
            X_chunk = X_batch[:, i : i + max_chunk_size].float()

            X_chunk = X_chunk.to(device, non_blocking=True)

            y_pred, h = model(X_chunk, h)
            h = [hi.detach() for hi in h]
            pred[:, i : i + max_chunk_size] = y_pred
            del X_chunk,y_pred

        loss = criterion(
            normalize(pred).float(),
            y_batch.float(),
        )
        loss.backward()
        train_loss += loss.item()
        n_tot_chunks+=1
        pbar.set_description(f'Training: loss = {(train_loss/n_tot_chunks):.2f}')

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e-1)
        optimizer.step()
        del pred,loss,y_batch,X_batch,h
        # gc.collect()
    train_loss /= len(train_loader)
    del pbar
    # gc.collect()
    # ctypes.CDLL("libc.so.6").malloc_trim(0)

    if epoch % 1 == 0:
        valid_loss = evaluate(
            model, max_chunk_size, valid_loader, device, criterion
        )

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(
                model.state_dict(),
                os.path.join(model_path, f"model_best.pth"),
            )

        dt = time.time() - dt
        print(
            f"{epoch}/{EPOCHS} -- ",
            f"train_loss = {train_loss:.6f} -- ",
            f"valid_loss = {valid_loss:.6f} -- ",
            f"time = {dt:.6f}s",
        )
        dt = time.time()

plot_history(history, model_path=model_path)
history_path = os.path.join(model_path, "history.json")
with open(history_path, "w", encoding="utf-8") as f:
    json.dump(history, f, ensure_ascii=False, indent=4)
