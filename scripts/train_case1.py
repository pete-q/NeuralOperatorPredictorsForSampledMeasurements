import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.case2_fno import MultistepPredictorFNO
from src.case2_trainer import build_multistep_fno_dataset, make_multistep_dataloaders, train_multistep_model, plot_multistep_training_history 


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import copy
import time

data_multistep = np.load("dataset/multistep_predictor_dataset_small.npz")
for key in data_multistep.files:
    if key != "config":
        print(key, data_multistep[key].shape)
X, Y = build_multistep_fno_dataset(data_multistep)

train_loader, val_loader, stats = make_multistep_dataloaders(
    X, Y,
    batch_size=256,
    val_fraction=0.2,
    seed=0,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultistepPredictorFNO(
    hidden_size=128,
    num_layers=4,
    modes=32,
    input_channel=X.shape[2],
    fno_output_channel=128,
    output_dim=Y.shape[2],
    output_horizon=Y.shape[1],
)

model, history = train_multistep_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=250,
    lr=5e-4,
    save_path="models/multistep_predictor_fno.pt",
)

plot_multistep_training_history(history)
