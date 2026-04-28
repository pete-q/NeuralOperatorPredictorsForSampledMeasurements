import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.case1_fno import PredictorFNO
from src.case1_trainer import build_fno_dataset, make_dataloaders, train_model, plot_training_history

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import copy

data_singlestep = np.load("dataset/singlestep_predictor_dataset_small.npz")
for key in data_singlestep.files:
    if key != "config":
        print(key, data_singlestep[key].shape)

X, Y = build_fno_dataset(data_singlestep)

train_loader, val_loader, stats = make_dataloaders(
    X, Y,
    batch_size=128,
    val_fraction=0.2,
    seed=0,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PredictorFNO(
    hidden_size=64,
    num_layers=4,
    modes=16,
    input_channel=X.shape[2],
    fno_output_channel=128,
    output_dim=Y.shape[1],
)

model, history = train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=500,
    lr=5e-4,
    save_path="models/singlestep_predictor_fno.pt",
)

plot_training_history(history)