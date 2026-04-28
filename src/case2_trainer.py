import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import copy
import time
import matplotlib.pyplot as plt

def build_fno_dataset(dataset):
    state = dataset["state"]          # (N, state_dim)
    u_hist = dataset["u_hist"]        # (N, delay_steps, control_dim)
    target = dataset["predictor"]     # (N, output_dim)

    N, delay_steps, _ = u_hist.shape

    state_rep = np.repeat(state[:, None, :], delay_steps, axis=1)
    X = np.concatenate([state_rep, u_hist], axis=2)

    return X, target

def fit_normalizers(X_train, Y_train, eps=1e-8):
    x_mean = X_train.mean(axis=(0, 1), keepdims=True)
    x_std = X_train.std(axis=(0, 1), keepdims=True) + eps

    y_mean = Y_train.mean(axis=0, keepdims=True)
    y_std = Y_train.std(axis=0, keepdims=True) + eps

    stats = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }
    return stats


def normalize_dataset(X, Y, stats):
    Xn = (X - stats["x_mean"]) / stats["x_std"]
    Yn = (Y - stats["y_mean"]) / stats["y_std"]
    return Xn, Yn


def denormalize_y(Yn, stats):
    return Yn * stats["y_std"] + stats["y_mean"]

def make_dataloaders(X, Y, batch_size=64, val_fraction=0.2, seed=0):
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=val_fraction, random_state=seed
    )

    stats = fit_normalizers(X_train, Y_train)

    X_train, Y_train = normalize_dataset(X_train, Y_train, stats)
    X_val, Y_val = normalize_dataset(X_val, Y_val, stats)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    Y_val = torch.tensor(Y_val, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(X_val, Y_val),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, stats

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.MSELoss()

    total_loss = 0.0
    total_count = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        pred = model(xb)
        loss = loss_fn(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = xb.shape[0]
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / total_count

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.MSELoss()

    total_loss = 0.0
    total_count = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        pred = model(xb)
        loss = loss_fn(pred, yb)

        batch_size = xb.shape[0]
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / total_count


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=100,
    lr=1e-3,
    weight_decay=1e-6,
    save_path="predictor_fno.pt",
):
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
    )

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

            torch.save(
                {
                    "model_state_dict": best_state,
                    "best_val_loss": best_val_loss,
                    "history": history,
                    "epoch": epoch,
                },
                save_path,
            )

        if epoch == 1 or epoch % 10 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"epoch {epoch:4d} | "
                f"train {train_loss:.6e} | "
                f"val {val_loss:.6e} | "
                f"lr {lr_now:.2e}"
            )
            # reload best weights safely
    clean_state = {k: v for k, v in best_state.items()}
    clean_state.pop("_metadata", None)

    model.load_state_dict(clean_state)
    return model, history

def load_trained_model(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict = checkpoint["model_state_dict"]
    state_dict.pop("_metadata", None)

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, checkpoint

def plot_training_history(history):
    plt.figure(figsize=(7, 4))
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title("Training history")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()