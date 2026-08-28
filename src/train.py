"""
Train the baseline SimpleCNN on the preprocessed Cats-vs-Dogs data and log
everything (params, metrics, loss curve, confusion matrix, model artifact)
to MLflow.

Usage:
    python src/train.py --epochs 5 --batch-size 16 --lr 1e-3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from model_utils import CLASS_NAMES, SimpleCNN

PROCESSED_DIR = Path("data/processed")
MODEL_OUT = Path("app/model.pt")


def load_split(name: str):
    data = np.load(PROCESSED_DIR / f"{name}.npz")
    X = torch.from_numpy(data["X"]).float()
    y = torch.from_numpy(data["y"]).long()
    return TensorDataset(X, y)


def plot_loss_curve(train_losses, val_losses, out_path):
    plt.figure()
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Loss curve")
    plt.savefig(out_path)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion matrix")
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.savefig(out_path)
    plt.close()
    return cm


def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = torch.nn.CrossEntropyLoss()
    all_preds, all_true = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            loss_sum += loss.item() * X.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += X.size(0)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_true.extend(y.cpu().numpy().tolist())
    return loss_sum / max(total, 1), correct / max(total, 1), all_true, all_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = load_split("train")
    val_ds = load_split("val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = SimpleCNN(num_classes=len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    mlflow.set_experiment("catsdogs-baseline-cnn")
    with mlflow.start_run():
        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "model": "SimpleCNN",
        })

        train_losses, val_losses = [], []
        for epoch in range(args.epochs):
            model.train()
            running_loss, n = 0.0, 0
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(X)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * X.size(0)
                n += X.size(0)
            train_loss = running_loss / max(n, 1)
            val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc},
                step=epoch,
            )
            print(f"epoch {epoch+1}/{args.epochs} train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        Path("artifacts").mkdir(exist_ok=True)
        loss_curve_path = Path("artifacts/loss_curve.png")
        cm_path = Path("artifacts/confusion_matrix.png")
        plot_loss_curve(train_losses, val_losses, loss_curve_path)
        plot_confusion_matrix(y_true, y_pred, cm_path)
        mlflow.log_artifact(str(loss_curve_path))
        mlflow.log_artifact(str(cm_path))

        MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODEL_OUT)
        mlflow.log_artifact(str(MODEL_OUT))
        mlflow.pytorch.log_model(model, "model")

        print(f"Model saved to {MODEL_OUT} and logged to MLflow.")


if __name__ == "__main__":
    main()
