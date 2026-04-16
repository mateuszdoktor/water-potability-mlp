import json

import torch
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from src.model import Net
from src.dataset import WaterDataset, create_stratified_splits, fit_scaler_on_train_split
from src.reproducibility import set_seed
import torch.nn as nn
from config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    DATA_PATH,
    METRICS_PATH,
    PROJECT_ROOT,
    REPORTS_DIR,
    SEED,
    TRAIN_SPLIT,
    VAL_SPLIT,
)

def evaluate_model():
    set_seed(SEED)

    if not BEST_MODEL_PATH.exists():
        print(f"Model checkpoint not found: {BEST_MODEL_PATH}")
        print("Run training first: python3 train.py")
        return

    dataset = WaterDataset(str(DATA_PATH))

    train_ds, _, test_ds = create_stratified_splits(
        dataset,
        train_ratio=TRAIN_SPLIT,
        val_ratio=VAL_SPLIT,
        seed=SEED,
    )

    checkpoint = torch.load(BEST_MODEL_PATH, map_location="cpu")

    if isinstance(checkpoint, dict) and "scaler_mean" in checkpoint and "scaler_scale" in checkpoint:
        scaler_mean = torch.tensor(checkpoint["scaler_mean"], dtype=torch.float32)
        scaler_scale = torch.tensor(checkpoint["scaler_scale"], dtype=torch.float32)
        scaler_scale = torch.where(scaler_scale == 0, torch.ones_like(scaler_scale), scaler_scale)
        dataset.features = (dataset.features - scaler_mean) / scaler_scale
    else:
        fit_scaler_on_train_split(dataset, train_ds.indices)

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    net = Net()
    model_state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    net.load_state_dict(model_state)
    net.eval()

    criterion = nn.BCEWithLogitsLoss()
    test_acc = Accuracy(task="binary")

    test_loss = 0.0

    with torch.no_grad():
        for features, label in test_loader:
            logits = net(features)
            probs = torch.sigmoid(logits)
            labels_view = label.view(-1, 1)
            
            loss = criterion(logits, labels_view)
            test_loss += loss.item()
            
            test_acc.update(probs, labels_view)

    mean_test_loss = test_loss / len(test_loader)
    test_accuracy = test_acc.compute()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        checkpoint_path_for_report = str(BEST_MODEL_PATH.relative_to(PROJECT_ROOT))
    except ValueError:
        checkpoint_path_for_report = str(BEST_MODEL_PATH)

    metrics = {
        "test_loss": round(float(mean_test_loss), 6),
        "test_accuracy": round(float(test_accuracy.item()), 6),
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "checkpoint": checkpoint_path_for_report,
    }
    if isinstance(checkpoint, dict):
        if "epoch" in checkpoint:
            metrics["checkpoint_epoch"] = int(checkpoint["epoch"])
        if "best_val_loss" in checkpoint:
            metrics["checkpoint_best_val_loss"] = round(float(checkpoint["best_val_loss"]), 6)

    with open(METRICS_PATH, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    print("=== Evaluation Results on Test Set ===")
    print(f"Test Loss: {mean_test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy.item():.4f}")
    print(f"Saved metrics: {METRICS_PATH}")

if __name__ == "__main__":
    evaluate_model()