import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import torch
from src.model import Net
from src.dataset import WaterDataset, create_stratified_splits, fit_scaler_on_train_split
from src.reproducibility import set_seed
from src.utils import plot_training_curves
from config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    DATA_PATH,
    EPOCH_NUM,
    FIGURES_DIR,
    LEARNING_RATE,
    MODEL_DIR,
    SEED,
    SHOW_PLOTS,
    TRAIN_SPLIT,
    VAL_SPLIT,
    WEIGHT_DECAY,
    HIDDEN_DIMS,
    INPUT_DIM,
    DROPOUT
)

def train_model():
    set_seed(SEED)

    dataset = WaterDataset(str(DATA_PATH))

    train_ds, val_ds, _ = create_stratified_splits(
        dataset,
        train_ratio=TRAIN_SPLIT,
        val_ratio=VAL_SPLIT,
        seed=SEED,
    )

    scaler = fit_scaler_on_train_split(dataset, train_ds.indices)

    train_loader_generator = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=train_loader_generator,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
    )

    best_val_loss = float("inf")
    train_acc = Accuracy(task="binary")
    val_acc = Accuracy(task="binary")
    accuracies = []
    losses = []

    net = Net(
        input_dim=INPUT_DIM,
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT,
    )
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        net.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    for epoch in range(EPOCH_NUM):
        train_loss = 0.0
        val_loss = 0.0

        # Training loop
        net.train()
        for features, label in train_loader:
            optimizer.zero_grad()
            logits = net(features)
            probs = torch.sigmoid(logits)
            labels_view = label.view(-1, 1)
            train_acc.update(probs, labels_view)
            loss = criterion(logits, labels_view)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation loop
        net.eval()
        with torch.no_grad():
            for features, label in val_loader:
                logits = net(features)
                probs = torch.sigmoid(logits)
                labels_view = label.view(-1, 1)
                val_acc.update(probs, labels_view)
                loss = criterion(logits, labels_view)
                val_loss += loss.item()

        mean_train_loss = train_loss / len(train_loader)
        mean_val_loss = val_loss / len(val_loader)
        losses.append([mean_train_loss, mean_val_loss])

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
            }
            torch.save(checkpoint, BEST_MODEL_PATH)

        train_accuracy = train_acc.compute()
        train_acc.reset()
        val_accuracy = val_acc.compute()
        val_acc.reset()
        accuracies.append([train_accuracy.item(), val_accuracy.item()])

        print(
            f"train_loss={mean_train_loss:.4f}\t"
            f"val_loss={mean_val_loss:.4f}\t"
            f"train_accuracy={train_accuracy.item():.4f}\t"
            f"val_accuracy={val_accuracy.item():.4f}"
        )

    saved_figures = plot_training_curves(
        losses=losses,
        accuracies=accuracies,
        output_dir=FIGURES_DIR,
        show_plots=SHOW_PLOTS,
    )

    if saved_figures["loss_curve"] is not None and saved_figures["accuracy_curve"] is not None:
        print(f"Saved loss curve: {saved_figures['loss_curve']}")
        print(f"Saved accuracy curve: {saved_figures['accuracy_curve']}")


if __name__ == "__main__":
    train_model()
