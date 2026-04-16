from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def plot_loss_curves(losses, output_path=None, show_plot=True):
    train_losses = [item[0] for item in losses]
    val_losses = [item[1] for item in losses]

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=train_losses, label="Train Loss")
    sns.lineplot(data=val_losses, label="Validation Loss")
    plt.title("Training and Validation Loss Over Epochs", fontsize=15)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss Value", fontsize=12)
    plt.legend()
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_accuracy_curves(accuracies, output_path=None, show_plot=True):
    train_accuracies = [item[0] for item in accuracies]
    val_accuracies = [item[1] for item in accuracies]

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=train_accuracies, label="Train Accuracy")
    sns.lineplot(data=val_accuracies, label="Validation Accuracy")
    plt.title("Training and Validation Accuracy Over Epochs", fontsize=15)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy Value", fontsize=12)
    plt.legend()
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_training_curves(losses, accuracies, output_dir=None, show_plots=True):
    output_paths = {"loss_curve": None, "accuracy_curve": None}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths["loss_curve"] = output_dir / "loss_curve.png"
        output_paths["accuracy_curve"] = output_dir / "accuracy_curve.png"

    plot_loss_curves(
        losses,
        output_path=output_paths["loss_curve"],
        show_plot=show_plots,
    )
    plot_accuracy_curves(
        accuracies,
        output_path=output_paths["accuracy_curve"],
        show_plot=show_plots,
    )

    return {
        key: (str(value) if value is not None else None)
        for key, value in output_paths.items()
    }