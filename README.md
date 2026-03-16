<div align="center">

# Water Potability Classification MLP

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)

*Binary classification of water potability using a Multilayer Perceptron.*

</div>

## About The Project

This repository serves as an applied machine learning project focused on **Binary Classification** for tabular data. The goal is to accurately predict whether water is potable based on various physicochemical features.

Beyond simply solving the classification problem, this project was designed to demonstrate a clean, reproducible pipeline in PyTorch, focusing on correct data scaling, deterministic evaluation, and training best practices for tabular datasets.

### Dataset
The project utilizes the **Water Potability** dataset. It contains physicochemical features like pH, hardness, solids, and chloramines to determine if the water is safe for human consumption. Rows with missing values are handled effectively during data loading to maintain data integrity.

### Key Highlights
* **Custom MLP Architecture:** Implemented a Multilayer Perceptron fully from scratch to build a solid foundation in neural network design for tabular data.
* **Rigorous Methodology:** Includes stratified train/validation/test splits (80/10/10) and feature scaling fitted exclusively on the training subset to prevent data leakage.
* **Deterministic Setup:** Configured with a fixed seed (`seed=42`) to guarantee reproducible experiments and checkpoints.
* **Comprehensive Evaluation:** Generates execution metrics, training curves, and evaluates using `BCEWithLogitsLoss`.

---

## Results & Benchmarks

The model was evaluated using the dedicated test split after selecting the best checkpoint based on validation loss.

| Architecture | Training Strategy | Accuracy | Test Loss | Parameter Scale | Observation |
|:---|:---|:---:|:---:|:---:|:---|
| **MLP** | From Scratch | **71.3%** | **0.6157** | Lightweight | Solid baseline for tabular physicochemical data. |

---

## Visualizing Model Behavior
Below is a snapshot of the training dynamics for our Multilayer Perceptron.

### Best Model Performance (MLP)

<details>
<summary><b>Click to see exactly how well the model learned (Accuracy & Loss Curves)</b></summary>
<br>

Visualizing the training phase confirms the convergence of the MLP and highlights where the best validation checkpoint was saved (epoch 69).

<div align="center">
  <img src="reports/figures/loss_curve.png" width="45%" alt="MLP Loss Curve"/>
  <img src="reports/figures/accuracy_curve.png" width="45%" alt="MLP Accuracy Curve"/>
</div>

</details>

*(Note: Full resolution plots are available in the `reports/figures/` directory)*

---

## Tech Stack

* **Deep Learning:** PyTorch, TorchMetrics
* **Data Processing:** NumPy, Pandas, Scikit-learn
* **Visualization:** Matplotlib, Seaborn

---

## Project Structure

```text
water-potability-mlp/
├── config.py           # Hyperparameters and path configurations
├── train.py            # Main entry point for training pipelines
├── evaluate.py         # Script to run inference and gather test metrics
├── data/               # Contains the dataset (water_potability.csv)
├── models/             # Saved model weights (.pth files)
├── reports/            # Generated metrics (.json) and evaluation plots
│   └── figures/
├── src/                # Model architectures and inner workings
│   ├── dataset.py      # Data loading and preprocessing
│   ├── model.py        # MLP architecture definition
│   ├── reproducibility.py # Functions to enforce determinism
│   └── utils.py        # Utility functions for plotting and saving
└── requirements.txt    # Project dependencies
```

---

## Getting Started

### Prerequisites
Make sure you have Python 3.10+ installed. It is highly recommended to run this project in a virtual environment.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/water-potability-mlp.git
   cd water-potability-mlp
   ```

2. Install the necessary dependencies:
   ```bash
   python3 -m pip install -r requirements.txt
   ```

3. Ensure your `data/` folder is correctly populated with the dataset:
   ```
   data/
       water_potability.csv
   ```

### Usage

**To train the model:**
Modify `config.py` to adjust hyperparameters (like learning rate, batch size, or epochs), then run:
```bash
python3 train.py
```
Outputs including the best checkpoint (`models/best_net.pth`) and training curves will be generated.

**To evaluate the model and generate metrics:**
```bash
python3 evaluate.py
```
This will output test metrics to the terminal and save `metrics.json` directly into the `reports/` folder.

