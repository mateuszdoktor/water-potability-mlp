from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
import torch
import pandas as pd
import numpy as np

class WaterDataset(Dataset):
    def __init__(self,csv_path):
        super().__init__()
        df = pd.read_csv(csv_path)
        df = df.dropna()
        
        features = df.iloc[:, :-1].values
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(df.iloc[:, -1].values, dtype=torch.float32)

    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, index):
        return self.features[index], self.labels[index]


def fit_scaler_on_train_split(dataset, train_indices):
    scaler = StandardScaler()
    train_features = dataset.features[train_indices].numpy()
    scaler.fit(train_features)

    all_features = dataset.features.numpy()
    transformed = scaler.transform(all_features)
    dataset.features = torch.tensor(transformed, dtype=torch.float32)
    return scaler


def create_stratified_splits(dataset, train_ratio, val_ratio, seed):
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be > 0 and sum to < 1")

    labels = dataset.labels.numpy()
    indices = np.arange(len(dataset))

    train_indices, temp_indices = train_test_split(
        indices,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=labels,
    )

    val_ratio_in_temp = val_ratio / (1.0 - train_ratio)
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=(1.0 - val_ratio_in_temp),
        random_state=seed,
        stratify=labels[temp_indices],
    )

    train_ds = Subset(dataset, train_indices.tolist())
    val_ds = Subset(dataset, val_indices.tolist())
    test_ds = Subset(dataset, test_indices.tolist())
    return train_ds, val_ds, test_ds