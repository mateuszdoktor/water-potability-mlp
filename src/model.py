import torch.nn as nn
import torch.nn.functional as functional
import torch.nn.init as init

DATA_FEATURES = 9

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(DATA_FEATURES, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(p=0.20)
        self.fc3 = nn.Linear(32, 1)

        init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.fc3.weight, nonlinearity='linear')

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = functional.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = functional.relu(x)

        x = self.fc3(x)

        return x