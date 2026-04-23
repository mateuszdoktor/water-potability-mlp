import torch.nn as nn
import torch.nn.init as init


class Net(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.1) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer size")

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        linear_layers = [layer for layer in self.network if isinstance(layer, nn.Linear)]
        if not linear_layers:
            return

        for layer in linear_layers[:-1]:
            init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                init.zeros_(layer.bias)

        output_layer = linear_layers[-1]
        init.kaiming_uniform_(output_layer.weight, nonlinearity="linear")
        if output_layer.bias is not None:
            init.zeros_(output_layer.bias)

    def forward(self, x):
        return self.network(x)
