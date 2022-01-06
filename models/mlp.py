import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, hidden_layers, input_size, hidden_size, output_size, activation=nn.ReLU()):
        assert hidden_layers > 0, 'There has to be at least one hidden layer.'
        super().__init__()

        self._layers = nn.ModuleList()
        self._layers.append(nn.Linear(input_size, hidden_size))
        self._layers.append(activation)
        for _ in range(hidden_layers - 1):
            self._layers.append(nn.Linear(hidden_size, hidden_size))
            self._layers.append(activation)
        self._layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for l in self._layers:
            x = l(x)
        return x
