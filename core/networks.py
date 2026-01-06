import torch
import torch.nn as nn

class PINNNet(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.Tanh())
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, *inputs):
        # Concatenate inputs like (x, t) or (x, y, t)
        X = torch.cat(inputs, dim=1)
        return self.net(X)
