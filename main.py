from core.networks import PINNNet
from core.pinn_solver import PINNSolver
from problems.burgers_1d import burgers_residual_1d
import torch

# Configuration (In a real scenario, load this from config/burgers.yaml)
layers = [30, 30]
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Initialize Network
model = PINNNet(input_dim=2, hidden_layers=layers)

# 2. Initialize Solver with specific physics
solver = PINNSolver(model, burgers_residual_1d, device=device)

# 3. Setup dummy training data (Replace with real IC/BC data)
data = {
    'ic_x': torch.rand(100, 1).to(device),
    'ic_t': torch.zeros(100, 1).to(device),
    'ic_u': torch.zeros(100, 1).to(device), # Target IC
    'bc_x': torch.ones(100, 1).to(device),
    'bc_t': torch.rand(100, 1).to(device),
    'bc_u': torch.zeros(100, 1).to(device),
    'f_x': torch.rand(2000, 1, requires_grad=True).to(device),
    'f_t': torch.rand(2000, 1, requires_grad=True).to(device)
}

# 4. Train
solver.train(data, epochs=1000)
