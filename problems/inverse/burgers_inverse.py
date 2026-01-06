import torch
import torch.nn as nn
from core.base_problem import BaseProblem

class BurgersInverse(BaseProblem):
    def __init__(self, device):
        super().__init__(device)
        # Initialize nu as a learnable parameter (starting at 0.0)
        self.nu = nn.Parameter(torch.tensor([0.0], requires_grad=True).to(device))

    def pde_residual(self, model, x, t):
        u = model(x, t)
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        
        # The residual now uses the 'learned' self.nu
        f = u_t + u * u_x - self.nu * u_xx
        return torch.mean(f**2)
