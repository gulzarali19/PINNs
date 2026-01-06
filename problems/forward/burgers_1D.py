import torch
import numpy as np

class BurgersForward:
    def __init__(self, nu=0.01/np.pi):
        self.nu = nu

    def pde_residual(self, model, x, t):
        # Ensure gradients are tracked
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        u = model(x, t)
        
        # Calculate derivatives
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        
        # Physics-informed residual
        f = u_t + u * u_x - self.nu * u_xx
        return torch.mean(f**2)
