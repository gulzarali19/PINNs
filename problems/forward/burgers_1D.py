import torch
import numpy as np

def burgers_residual_1d(model, x, t, nu=0.01/np.pi):
    u = model(x, t)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    
    f = u_t + u * u_x - nu * u_xx
    return torch.mean(f**2)
