import torch
import torch.nn as nn

class Heat2DInverse:
    def __init__(self, device):
        self.device = device
        # Initialize alpha as a learnable parameter
        self.alpha = nn.Parameter(torch.tensor([0.0], requires_grad=True).to(device))

    def pde_residual(self, model, x, y, t):
        u = model(x, y, t)
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
        
        f = u_t - self.alpha * (u_xx + u_yy)
        return torch.mean(f**2)
