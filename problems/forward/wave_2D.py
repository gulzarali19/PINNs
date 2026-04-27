"""
2D Wave Equation PINN solver.

The 2D wave equation describes wave propagation in 2D:
    u_tt = c^2 * (u_xx + u_yy)

where c is the wave speed.
"""

import torch
import numpy as np
from core.base_problem import ForwardProblem
from typing import Dict


class Wave2DForward(ForwardProblem):
    """
    Forward 2D Wave Equation Problem.
    
    Solves: u_tt - c^2 * (u_xx + u_yy) = 0
    """
    
    def __init__(self, c: float = 1.0, device: str = "cpu"):
        """
        Initialize 2D Wave Equation problem.
        
        Args:
            c: Wave speed (default: 1.0)
            device: Computation device
        """
        super().__init__(device)
        self.c = c

    def pde_residual(self, model, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the 2D wave equation residual.
        
        u_tt - c^2 * (u_xx + u_yy) = 0
        
        Args:
            model: Neural network model
            x: X-spatial coordinate
            y: Y-spatial coordinate
            t: Temporal coordinate
            
        Returns:
            Mean squared residual
        """
        x = x.detach().requires_grad_(True).to(self.device)
        y = y.detach().requires_grad_(True).to(self.device)
        t = t.detach().requires_grad_(True).to(self.device)
        
        u = model(x, y, t)
        
        # Time derivatives
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]
        
        # Spatial derivatives
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        
        # PDE residual
        f = u_tt - (self.c ** 2) * (u_xx + u_yy)
        
        return torch.mean(f ** 2)

    def generate_data(self, n_samples: int = 1024) -> Dict[str, torch.Tensor]:
        """
        Generate training data for 2D wave equation.
        
        Args:
            n_samples: Number of collocation points
            
        Returns:
            Dictionary with IC, BC, and PDE training data
        """
        # Initial condition: u(x, y, 0) = sin(pi*x)*sin(pi*y)
        n_ic = 50
        x_ic = torch.rand(n_ic, 1)
        y_ic = torch.rand(n_ic, 1)
        t_ic = torch.zeros(n_ic, 1)
        ic_u = torch.sin(np.pi * x_ic) * torch.sin(np.pi * y_ic)
        
        # Boundary conditions: u = 0 on all edges
        n_bc = 50
        bc_t = torch.rand(n_bc * 4, 1)
        
        bc_x = torch.cat([
            torch.zeros(n_bc, 1),
            torch.ones(n_bc, 1),
            torch.rand(n_bc, 1),
            torch.rand(n_bc, 1)
        ])
        bc_y = torch.cat([
            torch.rand(n_bc, 1),
            torch.rand(n_bc, 1),
            torch.zeros(n_bc, 1),
            torch.ones(n_bc, 1)
        ])
        bc_u = torch.zeros(n_bc * 4, 1)
        
        # Collocation points
        f_x = torch.rand(n_samples, 1).requires_grad_(True)
        f_y = torch.rand(n_samples, 1).requires_grad_(True)
        f_t = torch.rand(n_samples, 1).requires_grad_(True)
        
        return {
            'ic_x': x_ic.to(self.device),
            'ic_y': y_ic.to(self.device),
            'ic_t': t_ic.to(self.device),
            'ic_u': ic_u.to(self.device),
            'bc_x': bc_x.to(self.device),
            'bc_y': bc_y.to(self.device),
            'bc_t': bc_t.to(self.device),
            'bc_u': bc_u.to(self.device),
            'f_x': f_x.to(self.device),
            'f_y': f_y.to(self.device),
            'f_t': f_t.to(self.device)
        }

    def analytical_solution(self, x: np.ndarray, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Analytical solution for separated initial condition.
        """
        return np.sin(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * np.sqrt(2) * self.c * t)
