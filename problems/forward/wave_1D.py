"""
1D Wave Equation PINN solver.

The wave equation describes wave propagation:
    u_tt = c^2 * u_xx

where c is the wave speed.
"""

import torch
import numpy as np
from core.base_problem import ForwardProblem
from typing import Dict


class Wave1DForward(ForwardProblem):
    """
    Forward 1D Wave Equation Problem.
    
    Solves: u_tt - c^2 * u_xx = 0
    """
    
    def __init__(self, c: float = 1.0, device: str = "cpu"):
        """
        Initialize 1D Wave Equation problem.
        
        Args:
            c: Wave speed (default: 1.0)
            device: Computation device
        """
        super().__init__(device)
        self.c = c

    def pde_residual(self, model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the 1D wave equation residual.
        
        u_tt - c^2 * u_xx = 0
        
        Args:
            model: Neural network model
            x: Spatial coordinate
            t: Temporal coordinate
            
        Returns:
            Mean squared residual
        """
        x = x.detach().requires_grad_(True).to(self.device)
        t = t.detach().requires_grad_(True).to(self.device)
        
        u = model(x, t)
        
        # First derivatives
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        
        # Second derivatives
        u_tt = torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        
        # PDE residual: u_tt - c^2 * u_xx
        f = u_tt - (self.c ** 2) * u_xx
        
        return torch.mean(f ** 2)

    def generate_data(self, n_samples: int = 1024) -> Dict[str, torch.Tensor]:
        """
        Generate training data for 1D wave equation.
        
        Args:
            n_samples: Number of collocation points
            
        Returns:
            Dictionary with IC, BC, and PDE training data
        """
        # Initial condition: u(x, 0) = sin(pi*x)
        n_ic = 100
        ic_x = torch.linspace(0, 1, n_ic).view(-1, 1)
        ic_t = torch.zeros(n_ic, 1)
        ic_u = torch.sin(np.pi * ic_x)
        
        # Initial velocity: u_t(x, 0) = 0
        n_ic_v = 100
        ic_vx = torch.linspace(0, 1, n_ic_v).view(-1, 1)
        ic_vt = torch.zeros(n_ic_v, 1)
        ic_vu = torch.zeros(n_ic_v, 1)
        
        # Boundary conditions: u(0, t) = u(1, t) = 0
        n_bc = 50
        bc_t = torch.rand(n_bc * 2, 1)
        bc_x = torch.cat([
            torch.zeros(n_bc, 1),
            torch.ones(n_bc, 1)
        ])
        bc_u = torch.zeros(n_bc * 2, 1)
        
        # Collocation points for PDE
        f_x = torch.rand(n_samples, 1).requires_grad_(True)
        f_t = torch.rand(n_samples, 1).requires_grad_(True)
        
        return {
            'ic_x': ic_x.to(self.device),
            'ic_t': ic_t.to(self.device),
            'ic_u': ic_u.to(self.device),
            'ic_vx': ic_vx.to(self.device),
            'ic_vt': ic_vt.to(self.device),
            'ic_vu': ic_vu.to(self.device),
            'bc_x': bc_x.to(self.device),
            'bc_t': bc_t.to(self.device),
            'bc_u': bc_u.to(self.device),
            'f_x': f_x.to(self.device),
            'f_t': f_t.to(self.device)
        }

    def analytical_solution(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Analytical solution for plucked string initial condition.
        """
        return np.sin(np.pi * x) * np.cos(np.pi * self.c * t)
