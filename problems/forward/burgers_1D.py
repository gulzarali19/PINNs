"""
1D Burgers' Equation PINN solver.

The Burgers' equation is a fundamental PDE combining nonlinearity and diffusion:
    u_t + u * u_x = nu * u_xx

where nu is the viscosity coefficient.
"""

import torch
import numpy as np
from core.base_problem import ForwardProblem
from typing import Dict


class BurgersForward(ForwardProblem):
    """
    Forward Burgers' Equation Problem.
    
    Solves: u_t + u * u_x = nu * u_xx
    """
    
    def __init__(self, nu: float = 0.01 / np.pi, device: str = "cpu"):
        """
        Initialize Burgers problem.
        
        Args:
            nu: Viscosity coefficient (default: 0.01/pi)
            device: Computation device
        """
        super().__init__(device)
        self.nu = nu

    def pde_residual(self, model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the Burgers' equation residual.
        
        u_t + u * u_x - nu * u_xx = 0
        
        Args:
            model: Neural network model
            x: Spatial coordinates
            t: Temporal coordinates
            
        Returns:
            Mean squared residual
        """
        # Ensure gradients are tracked
        x = x.detach().requires_grad_(True).to(self.device)
        t = t.detach().requires_grad_(True).to(self.device)
        
        u = model(x, t)
        
        # First derivatives
        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]
        
        # Second derivative
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True
        )[0]
        
        # PDE residual: u_t + u * u_x - nu * u_xx
        f = u_t + u * u_x - self.nu * u_xx
        
        return torch.mean(f ** 2)

    def generate_data(self, n_samples: int = 1024) -> Dict[str, torch.Tensor]:
        """
        Generate training data for Burgers equation.
        
        Args:
            n_samples: Number of collocation points
            
        Returns:
            Dictionary with IC, BC, and PDE training data
        """
        # Initial condition: u(x, 0) = -sin(pi*x)
        n_ic = 100
        ic_x = torch.linspace(-1, 1, n_ic).view(-1, 1)
        ic_t = torch.zeros(n_ic, 1)
        ic_u = -torch.sin(np.pi * ic_x)
        
        # Boundary conditions: u(-1, t) = u(1, t) = 0
        n_bc = 50
        bc_t = torch.rand(n_bc * 2, 1)
        bc_x = torch.cat([
            torch.full((n_bc, 1), -1.0),
            torch.full((n_bc, 1), 1.0)
        ])
        bc_u = torch.zeros(n_bc * 2, 1)
        
        # Collocation points for PDE residual
        f_x = (torch.rand(n_samples, 1) * 2 - 1).requires_grad_(True)
        f_t = torch.rand(n_samples, 1).requires_grad_(True)
        
        return {
            'ic_x': ic_x.to(self.device),
            'ic_t': ic_t.to(self.device),
            'ic_u': ic_u.to(self.device),
            'bc_x': bc_x.to(self.device),
            'bc_t': bc_t.to(self.device),
            'bc_u': bc_u.to(self.device),
            'f_x': f_x.to(self.device),
            'f_t': f_t.to(self.device)
        }

    def analytical_solution(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Compute analytical solution (for verification).
        Note: Only available for specific initial conditions.
        """
        # This is a simplified version - exact solution requires more complex computation
        return -np.sin(np.pi * x) * np.exp(-np.pi ** 2 * self.nu * t)
