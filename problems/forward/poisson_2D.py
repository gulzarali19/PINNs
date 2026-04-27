"""
Poisson Equation PINN solver (2D).

The Poisson equation appears in electrostatics, heat conduction, etc.:
    -u_xx - u_yy = f(x, y)

where f is a source term.
"""

import torch
import numpy as np
from core.base_problem import ForwardProblem
from typing import Dict, Callable, Optional


class Poisson2DForward(ForwardProblem):
    """
    Forward 2D Poisson Equation Problem.
    
    Solves: -u_xx - u_yy = f(x, y)
    """
    
    def __init__(
        self,
        source: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        device: str = "cpu"
    ):
        """
        Initialize 2D Poisson problem.
        
        Args:
            source: Function f(x, y) defining the source term. Default: f = 0
            device: Computation device
        """
        super().__init__(device)
        self.source = source or (lambda x, y: torch.zeros_like(x))

    def pde_residual(self, model, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute Poisson residual.
        
        Args:
            model: Neural network model
            x: X-spatial coordinate
            y: Y-spatial coordinate
            
        Returns:
            Mean squared residual
        """
        x = x.detach().requires_grad_(True).to(self.device)
        y = y.detach().requires_grad_(True).to(self.device)
        
        u = model(x, y)
        
        # Spatial derivatives
        u_x = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True
        )[0]
        
        u_y = torch.autograd.grad(
            u, y, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        u_yy = torch.autograd.grad(
            u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True
        )[0]
        
        # Source term
        f = self.source(x, y)
        
        # Poisson residual: -u_xx - u_yy - f
        residual = -u_xx - u_yy - f
        
        return torch.mean(residual ** 2)

    def generate_data(self, n_samples: int = 1024) -> Dict[str, torch.Tensor]:
        """
        Generate training data for 2D Poisson equation.
        
        Note: Poisson is time-independent, so we use dummy time dimension.
        
        Args:
            n_samples: Number of collocation points
            
        Returns:
            Dictionary with BC and PDE training data
        """
        # Boundary conditions: u = 0 on all edges (Dirichlet)
        n_bc = 100
        bc_t = torch.zeros(n_bc * 4, 1)  # Dummy time coordinate
        
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
        
        # Collocation points for PDE
        f_x = torch.rand(n_samples, 1).requires_grad_(True)
        f_y = torch.rand(n_samples, 1).requires_grad_(True)
        f_t = torch.zeros(n_samples, 1)  # Dummy time
        
        # For consistency with the framework, we add dummy IC
        ic_x = torch.zeros(1, 1)
        ic_y = torch.zeros(1, 1)
        ic_t = torch.zeros(1, 1)
        ic_u = torch.zeros(1, 1)
        
        return {
            'ic_x': ic_x.to(self.device),
            'ic_y': ic_y.to(self.device),
            'ic_t': ic_t.to(self.device),
            'ic_u': ic_u.to(self.device),
            'bc_x': bc_x.to(self.device),
            'bc_y': bc_y.to(self.device),
            'bc_t': bc_t.to(self.device),
            'bc_u': bc_u.to(self.device),
            'f_x': f_x.to(self.device),
            'f_y': f_y.to(self.device),
            'f_t': f_t.to(self.device)
        }

    def analytical_solution(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Analytical solution for source-free case with Dirichlet BC.
        """
        return np.sin(np.pi * x) * np.sin(np.pi * y) / (2 * np.pi ** 2)
