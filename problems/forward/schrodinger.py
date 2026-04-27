"""
Schrodinger Equation PINN solver (1D and 2D).

The Schrodinger equation describes quantum mechanical systems:
    i * u_t = -0.5 * u_xx + V(x) * u  (1D)
    i * u_t = -0.5 * (u_xx + u_yy) + V(x,y) * u  (2D)

where V is the potential.
"""

import torch
import numpy as np
from core.base_problem import ForwardProblem
from typing import Dict, Callable, Optional


class Schrodinger1DForward(ForwardProblem):
    """
    Forward 1D Schrodinger Equation Problem.
    
    Solves: i * u_t + 0.5 * u_xx - V(x) * u = 0
    
    Note: u is complex-valued; real and imaginary parts are solved simultaneously.
    """
    
    def __init__(
        self,
        potential: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        device: str = "cpu"
    ):
        """
        Initialize 1D Schrodinger problem.
        
        Args:
            potential: Function V(x) that returns potential values. Default: V(x) = 0
            device: Computation device
        """
        super().__init__(device)
        self.potential = potential or (lambda x: torch.zeros_like(x))

    def pde_residual(self, model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute Schrodinger residual for real and imaginary parts.
        
        The model outputs [u_real, u_imag] concatenated.
        
        Args:
            model: Neural network (outputs [u_real, u_imag])
            x: Spatial coordinate
            t: Temporal coordinate
            
        Returns:
            Mean squared residual
        """
        x = x.detach().requires_grad_(True).to(self.device)
        t = t.detach().requires_grad_(True).to(self.device)
        
        u_complex = model(x, t)
        u_real = u_complex[..., :1]
        u_imag = u_complex[..., 1:]
        
        # Time derivatives
        u_real_t = torch.autograd.grad(
            u_real, t, grad_outputs=torch.ones_like(u_real), create_graph=True
        )[0]
        u_imag_t = torch.autograd.grad(
            u_imag, t, grad_outputs=torch.ones_like(u_imag), create_graph=True
        )[0]
        
        # Spatial derivatives
        u_real_x = torch.autograd.grad(
            u_real, x, grad_outputs=torch.ones_like(u_real), create_graph=True
        )[0]
        u_imag_x = torch.autograd.grad(
            u_imag, x, grad_outputs=torch.ones_like(u_imag), create_graph=True
        )[0]
        
        u_real_xx = torch.autograd.grad(
            u_real_x, x, grad_outputs=torch.ones_like(u_real_x), create_graph=True
        )[0]
        u_imag_xx = torch.autograd.grad(
            u_imag_x, x, grad_outputs=torch.ones_like(u_imag_x), create_graph=True
        )[0]
        
        # Potential
        V = self.potential(x)
        
        # Schrodinger equation: i*u_t = -0.5*u_xx + V*u
        # Real part: u_real_t = 0.5*u_imag_xx - V*u_imag
        # Imag part: u_imag_t = -0.5*u_real_xx + V*u_real
        
        f_real = u_real_t - (0.5 * u_imag_xx - V * u_imag)
        f_imag = u_imag_t - (-0.5 * u_real_xx + V * u_real)
        
        return torch.mean(f_real ** 2 + f_imag ** 2)

    def generate_data(self, n_samples: int = 1024) -> Dict[str, torch.Tensor]:
        """
        Generate training data for 1D Schrodinger equation.
        
        Args:
            n_samples: Number of collocation points
            
        Returns:
            Dictionary with IC, BC, and PDE training data
        """
        # Initial condition: Gaussian packet
        n_ic = 100
        ic_x = torch.linspace(-5, 5, n_ic).view(-1, 1)
        ic_t = torch.zeros(n_ic, 1)
        
        # Gaussian wave packet: exp(-(x-x0)^2) * exp(i*k*x)
        x0, k0 = 0.0, 2.0
        ic_u_real = torch.exp(-ic_x ** 2) * torch.cos(k0 * ic_x)
        ic_u_imag = torch.exp(-ic_x ** 2) * torch.sin(k0 * ic_x)
        
        # Boundary conditions: u -> 0 at boundaries
        n_bc = 50
        bc_t = torch.rand(n_bc * 2, 1)
        bc_x = torch.cat([
            torch.full((n_bc, 1), -5.0),
            torch.full((n_bc, 1), 5.0)
        ])
        bc_u_real = torch.zeros(n_bc * 2, 1)
        bc_u_imag = torch.zeros(n_bc * 2, 1)
        
        # Collocation points
        f_x = (torch.rand(n_samples, 1) * 10 - 5).requires_grad_(True)
        f_t = torch.rand(n_samples, 1).requires_grad_(True)
        
        return {
            'ic_x': ic_x.to(self.device),
            'ic_t': ic_t.to(self.device),
            'ic_u_real': ic_u_real.to(self.device),
            'ic_u_imag': ic_u_imag.to(self.device),
            'bc_x': bc_x.to(self.device),
            'bc_t': bc_t.to(self.device),
            'bc_u_real': bc_u_real.to(self.device),
            'bc_u_imag': bc_u_imag.to(self.device),
            'f_x': f_x.to(self.device),
            'f_t': f_t.to(self.device)
        }


class Schrodinger2DForward(ForwardProblem):
    """
    Forward 2D Schrodinger Equation Problem.
    
    Solves: i * u_t + 0.5 * (u_xx + u_yy) - V(x,y) * u = 0
    """
    
    def __init__(
        self,
        potential: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        device: str = "cpu"
    ):
        """
        Initialize 2D Schrodinger problem.
        
        Args:
            potential: Function V(x, y) that returns potential values. Default: V = 0
            device: Computation device
        """
        super().__init__(device)
        self.potential = potential or (lambda x, y: torch.zeros_like(x))

    def pde_residual(
        self, model, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Compute 2D Schrodinger residual."""
        x = x.detach().requires_grad_(True).to(self.device)
        y = y.detach().requires_grad_(True).to(self.device)
        t = t.detach().requires_grad_(True).to(self.device)
        
        u_complex = model(x, y, t)
        u_real = u_complex[..., :1]
        u_imag = u_complex[..., 1:]
        
        # Time derivatives
        u_real_t = torch.autograd.grad(
            u_real, t, grad_outputs=torch.ones_like(u_real), create_graph=True
        )[0]
        u_imag_t = torch.autograd.grad(
            u_imag, t, grad_outputs=torch.ones_like(u_imag), create_graph=True
        )[0]
        
        # Spatial derivatives for real part
        u_real_x = torch.autograd.grad(
            u_real, x, grad_outputs=torch.ones_like(u_real), create_graph=True
        )[0]
        u_real_xx = torch.autograd.grad(
            u_real_x, x, grad_outputs=torch.ones_like(u_real_x), create_graph=True
        )[0]
        
        u_real_y = torch.autograd.grad(
            u_real, y, grad_outputs=torch.ones_like(u_real), create_graph=True
        )[0]
        u_real_yy = torch.autograd.grad(
            u_real_y, y, grad_outputs=torch.ones_like(u_real_y), create_graph=True
        )[0]
        
        # Spatial derivatives for imag part
        u_imag_x = torch.autograd.grad(
            u_imag, x, grad_outputs=torch.ones_like(u_imag), create_graph=True
        )[0]
        u_imag_xx = torch.autograd.grad(
            u_imag_x, x, grad_outputs=torch.ones_like(u_imag_x), create_graph=True
        )[0]
        
        u_imag_y = torch.autograd.grad(
            u_imag, y, grad_outputs=torch.ones_like(u_imag), create_graph=True
        )[0]
        u_imag_yy = torch.autograd.grad(
            u_imag_y, y, grad_outputs=torch.ones_like(u_imag_y), create_graph=True
        )[0]
        
        V = self.potential(x, y)
        
        f_real = u_real_t - (0.5 * (u_imag_xx + u_imag_yy) - V * u_imag)
        f_imag = u_imag_t - (-0.5 * (u_real_xx + u_real_yy) + V * u_real)
        
        return torch.mean(f_real ** 2 + f_imag ** 2)

    def generate_data(self, n_samples: int = 1024) -> Dict[str, torch.Tensor]:
        """Generate training data for 2D Schrodinger equation."""
        n_ic = 50
        x_ic = torch.rand(n_ic, 1) * 2 - 1
        y_ic = torch.rand(n_ic, 1) * 2 - 1
        t_ic = torch.zeros(n_ic, 1)
        
        # 2D Gaussian wave packet
        r_sq = (x_ic ** 2 + y_ic ** 2)
        ic_u_real = torch.exp(-r_sq)
        ic_u_imag = torch.zeros_like(ic_u_real)
        
        # Boundary conditions
        n_bc = 50
        bc_t = torch.rand(n_bc * 4, 1)
        bc_x = torch.cat([
            torch.full((n_bc, 1), -1.0),
            torch.full((n_bc, 1), 1.0),
            torch.rand(n_bc, 1),
            torch.rand(n_bc, 1)
        ])
        bc_y = torch.cat([
            torch.rand(n_bc, 1),
            torch.rand(n_bc, 1),
            torch.full((n_bc, 1), -1.0),
            torch.full((n_bc, 1), 1.0)
        ])
        bc_u_real = torch.zeros(n_bc * 4, 1)
        bc_u_imag = torch.zeros(n_bc * 4, 1)
        
        # Collocation points
        f_x = (torch.rand(n_samples, 1) * 2 - 1).requires_grad_(True)
        f_y = (torch.rand(n_samples, 1) * 2 - 1).requires_grad_(True)
        f_t = torch.rand(n_samples, 1).requires_grad_(True)
        
        return {
            'ic_x': x_ic.to(self.device),
            'ic_y': y_ic.to(self.device),
            'ic_t': t_ic.to(self.device),
            'ic_u_real': ic_u_real.to(self.device),
            'ic_u_imag': ic_u_imag.to(self.device),
            'bc_x': bc_x.to(self.device),
            'bc_y': bc_y.to(self.device),
            'bc_t': bc_t.to(self.device),
            'bc_u_real': bc_u_real.to(self.device),
            'bc_u_imag': bc_u_imag.to(self.device),
            'f_x': f_x.to(self.device),
            'f_y': f_y.to(self.device),
            'f_t': f_t.to(self.device)
        }
