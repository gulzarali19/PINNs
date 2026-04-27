"""
Base Problem Class for Physics-Informed Neural Networks.

Provides a standard interface for all physics problems to ensure consistency
across the framework and enable easy switching between different equations.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional


class BaseProblem(ABC):
    """
    Abstract base class for all physics problems in the PINN framework.
    
    Every physics problem should inherit from this class and implement:
    - pde_residual: Computes the PDE loss term
    - generate_data: Creates training data for the problem
    - get_physics_params: Returns current learnable physics parameters
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize the base problem.
        
        Args:
            device: Device to run computations on ('cpu' or 'cuda')
        """
        self.device = device
        self.learnable_params = []

    @abstractmethod
    def pde_residual(self, model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the PDE residual (physics loss).
        
        Args:
            model: Neural network model
            x: Spatial coordinate(s)
            t: Time coordinate
            
        Returns:
            Mean squared PDE residual
        """
        pass

    @abstractmethod
    def generate_data(self, n_samples: int = 1024) -> Dict[str, torch.Tensor]:
        """
        Generate training data for the problem.
        
        Args:
            n_samples: Number of collocation points
            
        Returns:
            Dictionary with keys: 'ic_x', 'ic_t', 'ic_u', 'bc_x', 'bc_t', 'bc_u', 'f_x', 'f_t'
        """
        pass

    def get_physics_params(self) -> Dict[str, float]:
        """
        Get current values of learnable physics parameters.
        
        Returns:
            Dictionary of parameter names and values
        """
        return {}

    def set_device(self, device: str):
        """Move all parameters to the specified device."""
        self.device = device
        for param in self.learnable_params:
            param.to(device)


class ForwardProblem(BaseProblem):
    """Base class for forward problems where physics parameters are known."""
    pass


class InverseProblem(BaseProblem):
    """Base class for inverse problems where physics parameters need to be learned."""
    
    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        self._physics_params = {}

    def register_learnable_param(self, name: str, param: nn.Parameter):
        """Register a learnable physics parameter."""
        self._physics_params[name] = param
        self.learnable_params.append(param)

    def get_physics_params(self) -> Dict[str, float]:
        """Return current values of learned parameters."""
        return {name: param.item() for name, param in self._physics_params.items()}
