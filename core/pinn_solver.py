"""
PINN Solver Engine with training and optimization logic.

Handles the training loop, loss computation, and parameter updates
for physics-informed neural networks.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import warnings


class PINNSolver:
    """
    Solver for training Physics-Informed Neural Networks.
    
    Manages the training loop, loss computation, and optimization
    for both forward and inverse problems.
    """
    
    def __init__(
        self,
        model: nn.Module,
        problem,
        device: str = "cpu",
        learning_rate: float = 1e-3,
        lr_scheduler: Optional[str] = None,
        weight_decay: float = 0.0
    ):
        """
        Initialize the PINN solver.
        
        Args:
            model: Neural network model
            problem: Physics problem object (must have pde_residual method)
            device: Device for computation ('cpu' or 'cuda')
            learning_rate: Initial learning rate
            lr_scheduler: Optional scheduler type ('exponential', 'linear', 'cosine')
            weight_decay: L2 regularization parameter
        """
        self.model = model.to(device)
        self.problem = problem
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Collect all parameters (Model + Physics)
        params = list(self.model.parameters())
        if hasattr(self.problem, 'learnable_params'):
            params.extend(self.problem.learnable_params)
            if self.problem.learnable_params:
                print(f"[INVERSE MODE] Discovered {len(self.problem.learnable_params)} learnable physics parameter(s)")
        
        self.optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        
        # Learning rate scheduler
        self.scheduler = None
        if lr_scheduler == 'exponential':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9999)
        elif lr_scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5000)
        elif lr_scheduler == 'linear':
            self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, total_iters=5000)
        
        self.loss_history = []
        self.loss_components = {'ic': [], 'bc': [], 'pde': []}

    def compute_data_loss(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        u_true: torch.Tensor,
        loss_weight: float = 1.0
    ) -> torch.Tensor:
        """
        Compute MSE loss between predictions and data.
        
        Args:
            x: Spatial coordinates
            t: Temporal coordinates
            u_true: True solution values
            loss_weight: Weight for this loss component
            
        Returns:
            Weighted MSE loss
        """
        u_pred = self.model(x, t)
        return loss_weight * torch.mean((u_pred - u_true) ** 2)

    def compute_pde_loss(
        self,
        f_x: torch.Tensor,
        f_t: torch.Tensor,
        loss_weight: float = 1.0
    ) -> torch.Tensor:
        """
        Compute PDE residual loss.
        
        Args:
            f_x: Spatial coordinates for PDE points
            f_t: Temporal coordinates for PDE points
            loss_weight: Weight for this loss component
            
        Returns:
            Weighted PDE residual loss
        """
        return loss_weight * self.problem.pde_residual(self.model, f_x, f_t)

    def train(
        self,
        data: Dict[str, torch.Tensor],
        epochs: int = 5000,
        loss_weights: Optional[Dict[str, float]] = None,
        verbose: bool = True,
        verbose_interval: int = 500
    ) -> Dict[str, List[float]]:
        """
        Train the PINN model.
        
        Args:
            data: Dictionary with keys 'ic_x', 'ic_t', 'ic_u', 'bc_x', 'bc_t', 'bc_u', 'f_x', 'f_t'
            epochs: Number of training epochs
            loss_weights: Optional dictionary specifying weights for IC, BC, PDE losses
            verbose: Whether to print progress
            verbose_interval: Print progress every N epochs
            
        Returns:
            Dictionary with loss histories
        """
        self.loss_history = []
        self.loss_components = {'ic': [], 'bc': [], 'pde': []}
        
        # Default loss weights
        if loss_weights is None:
            loss_weights = {'ic': 1.0, 'bc': 1.0, 'pde': 1.0}
        
        self.model.train()
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Compute individual losses
            loss_ic = self.compute_data_loss(
                data['ic_x'], data['ic_t'], data['ic_u'],
                loss_weight=loss_weights.get('ic', 1.0)
            )
            loss_bc = self.compute_data_loss(
                data['bc_x'], data['bc_t'], data['bc_u'],
                loss_weight=loss_weights.get('bc', 1.0)
            )
            loss_pde = self.compute_pde_loss(
                data['f_x'], data['f_t'],
                loss_weight=loss_weights.get('pde', 1.0)
            )
            
            total_loss = loss_ic + loss_bc + loss_pde
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Record history
            self.loss_history.append(total_loss.item())
            self.loss_components['ic'].append(loss_ic.item())
            self.loss_components['bc'].append(loss_bc.item())
            self.loss_components['pde'].append(loss_pde.item())
            
            # Print progress
            if verbose and (epoch % verbose_interval == 0):
                param_status = ""
                if hasattr(self.problem, 'get_physics_params'):
                    params = self.problem.get_physics_params()
                    if params:
                        param_str = ", ".join([f"{k}={v:.6f}" for k, v in params.items()])
                        param_status = f" | {param_str}"
                
                print(
                    f"Epoch {epoch:6d}/{epochs} | "
                    f"Loss {total_loss.item():.6e} | "
                    f"IC {loss_ic.item():.6e} | "
                    f"BC {loss_bc.item():.6e} | "
                    f"PDE {loss_pde.item():.6e}{param_status}"
                )
        
        return {
            'total': self.loss_history,
            'ic': self.loss_components['ic'],
            'bc': self.loss_components['bc'],
            'pde': self.loss_components['pde']
        }

    def predict(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        return_numpy: bool = False
    ):
        """
        Make predictions on new data.
        
        Args:
            x: Spatial coordinates
            t: Temporal coordinates
            return_numpy: Whether to return numpy array
            
        Returns:
            Predictions (tensor or numpy array)
        """
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x, t)
        
        if return_numpy:
            return pred.cpu().numpy()
        return pred
