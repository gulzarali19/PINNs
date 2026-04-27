"""
Neural Network architectures for Physics-Informed Neural Networks.

Provides flexible MLP and specialized architectures with various
activation functions and normalization options.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class PINNNet(nn.Module):
    """
    Flexible Multi-Layer Perceptron for PINN problems.
    
    Supports various activation functions and optional batch normalization.
    """
    
    def __init__(
        self, 
        input_dim: int,
        hidden_layers: List[int],
        output_dim: int = 1,
        activation: str = 'tanh',
        use_batch_norm: bool = False,
        final_activation: Optional[str] = None
    ):
        """
        Initialize PINN network.
        
        Args:
            input_dim: Input dimension (e.g., 2 for (x, t), 3 for (x, y, t))
            hidden_layers: List of hidden layer dimensions
            output_dim: Output dimension (default 1 for scalar field)
            activation: Activation function ('tanh', 'relu', 'sigmoid', 'gelu')
            use_batch_norm: Whether to use batch normalization
            final_activation: Optional activation after output layer
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_name = activation
        
        # Select activation function
        activation_dict = {
            'tanh': nn.Tanh,
            'relu': nn.ReLU,
            'elu': nn.ELU,
            'gelu': nn.GELU,
            'sigmoid': nn.Sigmoid,
        }
        
        if activation not in activation_dict:
            raise ValueError(f"Unsupported activation: {activation}. Choose from {list(activation_dict.keys())}")
        
        self.activation = activation_dict[activation]
        self.use_batch_norm = use_batch_norm
        
        # Build network
        layers = []
        curr_dim = input_dim
        
        for h_dim in hidden_layers:
            layers.append(nn.Linear(curr_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(self.activation())
            curr_dim = h_dim
        
        # Output layer
        layers.append(nn.Linear(curr_dim, output_dim))
        
        # Final activation (optional)
        if final_activation:
            if final_activation not in activation_dict:
                raise ValueError(f"Unsupported final activation: {final_activation}")
            layers.append(activation_dict[final_activation]())
        
        self.net = nn.Sequential(*layers)
        
        # Xavier initialization for better convergence
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, *inputs) -> torch.Tensor:
        """
        Forward pass: concatenates all inputs and passes through network.
        
        Args:
            *inputs: Variable number of input tensors (e.g., x, t or x, y, t)
            
        Returns:
            Network output (shape: [batch_size, output_dim])
        """
        # Concatenate inputs
        X = torch.cat(inputs, dim=1)
        return self.net(X)


class DeepONet(nn.Module):
    """
    DeepONet architecture for learning operators.
    
    Maps function spaces to function spaces using branch and trunk networks.
    """
    
    def __init__(
        self,
        input_dim: int,
        branch_layers: List[int],
        trunk_layers: List[int],
        output_dim: int = 1,
        activation: str = 'tanh'
    ):
        """
        Initialize DeepONet.
        
        Args:
            input_dim: Dimension of input functions
            branch_layers: Dimensions of branch network layers
            trunk_layers: Dimensions of trunk network layers
            output_dim: Output dimension
            activation: Activation function name
        """
        super().__init__()
        
        activation_dict = {
            'tanh': nn.Tanh,
            'relu': nn.ReLU,
            'elu': nn.ELU,
            'gelu': nn.GELU,
        }
        
        act = activation_dict[activation]
        
        # Branch network (processes input functions)
        branch_layers_list = [nn.Linear(input_dim, branch_layers[0]), act()]
        for i in range(len(branch_layers) - 1):
            branch_layers_list.append(nn.Linear(branch_layers[i], branch_layers[i + 1]))
            branch_layers_list.append(act())
        
        # Trunk network (processes spatial/temporal coordinates)
        trunk_layers_list = [nn.Linear(input_dim, trunk_layers[0]), act()]
        for i in range(len(trunk_layers) - 1):
            trunk_layers_list.append(nn.Linear(trunk_layers[i], trunk_layers[i + 1]))
            trunk_layers_list.append(act())
        
        self.branch = nn.Sequential(*branch_layers_list)
        self.trunk = nn.Sequential(*trunk_layers_list)
        
        # Bias term
        self.bias = nn.Linear(1, output_dim)
    
    def forward(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            u: Input function values
            y: Spatial/temporal coordinates
            
        Returns:
            Operator output
        """
        branch_out = self.branch(u)
        trunk_out = self.trunk(y)
        
        # Element-wise product
        out = torch.sum(branch_out * trunk_out, dim=1, keepdim=True)
        out = out + self.bias(torch.ones_like(out))
        
        return out
