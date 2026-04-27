"""
ADVANCED.md - Advanced Features and Best Practices

This document covers advanced usage patterns, optimization techniques, and
framework extension strategies for power users.
"""

# Advanced Features Guide

## Table of Contents
1. [Custom Physics Problems](#custom-physics-problems)
2. [Inverse Problem Solving](#inverse-problem-solving)
3. [Loss Function Weighting](#loss-function-weighting)
4. [Network Architecture Optimization](#network-architecture-optimization)
5. [Computational Efficiency](#computational-efficiency)
6. [Debugging and Diagnostics](#debugging-and-diagnostics)

---

## Custom Physics Problems

### Creating a New Problem Class

All physics problems should inherit from `BaseProblem` or one of its subclasses:

```python
from core.base_problem import ForwardProblem
import torch
import torch.nn as nn
import numpy as np
from typing import Dict

class MyAdvancedPDE(ForwardProblem):
    """
    Solver for: u_t + u_x^2 = epsilon * u_xx + source(x, t)
    
    A nonlinear advection-diffusion equation with source term.
    """
    
    def __init__(
        self,
        epsilon: float = 0.01,
        source_type: str = "gaussian",
        device: str = "cpu"
    ):
        super().__init__(device)
        self.epsilon = epsilon
        self.source_type = source_type
    
    def source_function(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Define source term."""
        if self.source_type == "gaussian":
            return torch.exp(-((x - 0.5) ** 2 + (t - 0.5) ** 2) / 0.1)
        elif self.source_type == "sinusoidal":
            return 0.1 * torch.sin(np.pi * x) * torch.cos(np.pi * t)
        else:
            return torch.zeros_like(x)
    
    def pde_residual(self, model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute: (u_t + u_x^2 - epsilon*u_xx - source)^2
        """
        x = x.detach().requires_grad_(True).to(self.device)
        t = t.detach().requires_grad_(True).to(self.device)
        
        u = model(x, t)
        
        # First derivatives
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        
        # Second derivative
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        
        # Source term
        S = self.source_function(x, t)
        
        # PDE residual
        f = u_t + u_x ** 2 - self.epsilon * u_xx - S
        
        return torch.mean(f ** 2)
    
    def generate_data(self, n_samples: int = 1024) -> Dict[str, torch.Tensor]:
        """Generate training data with proper normalization."""
        
        # Initial condition: smooth Gaussian
        n_ic = 150
        ic_x = torch.linspace(0, 1, n_ic).view(-1, 1)
        ic_t = torch.zeros(n_ic, 1)
        ic_u = torch.exp(-100 * (ic_x - 0.3) ** 2)
        
        # Boundary conditions
        n_bc = 100
        bc_t = torch.rand(n_bc * 2, 1)
        bc_x = torch.cat([
            torch.zeros(n_bc, 1),
            torch.ones(n_bc, 1)
        ])
        bc_u = torch.zeros(n_bc * 2, 1)
        
        # Collocation points
        f_x = torch.rand(n_samples, 1).requires_grad_(True)
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
```

### Key Implementation Guidelines

1. **Always enable gradients** for PDE points:
   ```python
   x = x.detach().requires_grad_(True)
   t = t.detach().requires_grad_(True)
   ```

2. **Use create_graph=True** for higher-order derivatives:
   ```python
   u_t = torch.autograd.grad(..., create_graph=True)[0]
   u_tt = torch.autograd.grad(u_t, ..., create_graph=True)[0]
   ```

3. **Return mean squared residual**:
   ```python
   return torch.mean(residual ** 2)
   ```

4. **Normalize data** for better convergence:
   ```python
   # In generate_data():
   ic_u = (ic_u - ic_u.mean()) / (ic_u.std() + 1e-8)
   ```

---

## Inverse Problem Solving

### Discovering Unknown Physics Parameters

Inverse problems learn PDE coefficients from data:

```python
from core.base_problem import InverseProblem
import torch.nn as nn

class BurgersInverse(InverseProblem):
    """Discover viscosity coefficient nu from solution data."""
    
    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        
        # Initialize learnable parameters
        # Start at a reasonable guess
        nu_param = nn.Parameter(torch.tensor([0.05], requires_grad=True, device=device))
        self.register_learnable_param('viscosity', nu_param)
    
    def pde_residual(self, model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Use discovered parameter in PDE."""
        x = x.detach().requires_grad_(True).to(self.device)
        t = t.detach().requires_grad_(True).to(self.device)
        
        u = model(x, t)
        
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        
        # Use discovered parameter
        nu = self._physics_params['viscosity']
        
        # Burgers equation: u_t + u*u_x - nu*u_xx = 0
        f = u_t + u * u_x - nu * u_xx
        
        return torch.mean(f ** 2)
    
    def generate_data(self, n_samples: int = 1024):
        # ... same as forward problem
        pass
```

### Training Inverse Problems

```python
from core.pinn_solver import PINNSolver

device = "cuda" if torch.cuda.is_available() else "cpu"

# Create inverse problem
problem = BurgersInverse(device=device)

# Network should be wider for inverse problems
model = PINNNet(
    input_dim=2,
    hidden_layers=[256, 256, 256, 256, 256],
    output_dim=1,
    activation='tanh'
)

# Solver automatically detects inverse mode
solver = PINNSolver(
    model, problem,
    device=device,
    learning_rate=1e-3,
    lr_scheduler='exponential',
    weight_decay=1e-5  # Small regularization helps
)

# Generate data (mix of IC, BC, and measurement data)
data = problem.generate_data(n_samples=2048)

# Train
history = solver.train(
    data,
    epochs=50000,
    verbose=True,
    verbose_interval=1000
)

# Check discovered parameters
print(f"Discovered parameters: {problem.get_physics_params()}")
# Output: {'viscosity': 0.009998...}
```

---

## Loss Function Weighting

### Dynamic Loss Weighting

Different problems require different weight balances:

```python
# Weight initial and boundary conditions heavily
loss_weights_ic_heavy = {
    'ic': 10.0,   # High weight for IC
    'bc': 10.0,   # High weight for BC
    'pde': 1.0    # Lower PDE weight
}

# Balance all losses equally (default)
loss_weights_balanced = {
    'ic': 1.0,
    'bc': 1.0,
    'pde': 1.0
}

# Weight PDE heavily (for problems with little data)
loss_weights_pde_heavy = {
    'ic': 1.0,
    'bc': 1.0,
    'pde': 100.0  # Very high PDE weight
}

# Train with custom weights
history = solver.train(
    data,
    epochs=10000,
    loss_weights=loss_weights_pde_heavy
)
```

### Adaptive Loss Weighting (Advanced)

```python
class AdaptivePINNSolver(PINNSolver):
    """Solver with adaptive loss weighting during training."""
    
    def train(self, data, epochs=5000, adaptive_weighting=True, **kwargs):
        """Train with adaptive loss weights."""
        
        loss_weights = {'ic': 1.0, 'bc': 1.0, 'pde': 1.0}
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Compute losses
            loss_ic = self.compute_data_loss(data['ic_x'], data['ic_t'], data['ic_u'])
            loss_bc = self.compute_data_loss(data['bc_x'], data['bc_t'], data['bc_u'])
            loss_pde = self.compute_pde_loss(data['f_x'], data['f_t'])
            
            # Adaptive weighting: balance loss magnitudes
            if adaptive_weighting and epoch % 100 == 0:
                with torch.no_grad():
                    avg_ic = loss_ic.item()
                    avg_bc = loss_bc.item()
                    avg_pde = loss_pde.item()
                    
                    # Normalize by median loss
                    losses = [avg_ic, avg_bc, avg_pde]
                    median_loss = np.median(losses)
                    
                    loss_weights['ic'] = median_loss / (avg_ic + 1e-8)
                    loss_weights['bc'] = median_loss / (avg_bc + 1e-8)
                    loss_weights['pde'] = median_loss / (avg_pde + 1e-8)
            
            # Weighted total loss
            total_loss = (
                loss_weights['ic'] * loss_ic +
                loss_weights['bc'] * loss_bc +
                loss_weights['pde'] * loss_pde
            )
            
            total_loss.backward()
            self.optimizer.step()
            
            if epoch % 500 == 0:
                print(f"Epoch {epoch} | Weights: IC={loss_weights['ic']:.2f}, "
                      f"BC={loss_weights['bc']:.2f}, PDE={loss_weights['pde']:.2f}")
```

---

## Network Architecture Optimization

### Activation Function Impact

```python
# Test different activations
activations = ['tanh', 'relu', 'elu', 'gelu', 'sigmoid']

for act in activations:
    model = PINNNet(
        input_dim=2,
        hidden_layers=[128, 128, 128],
        output_dim=1,
        activation=act
    )
    
    solver = PINNSolver(model, problem)
    history = solver.train(data, epochs=5000)
    
    final_loss = history['total'][-1]
    print(f"{act:10s}: Final Loss = {final_loss:.6e}")
```

Expected ranking (problem-dependent):
1. **tanh** - Best for PINNs in general (smooth, bounded)
2. **elu** - Good alternative, avoids dying ReLU
3. **gelu** - Smooth, works well for complex PDEs
4. **relu** - Fast but can be choppy
5. **sigmoid** - Usually too restrictive

### Depth vs Width Trade-off

```python
# Shallow & Wide (fewer gradient steps)
config_shallow = [256, 256, 256, 256]

# Deep & Narrow (more gradient steps)
config_deep = [64, 64, 64, 64, 64, 64, 64, 64]

# For 1D problems: prefer wide
# For 2D+ problems: prefer deep
# For inverse: extra-wide helps parameter discovery
```

### Batch Normalization

Generally **not recommended** for PINNs:
```python
# DON'T do this for PINNs (usually hurts convergence)
model = PINNNet(..., use_batch_norm=True)

# Better: let network normalize internally via tanh activation
model = PINNNet(..., use_batch_norm=False, activation='tanh')
```

---

## Computational Efficiency

### GPU Acceleration

```python
import torch

# Check GPU availability
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# Move computation to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Pre-allocate tensors on GPU for better performance
def generate_data_gpu(n_samples, device):
    """Generate data directly on GPU."""
    x = torch.rand(n_samples, 1, device=device, requires_grad=True)
    t = torch.rand(n_samples, 1, device=device, requires_grad=True)
    return x, t

# Measure GPU memory usage
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    # ... training code ...
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

### Batch Size Tuning

```python
# Memory-speed trade-off
# Larger batches = faster but more memory

# Rule of thumb:
# - Small GPU (2GB): batch_size = 512
# - Medium GPU (8GB): batch_size = 2048-4096
# - Large GPU (24GB): batch_size = 8192+

batch_sizes = [512, 1024, 2048, 4096]
for bs in batch_sizes:
    data = problem.generate_data(n_samples=bs)
    history = solver.train(data, epochs=10000)
    final_loss = history['total'][-1]
    print(f"Batch size {bs:5d}: Final loss = {final_loss:.6e}")
```

---

## Debugging and Diagnostics

### Loss Curve Analysis

```python
import matplotlib.pyplot as plt

def analyze_training_quality(history):
    """Diagnose training problems from loss curves."""
    
    total_loss = history['total']
    ic_loss = history['ic']
    bc_loss = history['bc']
    pde_loss = history['pde']
    
    # Check for divergence
    if total_loss[-1] > total_loss[0]:
        print("⚠️  DIVERGENCE: Loss increased overall")
    
    # Check for plateau
    if abs(total_loss[-1] - total_loss[-1000]) < 1e-6:
        print("⚠️  PLATEAU: Loss not improving")
    
    # Check for component dominance
    ic_ratio = ic_loss[-1] / total_loss[-1]
    bc_ratio = bc_loss[-1] / total_loss[-1]
    pde_ratio = pde_loss[-1] / total_loss[-1]
    
    print(f"Final loss composition:")
    print(f"  IC:  {ic_ratio*100:5.1f}%")
    print(f"  BC:  {bc_ratio*100:5.1f}%")
    print(f"  PDE: {pde_ratio*100:5.1f}%")
    
    if pde_ratio > 0.9:
        print("⚠️  PDE dominates: Increase IC/BC weights or improve collocation points")
    elif ic_ratio > 0.5:
        print("⚠️  IC dominates: Network struggle to satisfy PDE")
    
    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Log scale
    axes[0].semilogy(total_loss, label='Total', linewidth=2)
    axes[0].semilogy(ic_loss, label='IC', alpha=0.7)
    axes[0].semilogy(bc_loss, label='BC', alpha=0.7)
    axes[0].semilogy(pde_loss, label='PDE', alpha=0.7)
    axes[0].set_ylabel('Loss (log scale)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Linear scale
    axes[1].plot(total_loss, linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Total Loss')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Use after training
analyze_training_quality(history)
```

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Loss diverges | Learning rate too high | Reduce LR to 1e-4 |
| Loss plateaus | Network too small | Add more hidden layers |
| Oscillating loss | Unstable gradients | Use gradient clipping, weight decay |
| IC/BC loss high | Data constraint conflict | Check generate_data() |
| PDE loss high | Insufficient network capacity | Increase hidden_layers |
| Slow convergence | Wrong activation function | Try tanh instead of relu |

### Gradient Flow Analysis

```python
def check_gradient_flow(model, loss):
    """Diagnose gradient flow issues."""
    
    loss.backward(retain_graph=True)
    
    total_grad_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.norm().item() ** 2
    
    total_grad_norm = np.sqrt(total_grad_norm)
    
    if total_grad_norm > 1.0:
        print(f"⚠️  Large gradients: {total_grad_norm:.2e} (may cause instability)")
    elif total_grad_norm < 1e-6:
        print(f"⚠️  Tiny gradients: {total_grad_norm:.2e} (learning stalled)")
    else:
        print(f"✓ Gradient norm healthy: {total_grad_norm:.2e}")
```

---

## References & Further Reading

- [Physics-Informed Neural Networks (DeepXDE)](https://github.com/lululxvi/deepxde)
- [Original PINN Paper](https://arxiv.org/abs/1711.10566)
- [Inverse Problems with PINNs](https://arxiv.org/abs/1907.04026)
- [Variational Physics-Informed Neural Networks](https://arxiv.org/abs/1912.00873)

