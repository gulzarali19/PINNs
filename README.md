# PINNs: Physics-Informed Neural Networks Suite

A **modular, scalable, and extensible** framework for solving forward and inverse problems in physics using Physics-Informed Neural Networks (PINNs). Unlike standard deep learning approaches that rely solely on data, PINNs embed physical laws (PDEs) directly into the loss function, enabling accurate solutions with limited data.

![Status](https://img.shields.io/badge/status-active-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.8+-blue)

---

## 🎯 Features

-  **Multiple Physics Equations**: Burgers, Heat, Wave, Schrodinger, Poisson equations
-  **Flexible Architecture**: Customizable network depth, width, and activation functions
-  **Forward & Inverse Problems**: Discover unknown physics parameters during training
-  **Automatic Differentiation**: PyTorch autograd for exact PDE residuals
-  **Comprehensive Visualization**: Loss curves, spatial-temporal solutions, residuals
-  **Easy Configuration**: YAML-based hyperparameter management
-  **Production Ready**: Type hints, docstrings, error handling

---

## 📋 Supported Physics Equations

| Equation | Type | Dimensions | Config File |
|----------|------|-----------|-------------|
| **Burgers' Equation** | Nonlinear Advection-Diffusion | 1D | `config/burgers.yaml` |
| **Heat Equation** | Parabolic PDE | 1D/2D | `config/heat_2d.yaml` |
| **Wave Equation** | Hyperbolic PDE | 1D/2D | `config/wave_1d.yaml`, `wave_2d.yaml` |
| **Schrodinger Equation** | Quantum Mechanics | 1D/2D | `config/schrodinger_1d.yaml`, `schrodinger_2d.yaml` |
| **Poisson Equation** | Elliptic PDE | 2D | `config/poisson_2d.yaml` |

---

## 🏗️ Project Structure

```
PINNs/
├── core/
│   ├── base_problem.py       # Abstract base classes for problems
│   ├── networks.py           # Neural network architectures (MLP, DeepONet)
│   └── pinn_solver.py        # Training engine with loss computation
├── problems/
│   ├── forward/
│   │   ├── burgers_1D.py     # Burgers' equation implementation
│   │   ├── heat_2D.py        # 2D Heat equation
│   │   ├── wave_1D.py        # 1D Wave equation
│   │   ├── wave_2D.py        # 2D Wave equation
│   │   ├── schrodinger.py    # 1D/2D Schrodinger equation
│   │   └── poisson_2D.py     # 2D Poisson equation
│   └── inverse/              # Inverse problem solvers
├── utils/
│   └── plotting.py           # Comprehensive visualization tools
├── config/                   # YAML configuration files
│   ├── burgers.yaml
│   ├── heat_2d.yaml
│   ├── wave_1d.yaml
│   ├── wave_2d.yaml
│   ├── schrodinger_1d.yaml
│   ├── schrodinger_2d.yaml
│   └── poisson_2d.yaml
├── main.py                   # Entry point for training
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/gulzarali19/PINNs.git
cd PINNs

# Install dependencies
pip install -r requirements.txt
```

### Training a PINN

**Example 1: Burgers' Equation**
```bash
python main.py --config config/burgers.yaml
```

**Example 2: 2D Heat Equation**
```bash
python main.py --config config/heat_2d.yaml
```

**Example 3: Override Configuration**
```bash
python main.py --config config/wave_1d.yaml --epochs 10000 --lr 0.0005 --device cuda
```

### Output

Training produces:
- **Loss curves** (`results/loss_curve.png`): Total loss and component breakdown
- **Solution visualization** (`results/solution_*.png`): Spatial-temporal predictions
- **Residual analysis** (`results/residuals_*.png`): PDE residual distribution

---

## 📖 Theory & Mathematical Background

### What are PINNs?

Physics-Informed Neural Networks embed differential equations into the loss function:

$$\mathcal{L} = \lambda_{IC} \mathcal{L}_{IC} + \lambda_{BC} \mathcal{L}_{BC} + \lambda_{PDE} \mathcal{L}_{PDE}$$

where:
- $\mathcal{L}_{IC}$ = Initial condition loss
- $\mathcal{L}_{BC}$ = Boundary condition loss  
- $\mathcal{L}_{PDE}$ = PDE residual loss

### PDE Residuals

The network predicts $u(x,t)$, and we compute:
$$f(x,t) = u_t + u u_x - \nu u_{xx}$$

Then minimize $\frac{1}{N} \sum_{i=1}^N f(x_i, t_i)^2$

### Automatic Differentiation

PyTorch's `autograd` computes derivatives:
```python
u_t = torch.autograd.grad(u, t, create_graph=True)[0]
u_xx = torch.autograd.grad(u_x, x, create_graph=True)[0]  # Second derivative
```

---

## 🔧 Configuration Guide

### Basic Configuration (`config/*.yaml`)

```yaml
problem: "burgers_1d"              # Problem type
network:
  input_dim: 2                     # (x, t) dimensions
  hidden_layers: [128, 128, 128]  # Network architecture
  output_dim: 1                    # Output: scalar field
  activation: "tanh"               # Activation function
  use_batch_norm: false            # Batch normalization
training:
  epochs: 50000                    # Training iterations
  learning_rate: 0.001             # Initial learning rate
  lr_scheduler: "exponential"      # LR schedule
  batch_size: 1024                 # Collocation points
  weight_decay: 0.0                # L2 regularization
physics:
  nu: 0.003183                     # PDE parameter (viscosity)
```

### Supported Activation Functions
- `tanh` (default) - Best for PINNs
- `relu` - Faster convergence, less smooth
- `elu` - Smooth ReLU variant
- `gelu` - Gaussian Error Linear Unit
- `sigmoid` - Bounded output

### Learning Rate Schedulers
- `exponential` - Decay: $\gamma = 0.9999^{\text{epoch}}$
- `cosine` - Cosine annealing
- `linear` - Linear decay
- `None` - Constant learning rate

---

## 💡 Advanced Usage

### Custom Physics Problem

Create a new problem by inheriting from `BaseProblem`:

```python
from core.base_problem import ForwardProblem
import torch

class MyEquation(ForwardProblem):
    def __init__(self, param: float, device: str = "cpu"):
        super().__init__(device)
        self.param = param
    
    def pde_residual(self, model, x, t):
        """Implement your PDE residual."""
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        u = model(x, t)
        u_t = torch.autograd.grad(u, t, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, create_graph=True)[0]
        
        # Define PDE: u_t = something(u_x, u_xx, ...)
        f = u_t - u_xx - self.param * u
        return torch.mean(f**2)
    
    def generate_data(self, n_samples: int = 1024):
        """Generate IC, BC, and collocation data."""
        # Return dict with keys: ic_x, ic_t, ic_u, bc_x, bc_t, bc_u, f_x, f_t
        ...
```

### Inverse Problems

To discover unknown parameters:

```python
from core.base_problem import InverseProblem
import torch.nn as nn

class MyInverseProblem(InverseProblem):
    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        # Register learnable parameter
        param = nn.Parameter(torch.tensor([0.01], requires_grad=True))
        self.register_learnable_param('viscosity', param)
    
    def pde_residual(self, model, x, t):
        u = model(x, t)
        # Use self._physics_params['viscosity'] in your PDE
        ...
```

### Visualization Customization

```python
from utils.plotting import plot_result_1d, plot_comparison_1d

# Custom 1D visualization
plot_result_1d(
    model,
    x_range=[-5, 5],
    t_range=[0, 10],
    device='cuda',
    title='Custom Solution',
    n_points=200
)

# Compare against reference solution
plot_comparison_1d(
    model,
    reference_sol=reference_array,
    x=spatial_coords,
    t_idx=10
)
```

---

## 🔬 Physics Equation Details

### 1D Burgers' Equation
$$u_t + u u_x = \nu u_{xx}$$
- **Application**: Shock wave formation, turbulence modeling
- **Difficulty**: Nonlinearity, shock discontinuities
- **Config**: `config/burgers.yaml`

### 2D Heat Equation
$$u_t = \alpha (u_{xx} + u_{yy})$$
- **Application**: Heat diffusion, diffusion processes
- **Difficulty**: Multi-dimensional, source-dependent
- **Config**: `config/heat_2d.yaml`

### Wave Equations (1D/2D)
$$u_{tt} = c^2 \nabla^2 u$$
- **Application**: Sound/light propagation
- **Difficulty**: Second-order time derivative, oscillatory
- **Config**: `config/wave_1d.yaml`, `config/wave_2d.yaml`

### Schrodinger Equation (1D/2D)
$$i u_t = -\frac{1}{2} \nabla^2 u + V(x)u$$
- **Application**: Quantum mechanics, wave packet dynamics
- **Difficulty**: Complex-valued, potential-dependent
- **Config**: `config/schrodinger_1d.yaml`, `config/schrodinger_2d.yaml`

### Poisson Equation (2D)
$$-\nabla^2 u = f(x,y)$$
- **Application**: Electrostatics, gravitational fields
- **Difficulty**: Time-independent, source-dependent
- **Config**: `config/poisson_2d.yaml`

---

## 🎓 Training Tips

### For Better Convergence

1. **Increase network width**: More hidden units → better expressivity
   ```yaml
   hidden_layers: [256, 256, 256, 256]
   ```

2. **Use learning rate schedules**: Exponential decay often works well
   ```yaml
   lr_scheduler: "exponential"
   ```

3. **Increase collocation points**: More PDE constraint points
   ```yaml
   batch_size: 4096  # or higher
   ```

4. **Adjust loss weights**: Balance IC, BC, and PDE
   ```python
   loss_weights = {'ic': 1.0, 'bc': 1.0, 'pde': 10.0}
   solver.train(data, loss_weights=loss_weights)
   ```

5. **Try different activations**: `tanh` usually best, but `relu` sometimes faster
   ```yaml
   activation: "relu"
   ```

---

## 📚 References

- **Original PINN Paper**: [Physics-Informed Neural Networks](https://arxiv.org/abs/1711.10566)
- **Burgers' Equation**: [Raissi et al. 2017](https://arxiv.org/abs/1711.10566)
- **Wave Equations**: [Raissi et al. 2018](https://arxiv.org/abs/1711.10566)
- **Inverse Problems**: [Raissi et al. 2019](https://arxiv.org/abs/1907.04026)

---

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- [ ] More equation types (Navier-Stokes, Maxwell's equations)
- [ ] Attention mechanisms for improved accuracy
- [ ] GPU optimization and distributed training
- [ ] Uncertainty quantification
- [ ] More example problems and benchmarks

---


