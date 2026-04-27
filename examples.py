"""
Examples demonstrating PINN framework usage for various physics problems.

Run individual examples to understand the framework:
    python examples.py --example burgers_forward
    python examples.py --example heat_2d_forward
    python examples.py --example wave_1d_forward
    python examples.py --example schrodinger_1d_forward
    python examples.py --example poisson_2d
"""

import torch
import numpy as np
from pathlib import Path
from core.networks import PINNNet
from core.pinn_solver import PINNSolver


def example_burgers_forward():
    """Example: Solve 1D Burgers' equation."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Burgers' Equation (Forward Problem)")
    print("="*60)
    
    from problems.forward.burgers_1D import BurgersForward
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize problem
    problem = BurgersForward(nu=0.01 / np.pi, device=device)
    
    # Create network
    model = PINNNet(
        input_dim=2,
        hidden_layers=[128, 128, 128, 128],
        output_dim=1,
        activation='tanh'
    )
    
    # Create solver
    solver = PINNSolver(
        model, problem,
        device=device,
        learning_rate=1e-3,
        lr_scheduler='exponential'
    )
    
    # Generate data
    data = problem.generate_data(n_samples=1024)
    
    # Train
    print(f"\nTraining on {device}...")
    history = solver.train(
        data,
        epochs=10000,
        verbose=True,
        verbose_interval=1000
    )
    
    # Visualize
    from utils.plotting import plot_loss, plot_result_1d
    plot_loss(history)
    plot_result_1d(model, device=device, title="Burgers 1D Solution")
    
    print("\n✓ Burgers example complete!")
    return model, solver, problem


def example_heat_2d_forward():
    """Example: Solve 2D Heat equation."""
    print("\n" + "="*60)
    print("EXAMPLE 2: 2D Heat Equation (Forward Problem)")
    print("="*60)
    
    from problems.forward.heat_2D import Heat2DForward
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize problem
    problem = Heat2DForward(alpha=0.01, device=device)
    
    # Create network
    model = PINNNet(
        input_dim=3,  # x, y, t
        hidden_layers=[128, 128, 128, 128],
        output_dim=1,
        activation='tanh'
    )
    
    # Create solver
    solver = PINNSolver(
        model, problem,
        device=device,
        learning_rate=1e-3,
        lr_scheduler='exponential'
    )
    
    # Generate data
    data = problem.generate_data(n_samples=2048)
    
    # Train
    print(f"\nTraining on {device}...")
    history = solver.train(
        data,
        epochs=8000,
        verbose=True,
        verbose_interval=500
    )
    
    # Visualize
    from utils.plotting import plot_loss, plot_result_2d
    plot_loss(history)
    plot_result_2d(
        model,
        time_slices=[0.0, 0.25, 0.5, 0.75, 1.0],
        device=device
    )
    
    print("\n✓ Heat 2D example complete!")
    return model, solver, problem


def example_wave_1d_forward():
    """Example: Solve 1D Wave equation."""
    print("\n" + "="*60)
    print("EXAMPLE 3: 1D Wave Equation (Forward Problem)")
    print("="*60)
    
    from problems.forward.wave_1D import Wave1DForward
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize problem
    problem = Wave1DForward(c=1.0, device=device)
    
    # Create network
    model = PINNNet(
        input_dim=2,
        hidden_layers=[128, 128, 128, 128],
        output_dim=1,
        activation='tanh'
    )
    
    # Create solver
    solver = PINNSolver(
        model, problem,
        device=device,
        learning_rate=1e-3,
        lr_scheduler='exponential'
    )
    
    # Generate data
    data = problem.generate_data(n_samples=1024)
    
    # Train
    print(f"\nTraining on {device}...")
    history = solver.train(
        data,
        epochs=12000,
        verbose=True,
        verbose_interval=1000
    )
    
    # Visualize
    from utils.plotting import plot_loss, plot_result_1d
    plot_loss(history)
    plot_result_1d(model, x_range=[0, 1], device=device, title="Wave 1D Solution")
    
    print("\n✓ Wave 1D example complete!")
    return model, solver, problem


def example_schrodinger_1d_forward():
    """Example: Solve 1D Schrodinger equation."""
    print("\n" + "="*60)
    print("EXAMPLE 4: 1D Schrodinger Equation (Forward Problem)")
    print("="*60)
    
    from problems.forward.schrodinger import Schrodinger1DForward
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize problem
    problem = Schrodinger1DForward(device=device)
    
    # Create network (outputs 2 channels for real and imaginary parts)
    model = PINNNet(
        input_dim=2,
        hidden_layers=[128, 128, 128, 128],
        output_dim=2,
        activation='tanh'
    )
    
    # Create solver
    solver = PINNSolver(
        model, problem,
        device=device,
        learning_rate=1e-3,
        lr_scheduler='exponential',
        weight_decay=1e-4
    )
    
    # Generate data
    data = problem.generate_data(n_samples=1024)
    
    # Train
    print(f"\nTraining on {device}...")
    history = solver.train(
        data,
        epochs=10000,
        verbose=True,
        verbose_interval=1000
    )
    
    # Visualize
    from utils.plotting import plot_loss
    plot_loss(history)
    
    print("\n✓ Schrodinger 1D example complete!")
    print("Note: Complex-valued outputs require custom visualization")
    return model, solver, problem


def example_poisson_2d():
    """Example: Solve 2D Poisson equation."""
    print("\n" + "="*60)
    print("EXAMPLE 5: 2D Poisson Equation (Elliptic Problem)")
    print("="*60)
    
    from problems.forward.poisson_2D import Poisson2DForward
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize problem
    problem = Poisson2DForward(device=device)
    
    # Create network (time-independent, only x, y inputs)
    model = PINNNet(
        input_dim=2,  # x, y only
        hidden_layers=[256, 256, 256, 256],
        output_dim=1,
        activation='tanh'
    )
    
    # Create solver
    solver = PINNSolver(
        model, problem,
        device=device,
        learning_rate=1e-3,
        lr_scheduler='exponential'
    )
    
    # Generate data
    data = problem.generate_data(n_samples=2048)
    
    # Train
    print(f"\nTraining on {device}...")
    history = solver.train(
        data,
        epochs=8000,
        verbose=True,
        verbose_interval=500
    )
    
    # Visualize
    from utils.plotting import plot_loss
    plot_loss(history)
    
    print("\n✓ Poisson 2D example complete!")
    return model, solver, problem


def example_custom_problem():
    """Example: Create and solve a custom physics problem."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Custom Physics Problem")
    print("="*60)
    
    from core.base_problem import ForwardProblem
    
    class AdaptedWaveEquation(ForwardProblem):
        """Modified wave equation with damping: u_tt + 0.1*u_t = u_xx"""
        
        def __init__(self, device: str = "cpu"):
            super().__init__(device)
            self.c = 1.0
            self.damping = 0.1
        
        def pde_residual(self, model, x, t):
            x = x.detach().requires_grad_(True).to(self.device)
            t = t.detach().requires_grad_(True).to(self.device)
            
            u = model(x, t)
            
            u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            
            u_tt = torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
            
            # Damped wave: u_tt + damping*u_t = c^2*u_xx
            f = u_tt + self.damping * u_t - (self.c ** 2) * u_xx
            
            return torch.mean(f ** 2)
        
        def generate_data(self, n_samples: int = 1024):
            n_ic = 100
            ic_x = torch.linspace(0, 1, n_ic).view(-1, 1)
            ic_t = torch.zeros(n_ic, 1)
            ic_u = torch.sin(np.pi * ic_x)
            
            n_bc = 50
            bc_t = torch.rand(n_bc * 2, 1)
            bc_x = torch.cat([torch.zeros(n_bc, 1), torch.ones(n_bc, 1)])
            bc_u = torch.zeros(n_bc * 2, 1)
            
            f_x = torch.rand(n_samples, 1).requires_grad_(True)
            f_t = torch.rand(n_samples, 1).requires_grad_(True)
            
            return {
                'ic_x': ic_x, 'ic_t': ic_t, 'ic_u': ic_u,
                'bc_x': bc_x, 'bc_t': bc_t, 'bc_u': bc_u,
                'f_x': f_x, 'f_t': f_t
            }
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize custom problem
    problem = AdaptedWaveEquation(device=device)
    
    # Create network
    model = PINNNet(
        input_dim=2,
        hidden_layers=[128, 128, 128],
        output_dim=1,
        activation='tanh'
    )
    
    # Create solver
    solver = PINNSolver(model, problem, device=device, learning_rate=1e-3)
    
    # Generate data
    data = problem.generate_data(n_samples=1024)
    
    # Train
    print(f"\nTraining custom problem on {device}...")
    history = solver.train(data, epochs=5000, verbose=True, verbose_interval=500)
    
    from utils.plotting import plot_loss
    plot_loss(history)
    
    print("\n✓ Custom problem example complete!")
    return model, solver, problem


def main():
    """Run examples."""
    import argparse
    
    parser = argparse.ArgumentParser(description="PINN Examples")
    parser.add_argument(
        '--example',
        type=str,
        default='burgers_forward',
        choices=[
            'burgers_forward',
            'heat_2d_forward',
            'wave_1d_forward',
            'schrodinger_1d_forward',
            'poisson_2d',
            'custom_problem',
            'all'
        ],
        help='Which example to run'
    )
    
    args = parser.parse_args()
    
    examples = {
        'burgers_forward': example_burgers_forward,
        'heat_2d_forward': example_heat_2d_forward,
        'wave_1d_forward': example_wave_1d_forward,
        'schrodinger_1d_forward': example_schrodinger_1d_forward,
        'poisson_2d': example_poisson_2d,
        'custom_problem': example_custom_problem,
    }
    
    if args.example == 'all':
        for name, func in examples.items():
            try:
                func()
            except Exception as e:
                print(f"\n✗ {name} failed: {e}")
    else:
        examples[args.example]()


if __name__ == "__main__":
    main()
