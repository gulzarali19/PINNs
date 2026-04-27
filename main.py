"""
Main entry point for PINN framework.

Demonstrates training and visualizing Physics-Informed Neural Networks
for various PDE systems (forward and inverse problems).
"""

import torch
import yaml
import argparse
import sys
from pathlib import Path

from core.networks import PINNNet
from core.pinn_solver import PINNSolver


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def select_problem(cfg: dict, device: str):
    """
    Dynamically select and instantiate the physics problem.
    
    Args:
        cfg: Configuration dictionary
        device: Computation device
        
    Returns:
        Instantiated problem object and input dimension
    """
    problem_name = cfg['problem'].lower()
    
    # Forward problems
    if problem_name == "burgers_1d":
        from problems.forward.burgers_1D import BurgersForward
        problem = BurgersForward(
            nu=cfg['physics'].get('nu', 0.01 / 3.14159),
            device=device
        )
        input_dim = 2  # (x, t)
        
    elif problem_name == "heat_1d":
        from problems.forward.heat_2D import Heat2DForward
        problem = Heat2DForward(
            alpha=cfg['physics'].get('alpha', 0.01),
            device=device
        )
        input_dim = 2  # Dummy y-coordinate
        
    elif problem_name == "heat_2d":
        from problems.forward.heat_2D import Heat2DForward
        problem = Heat2DForward(
            alpha=cfg['physics'].get('alpha', 0.01),
            device=device
        )
        input_dim = 3  # (x, y, t)
        
    elif problem_name == "wave_1d":
        from problems.forward.wave_1D import Wave1DForward
        problem = Wave1DForward(
            c=cfg['physics'].get('c', 1.0),
            device=device
        )
        input_dim = 2  # (x, t)
        
    elif problem_name == "wave_2d":
        from problems.forward.wave_2D import Wave2DForward
        problem = Wave2DForward(
            c=cfg['physics'].get('c', 1.0),
            device=device
        )
        input_dim = 3  # (x, y, t)
        
    elif problem_name == "schrodinger_1d":
        from problems.forward.schrodinger import Schrodinger1DForward
        problem = Schrodinger1DForward(device=device)
        input_dim = 2  # (x, t)
        
    elif problem_name == "schrodinger_2d":
        from problems.forward.schrodinger import Schrodinger2DForward
        problem = Schrodinger2DForward(device=device)
        input_dim = 3  # (x, y, t)
        
    elif problem_name == "poisson_2d":
        from problems.forward.poisson_2D import Poisson2DForward
        problem = Poisson2DForward(device=device)
        input_dim = 2  # (x, y) - time-independent
        
    else:
        raise ValueError(f"Unknown problem: {problem_name}")
    
    return problem, input_dim


def main():
    """Main training pipeline."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Physics-Informed Neural Networks Suite')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    parser.add_argument('--device', type=str, default='auto', help='Device: cpu, cuda, or auto')
    args = parser.parse_args()
    
    # Load configuration
    print("[*] Loading configuration...")
    cfg = load_config(args.config)
    
    # Determine device
    if args.device == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"[✓] Using device: {device}")
    print(f"[✓] Problem: {cfg['problem']}")
    
    # Select problem and get input dimension
    problem, input_dim = select_problem(cfg, device)
    
    # Override config if needed
    if args.epochs:
        cfg['training']['epochs'] = args.epochs
    if args.lr:
        cfg['training']['learning_rate'] = args.lr
    
    # Initialize network
    print("[*] Initializing neural network...")
    model = PINNNet(
        input_dim=input_dim,
        hidden_layers=cfg['network']['hidden_layers'],
        output_dim=cfg['network'].get('output_dim', 1),
        activation=cfg['network'].get('activation', 'tanh'),
        use_batch_norm=cfg['network'].get('use_batch_norm', False)
    )
    
    print(f"[✓] Network initialized: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize solver
    print("[*] Initializing PINN solver...")
    solver = PINNSolver(
        model,
        problem,
        device=device,
        learning_rate=cfg['training'].get('learning_rate', 1e-3),
        lr_scheduler=cfg['training'].get('lr_scheduler', None),
        weight_decay=cfg['training'].get('weight_decay', 0.0)
    )
    
    # Generate training data
    print("[*] Generating training data...")
    if hasattr(problem, 'generate_data'):
        data = problem.generate_data(
            n_samples=cfg['training'].get('batch_size', 1024)
        )
    else:
        # Fallback: use default data (backward compatibility)
        data = {
            'ic_x': torch.linspace(-1, 1, 100).view(-1, 1).to(device),
            'ic_t': torch.zeros(100, 1).to(device),
            'ic_u': -torch.sin(3.14159 * torch.linspace(-1, 1, 100).view(-1, 1)).to(device),
            'bc_x': torch.tensor([[-1.0], [1.0]]).repeat(50, 1).to(device),
            'bc_t': torch.rand(100, 1).to(device),
            'bc_u': torch.zeros(100, 1).to(device),
            'f_x': (torch.rand(cfg['training']['batch_size'], 1) * 2 - 1).requires_grad_(True).to(device),
            'f_t': torch.rand(cfg['training']['batch_size'], 1).requires_grad_(True).to(device)
        }
    
    # Train model
    print(f"[*] Training for {cfg['training']['epochs']} epochs...")
    loss_history = solver.train(
        data,
        epochs=cfg['training']['epochs'],
        verbose=True,
        verbose_interval=cfg['training'].get('verbose_interval', 500)
    )
    
    print("[✓] Training complete!")
    
    # Visualization
    try:
        from utils.plotting import plot_loss, plot_result_1d, plot_result_2d
        
        print("[*] Generating visualizations...")
        plot_loss(loss_history)
        
        # Problem-specific visualization
        problem_name = cfg['problem'].lower()
        
        if '1d' in problem_name and 'poisson' not in problem_name:
            plot_result_1d(model, device=device, title=f"{cfg['problem']} Solution")
        elif '2d' in problem_name:
            time_slices = cfg['visualization'].get('time_slices', [0.0, 0.5, 1.0]) if 'visualization' in cfg else [0.0, 0.5, 1.0]
            plot_result_2d(model, device=device, time_slices=time_slices)
        
        print("[✓] Visualizations saved to results/")
        
    except Exception as e:
        print(f"[!] Visualization failed: {e}")
    
    return model, solver, problem


if __name__ == "__main__":
    main()
