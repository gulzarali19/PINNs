"""
Comprehensive visualization utilities for PINN solutions.

Provides plotting functions for 1D, 2D, and 3D+ solutions with support
for various field types (real, complex, vector fields).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import torch
import os
from typing import Optional, List, Tuple, Dict


# Create results directory
if not os.path.exists('results'):
    os.makedirs('results')


def plot_loss(loss_history: Dict[str, List[float]], log_scale: bool = True):
    """
    Plot training loss history with components.
    
    Args:
        loss_history: Dictionary with 'total', 'ic', 'bc', 'pde' loss lists
        log_scale: Whether to use logarithmic scale for y-axis
    """
    plt.figure(figsize=(12, 5))
    
    # Determine if loss_history is dict or list (backward compatibility)
    if isinstance(loss_history, dict):
        plt.subplot(1, 2, 1)
        plt.plot(loss_history.get('total', []), label='Total', linewidth=2)
        if log_scale:
            plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Total Training Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot individual components
        plt.subplot(1, 2, 2)
        plt.plot(loss_history.get('ic', []), label='IC Loss', alpha=0.8)
        plt.plot(loss_history.get('bc', []), label='BC Loss', alpha=0.8)
        plt.plot(loss_history.get('pde', []), label='PDE Loss', alpha=0.8)
        if log_scale:
            plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Components')
        plt.grid(True, alpha=0.3)
        plt.legend()
    else:
        # Backward compatibility
        plt.plot(loss_history, linewidth=2)
        if log_scale:
            plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Convergence')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = 'results/loss_curve.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[✓] Loss plot saved to {save_path}")
    plt.show()
    plt.close()


def plot_result_1d(
    model,
    x_range: Tuple[float, float] = [-1, 1],
    t_range: Tuple[float, float] = [0, 1],
    device: str = "cpu",
    title: str = "PINN_Prediction",
    n_points: int = 100,
    save: bool = True
):
    """
    Visualize 1D + Time solution as heatmap.
    
    Args:
        model: Trained PINN model
        x_range: Spatial domain [x_min, x_max]
        t_range: Temporal domain [t_min, t_max]
        device: Computation device
        title: Plot title
        n_points: Resolution in each dimension
        save: Whether to save the figure
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    t = np.linspace(t_range[0], t_range[1], n_points)
    X, T = np.meshgrid(x, t)
    
    x_flat = torch.tensor(X.flatten()[:, None], dtype=torch.float32).to(device)
    t_flat = torch.tensor(T.flatten()[:, None], dtype=torch.float32).to(device)
    
    model.eval()
    with torch.no_grad():
        u_pred = model(x_flat, t_flat).cpu().numpy()
    
    U = u_pred.reshape(X.shape)

    plt.figure(figsize=(10, 6))
    im = plt.pcolormesh(T, X, U, cmap='RdYlBu_r', shading='auto')
    cbar = plt.colorbar(im, label='u(x, t)')
    plt.xlabel('Time (t)', fontsize=12)
    plt.ylabel('Space (x)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save:
        filename = f"results/{title.lower().replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"[✓] Result plot saved to {filename}")
    
    plt.show()
    plt.close()


def plot_result_2d(
    model,
    x_range: Tuple[float, float] = [0, 1],
    y_range: Tuple[float, float] = [0, 1],
    time_slices: Optional[List[float]] = None,
    device: str = "cpu",
    n_points: int = 100,
    save: bool = True,
    cmap: str = 'hot'
):
    """
    Visualize 2D + Time solution as spatial snapshots.
    
    Args:
        model: Trained PINN model
        x_range: X-domain [x_min, x_max]
        y_range: Y-domain [y_min, y_max]
        time_slices: List of times to plot (default: [0, 0.5, 1.0])
        device: Computation device
        n_points: Resolution in each spatial dimension
        save: Whether to save the figure
        cmap: Colormap name
    """
    if time_slices is None:
        time_slices = [0.0, 0.5, 1.0]
    
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    X, Y = np.meshgrid(x, y)
    
    fig, axes = plt.subplots(1, len(time_slices), figsize=(5 * len(time_slices), 5))
    if len(time_slices) == 1:
        axes = [axes]
    
    model.eval()
    for i, t_val in enumerate(time_slices):
        x_flat = torch.tensor(X.flatten()[:, None], dtype=torch.float32).to(device)
        y_flat = torch.tensor(Y.flatten()[:, None], dtype=torch.float32).to(device)
        t_flat = torch.ones_like(x_flat) * t_val
        
        with torch.no_grad():
            u_pred = model(x_flat, y_flat, t_flat).cpu().numpy()
        
        U = u_pred.reshape(X.shape)
        im = axes[i].pcolormesh(X, Y, U, cmap=cmap, shading='auto')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
        axes[i].set_title(f't = {t_val:.2f}')
        axes[i].set_aspect('equal')
        plt.colorbar(im, ax=axes[i])
        
    plt.tight_layout()
    
    if save:
        save_path = 'results/solution_2d_snapshots.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[✓] 2D snapshots saved to {save_path}")
    
    plt.show()
    plt.close()


def plot_result_2d_real_imag(
    model,
    x_range: Tuple[float, float] = [-5, 5],
    y_range: Tuple[float, float] = [-5, 5],
    t_val: float = 0.0,
    device: str = "cpu",
    n_points: int = 100,
    save: bool = True
):
    """
    Visualize complex-valued 2D solution (real and imaginary parts).
    
    Args:
        model: Trained PINN model (outputs complex values)
        x_range: X-domain
        y_range: Y-domain
        t_val: Time to evaluate at
        device: Computation device
        n_points: Resolution
        save: Whether to save the figure
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    X, Y = np.meshgrid(x, y)
    
    x_flat = torch.tensor(X.flatten()[:, None], dtype=torch.float32).to(device)
    y_flat = torch.tensor(Y.flatten()[:, None], dtype=torch.float32).to(device)
    t_flat = torch.ones_like(x_flat) * t_val
    
    model.eval()
    with torch.no_grad():
        u_pred = model(x_flat, y_flat, t_flat).cpu().numpy()
    
    u_real = u_pred[:, 0].reshape(X.shape)
    u_imag = u_pred[:, 1].reshape(X.shape) if u_pred.shape[1] > 1 else np.zeros_like(u_real)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = axes[0].pcolormesh(X, Y, u_real, cmap='RdBu_r', shading='auto')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title(f'Real Part (t={t_val})')
    axes[0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].pcolormesh(X, Y, u_imag, cmap='RdBu_r', shading='auto')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_title(f'Imaginary Part (t={t_val})')
    axes[1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    
    if save:
        save_path = 'results/complex_field_2d.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[✓] Complex field plot saved to {save_path}")
    
    plt.show()
    plt.close()


def plot_comparison_1d(
    model,
    reference_sol: np.ndarray,
    x: np.ndarray,
    t_idx: int = 0,
    device: str = "cpu",
    title: str = "Solution Comparison"
):
    """
    Compare PINN solution against reference solution at a time slice.
    
    Args:
        model: Trained PINN
        reference_sol: Reference solution array
        x: Spatial coordinates
        t_idx: Time index for comparison
        device: Computation device
        title: Plot title
    """
    x_tensor = torch.tensor(x[:, None], dtype=torch.float32).to(device)
    t_tensor = torch.ones_like(x_tensor) * reference_sol.shape[0] / len(x) * t_idx
    
    model.eval()
    with torch.no_grad():
        u_pred = model(x_tensor, t_tensor).cpu().numpy().flatten()
    
    u_ref = reference_sol[t_idx, :]
    error = np.abs(u_pred - u_ref)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot comparison
    axes[0].plot(x, u_ref, 'b-', linewidth=2, label='Reference')
    axes[0].plot(x, u_pred, 'r--', linewidth=2, label='PINN')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('u')
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot error
    axes[1].semilogy(x, error, 'k-', linewidth=2)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('|Error|')
    axes[1].set_title(f'Absolute Error (L∞ = {np.max(error):.2e})')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = 'results/comparison_1d.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[✓] Comparison plot saved to {save_path}")
    plt.show()
    plt.close()


def plot_mesh_2d(
    x_range: Tuple[float, float] = [0, 1],
    y_range: Tuple[float, float] = [0, 1],
    n_x: int = 20,
    n_y: int = 20,
    title: str = "Computational Domain"
):
    """
    Visualize the computational mesh and collocation points.
    
    Args:
        x_range: X-domain
        y_range: Y-domain
        n_x: Number of points in x-direction
        n_y: Number of points in y-direction
        title: Plot title
    """
    x = np.linspace(x_range[0], x_range[1], n_x)
    y = np.linspace(y_range[0], y_range[1], n_y)
    X, Y = np.meshgrid(x, y)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(X, Y, c='blue', s=20, alpha=0.6, label='Collocation points')
    
    # Add boundary
    plt.plot([x_range[0], x_range[1], x_range[1], x_range[0], x_range[0]],
             [y_range[0], y_range[0], y_range[1], y_range[1], y_range[0]],
             'r-', linewidth=2, label='Boundary')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    save_path = 'results/mesh_2d.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[✓] Mesh plot saved to {save_path}")
    plt.show()
    plt.close()


def plot_residuals_1d(
    model,
    problem,
    x_range: Tuple[float, float] = [-1, 1],
    t_range: Tuple[float, float] = [0, 1],
    device: str = "cpu",
    n_points: int = 50
):
    """
    Visualize PDE residuals across domain.
    
    Args:
        model: Trained PINN
        problem: Physics problem object with pde_residual method
        x_range: Spatial domain
        t_range: Temporal domain
        device: Computation device
        n_points: Resolution
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    t = np.linspace(t_range[0], t_range[1], n_points)
    X, T = np.meshgrid(x, t)
    
    x_flat = torch.tensor(X.flatten()[:, None], dtype=torch.float32, requires_grad=True).to(device)
    t_flat = torch.tensor(T.flatten()[:, None], dtype=torch.float32, requires_grad=True).to(device)
    
    model.eval()
    with torch.no_grad():
        # Compute local residuals
        u = model(x_flat, t_flat)
        
        # Simple approximation: gradient-based residual
        u.sum().backward()
        residuals = x_flat.grad.abs().cpu().numpy()
    
    Residuals = residuals.reshape(X.shape)
    
    plt.figure(figsize=(10, 6))
    im = plt.pcolormesh(T, X, np.log10(Residuals + 1e-10), cmap='viridis', shading='auto')
    plt.colorbar(im, label='log₁₀(|Residual|)')
    plt.xlabel('Time (t)')
    plt.ylabel('Space (x)')
    plt.title('PDE Residuals Distribution')
    
    save_path = 'results/residuals_1d.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[✓] Residuals plot saved to {save_path}")
    plt.show()
    plt.close()
