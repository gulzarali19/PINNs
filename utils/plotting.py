import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_loss(loss_history):
    """Plots the training loss curve."""
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history)
    plt.yscale('log')
    plt.xlabel('Epochs (x500)')
    plt.ylabel('Loss')
    plt.title('Training Convergence')
    plt.grid(True)
    plt.show()

def plot_result_1d(x_range, t_range, model, title="PINN Prediction"):
    """
    Generates a heatmap for 1D problems (1D Space + Time).
    Works for Burgers' 1D and Heat 1D.
    """
    x = np.linspace(x_range[0], x_range[1], 100)
    t = np.linspace(t_range[0], t_range[1], 100)
    X, T = np.meshgrid(x, t)
    
    # Prepare inputs for the model
    x_flat = torch.tensor(X.flatten()[:, None], dtype=torch.float32)
    t_flat = torch.tensor(T.flatten()[:, None], dtype=torch.float32)
    
    with torch.no_grad():
        u_pred = model(x_flat, t_flat).numpy()
    
    U = u_pred.reshape(X.shape)

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(T, X, U, cmap='rainbow')
    plt.colorbar(label='u(x, t)')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title(title)
    plt.show()

def plot_result_2d(x_range, y_range, time_slices, model):
    """
    Generates snapshots for 2D problems at specific time points.
    Works for Heat 2D.
    """
    x = np.linspace(x_range[0], x_range[1], 50)
    y = np.linspace(y_range[0], y_range[1], 50)
    X, Y = np.meshgrid(x, y)
    
    fig, axes = plt.subplots(1, len(time_slices), figsize=(15, 5))
    
    for i, t_val in enumerate(time_slices):
        t = torch.ones_like(torch.tensor(X.flatten()[:, None])) * t_val
        x_flat = torch.tensor(X.flatten()[:, None], dtype=torch.float32)
        y_flat = torch.tensor(Y.flatten()[:, None], dtype=torch.float32)
        
        with torch.no_grad():
            u_pred = model(x_flat, y_flat, t).numpy()
        
        U = u_pred.reshape(X.shape)
        im = axes[i].pcolormesh(X, Y, U, cmap='hot')
        axes[i].set_title(f"t = {t_val}")
        fig.colorbar(im, ax=axes[i])
        
    plt.tight_layout()
    plt.show()
