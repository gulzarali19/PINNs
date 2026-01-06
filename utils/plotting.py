import matplotlib.pyplot as plt
import numpy as np
import torch
import os

# Create a results directory to keep the sidebar clean
if not os.path.exists('results'):
    os.makedirs('results')

def plot_loss(loss_history):
    """Saves and displays the training loss curve."""
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history)
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Convergence')
    plt.grid(True)
    
    save_path = 'results/loss_curve.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Loss plot saved to {save_path}")
    plt.show()
    plt.close()

def plot_result_1d(model, x_range=[-1, 1], t_range=[0, 1], device="cpu", title="PINN_Prediction"):
    """Saves and displays heatmap for 1D + Time problems."""
    x = np.linspace(x_range[0], x_range[1], 100)
    t = np.linspace(t_range[0], t_range[1], 100)
    X, T = np.meshgrid(x, t)
    
    x_flat = torch.tensor(X.flatten()[:, None], dtype=torch.float32).to(device)
    t_flat = torch.tensor(T.flatten()[:, None], dtype=torch.float32).to(device)
    
    model.eval()
    with torch.no_grad():
        u_pred = model(x_flat, t_flat).cpu().numpy()
    
    U = u_pred.reshape(X.shape)

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(T, X, U, cmap='rainbow', shading='auto')
    plt.colorbar(label='u(x, t)')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title(title)
    
    filename = f"results/{title.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Result plot saved to {filename}")
    plt.show()
    plt.close()

def plot_result_2d(model, x_range=[0, 1], y_range=[0, 1], time_slices=[0.0, 0.5, 1.0], device="cpu"):
    """Saves and displays spatial snapshots for 2D + Time problems."""
    x = np.linspace(x_range[0], x_range[1], 50)
    y = np.linspace(y_range[0], y_range[1], 50)
    X, Y = np.meshgrid(x, y)
    
    fig, axes = plt.subplots(1, len(time_slices), figsize=(15, 5))
    if len(time_slices) == 1: axes = [axes]
    
    model.eval()
    for i, t_val in enumerate(time_slices):
        x_flat = torch.tensor(X.flatten()[:, None], dtype=torch.float32).to(device)
        y_flat = torch.tensor(Y.flatten()[:, None], dtype=torch.float32).to(device)
        t_flat = torch.ones_like(x_flat) * t_val
        
        with torch.no_grad():
            u_pred = model(x_flat, y_flat, t_flat).cpu().numpy()
        
        U = u_pred.reshape(X.shape)
        im = axes[i].pcolormesh(X, Y, U, cmap='hot', shading='auto')
        axes[i].set_title(f"t = {t_val}")
        fig.colorbar(im, ax=axes[i])
        
    plt.tight_layout()
    save_path = 'results/heat_2d_snapshots.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"2D snapshots saved to {save_path}")
    plt.show()
    plt.close()
