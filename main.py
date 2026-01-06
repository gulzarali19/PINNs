import torch
import yaml
import argparse
from core.networks import PINNNet
from core.pinn_solver import PINNSolver

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # 1. Parse Command Line Arguments
    parser = argparse.ArgumentParser(description='PINN Physics Suite')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    args = parser.parse_args()

    # 2. Load Configuration
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 3. Dynamic Problem Selection
    if cfg['problem'] == "burgers_1d":
        from problems.forward.burgers_1D import BurgersForward
        problem = BurgersForward(nu=cfg['physics']['nu'])
        input_dim = cfg['network']['input_dim']
    elif cfg['problem'] == "heat_2d":
        from problems.forward.heat_2D import Heat2DForward
        problem = Heat2DForward(alpha=cfg['physics']['alpha'])
        input_dim = cfg['network']['input_dim']
    
    # 4. Initialize Network and Solver
    model = PINNNet(
        input_dim=input_dim, 
        hidden_layers=cfg['network']['hidden_layers'],
        output_dim=cfg['network']['output_dim']
    )
    
    solver = PINNSolver(model, problem, device=device)

    # 5. Data Generation (Using parameters from config)
    # Note: In a real project, replace this with a proper data loader
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

    # 6. Train
    print(f"Running PINN for: {cfg['problem']} on {device}")
    solver.train(data, epochs=cfg['training']['epochs'])

if __name__ == "__main__":
    main()
