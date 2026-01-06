import torch
import os

class PINNSolver:
    def __init__(self, model, problem, device="cpu"):
        self.model = model.to(device)
        self.problem = problem
        self.device = device
        
        # 1. Collect all parameters (Model weights + Inverse Physics constants)
        params = list(self.model.parameters())
        if hasattr(self.problem, 'learnable_params'):
            params += self.problem.learnable_params
            print(f"Inverse Mode Detected: Discovering physics parameters...")
        
        self.optimizer = torch.optim.Adam(params, lr=1e-3)

    def train(self, data, epochs=5000):
        # Initialize history list
        loss_history = []
        self.model.train()
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # 3. Calculate Losses
            loss_ic = self.compute_data_loss(data['ic_x'], data['ic_t'], data['ic_u'])
            loss_bc = self.compute_data_loss(data['bc_x'], data['bc_t'], data['bc_u'])
            loss_pde = self.problem.pde_residual(self.model, data['f_x'], data['f_t'])

            total_loss = loss_ic + loss_bc + loss_pde
            total_loss.backward()
            self.optimizer.step()

            # Record history
            loss_history.append(total_loss.item())

            if epoch % 500 == 0:
                param_status = ""
                if hasattr(self.problem, 'get_params'):
                    param_status = f" | Discovered: {self.problem.get_params()}"
                
                print(f"Epoch {epoch}: Loss {total_loss.item():.6f}{param_status}")
        
        # IMPORTANT: Return history for plotting.py
        return loss_history

    def compute_data_loss(self, x, t, u_true):
        u_pred = self.model(x, t)
        return torch.mean((u_pred - u_true)**2)
