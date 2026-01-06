import torch
import os

class PINNSolver:
    def __init__(self, model, pde_residual_fn, device="cpu"):
        self.model = model.to(device)
        self.pde_residual_fn = pde_residual_fn
        self.device = device
        
	params = list(self.model.parameters())
        if hasattr(self.problem, 'learnable_params'):
            params += self.problem.learnable_params
            print(f"Inverse Mode: Discovering parameters...")
	
	self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def train(self, data, epochs=5000):
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Extract data from dictionary (ic, bc, collocation)
            loss_ic = self.compute_data_loss(data['ic_x'], data['ic_t'], data['ic_u'])
            loss_bc = self.compute_data_loss(data['bc_x'], data['bc_t'], data['bc_u'])
            loss_pde = self.pde_residual_fn(self.model, data['f_x'], data['f_t'])

            total_loss = loss_ic + loss_bc + loss_pde
            total_loss.backward()
            self.optimizer.step()

            if epoch % 500 == 0:
                print(f"Epoch {epoch}: Loss {total_loss.item():.6f}")

    def compute_data_loss(self, x, t, u_true):
        u_pred = self.model(x, t)
        return torch.mean((u_pred - u_true)**2)
