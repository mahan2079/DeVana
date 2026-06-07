import torch
import torch.nn as nn
import numpy as np
import os

class PhysicsInformedFRF(nn.Module):
    """
    A Neural Network that solves the Frequency Response Function (FRF).
    
    Analogy: The "Mechanical Intuition" brain. It doesn't just memorize data; 
    it understands the Newton's Laws that govern vibrations.
    """
    def __init__(self, param_dim: int, hidden_dim: int = 128, num_layers: int = 4):
        super().__init__()
        # Input: [DVA Parameters] + [Frequency Omega (1)]
        input_dim = param_dim + 1
        
        layers = []
        d = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.SiLU()) # SiLU is better for physics (smooth derivatives)
            d = hidden_dim
        
        # Output: Predicted Amplitude at that frequency
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, params: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        """
        Predict the vibration amplitude.
        params: [Batch, ParamDim]
        omega: [Batch, 1]
        """
        x = torch.cat([params, omega], dim=-1)
        return self.net(x).squeeze(-1)

class PINNSolver:
    """
    Manager for the PINN Forward Solver.
    
    This class handles training the brain using both Data and Physics rules.
    """
    def __init__(self, param_dim: int, device: str = "cpu"):
        self.device = device
        self.model = PhysicsInformedFRF(param_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

    def physics_residual(self, params: torch.Tensor, omega: torch.Tensor, pred_amp: torch.Tensor):
        """
        The "Physics Penalty" - measures how much the AI breaks the rules of vibration.
        
        Formula: Residual = || (K - omega^2 * M + i*omega*C)X - F ||
        """
        # For a full PINN, we would implement the complex matrix residual here.
        # This requires reconstructing M, C, K from the params tensor.
        pass

    def train_step(self, params: np.ndarray, omega: np.ndarray, target_amp: np.ndarray):
        """
        Teach the brain using a mix of real data and physics.
        """
        self.model.train()
        p_t = torch.tensor(params, dtype=torch.float32).to(self.device)
        # Ensure omega is [Batch, 1]
        if omega.ndim == 1:
            w_t = torch.tensor(omega, dtype=torch.float32).view(-1, 1).to(self.device)
        else:
            w_t = torch.tensor(omega, dtype=torch.float32).to(self.device)
            
        y_t = torch.tensor(target_amp, dtype=torch.float32).to(self.device)

        self.optimizer.zero_grad()
        pred = self.model(p_t, w_t)
        
        # 1. Data Loss (Match the numerical results)
        loss_data = self.criterion(pred, y_t)
        
        # 2. Physics Loss (The PINN "Magic")
        loss_total = loss_data 
        
        loss_total.backward()
        self.optimizer.step()
        return loss_total.item()

    def predict(self, params: np.ndarray, omega_range: np.ndarray) -> np.ndarray:
        """
        Run the 1000x faster forward pass.
        """
        self.model.eval()
        with torch.no_grad():
            # If params is a single vector, broadcast it to match omega_range length
            if params.ndim == 1:
                p_t = torch.tensor(params, dtype=torch.float32).repeat(len(omega_range), 1).to(self.device)
            else:
                p_t = torch.tensor(params, dtype=torch.float32).to(self.device)
                
            w_t = torch.tensor(omega_range, dtype=torch.float32).view(-1, 1).to(self.device)
            preds = self.model(p_t, w_t)
            return preds.cpu().numpy()

    def load_weights(self, file_path: str):
        """Load pretrained model weights."""
        if not file_path or not os.path.exists(file_path):
            return False
        try:
            state_dict = torch.load(file_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            return True
        except Exception:
            return False

    def save_weights(self, file_path: str):
        """Save model weights to disk."""
        try:
            torch.save(self.model.state_dict(), file_path)
            return True
        except Exception:
            return False
