try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Define dummy classes to prevent import errors in other files
    class nn:
        class Module: pass
from PyQt5.QtCore import QThread, pyqtSignal

class GearboxPINN(nn.Module):
    def __init__(self, P=3, layers=5, neurons=64, activation='tanh', use_fourier=False, omega_max=100.0, n_freqs=10, topology_mask=None):
        super().__init__()
        self.P = P
        self.use_fourier = use_fourier
        
        input_dim = 1
        if use_fourier:
            self.register_buffer('omega', torch.linspace(1, omega_max, n_freqs))
            input_dim = n_freqs * 2
        
        act = nn.Tanh() if activation == 'tanh' else nn.SiLU()
        
        layers_list = [nn.Linear(input_dim, neurons), act]
        for _ in range(layers - 1):
            layers_list += [nn.Linear(neurons, neurons), act]
        layers_list.append(nn.Linear(neurons, P))
        
        self.net = nn.Sequential(*layers_list)
        
        # Topology mask (PxP). 1 if connected, 0 otherwise.
        if topology_mask is None:
            self.topology_mask = torch.ones(P, P)
        else:
            self.topology_mask = torch.tensor(topology_mask, dtype=torch.float32)
            
        # Ensure mask is symmetric and upper triangular for connections
        self.topology_mask = torch.triu(self.topology_mask) + torch.triu(self.topology_mask, diagonal=1).T
        
        # Physical parameters (log-parameterized for positivity)
        self.log_m = nn.Parameter(torch.zeros(P)) # Diagonal mass
        # K_raw and C_raw represent the physical spring/damper constants
        # Diagonal elements: ground springs (k_i, c_i)
        # Off-diagonal elements: connection springs (k_ij, c_ij)
        self.K_raw = nn.Parameter(torch.randn(P, P) * 0.1) 
        self.C_raw = nn.Parameter(torch.randn(P, P) * 0.1) 

    def get_matrices(self):
        # M is diagonal
        M = torch.diag(torch.exp(self.log_m))
        
        # Force physical constants to be positive
        K_pos = torch.exp(self.K_raw) * self.topology_mask
        C_pos = torch.exp(self.C_raw) * self.topology_mask
        
        # Connections (off-diagonals)
        K_conn = torch.triu(K_pos, diagonal=1)
        K_conn_sym = K_conn + K_conn.T
        
        C_conn = torch.triu(C_pos, diagonal=1)
        C_conn_sym = C_conn + C_conn.T
        
        K = -K_conn_sym
        C = -C_conn_sym
        
        # Diagonals: K_ii = k_i(ground) + sum_j(k_ij)
        for i in range(self.P):
            K[i, i] = K_pos[i, i] + torch.sum(K_conn_sym[i, :])
            C[i, i] = C_pos[i, i] + torch.sum(C_conn_sym[i, :])
            
        return M, C, K

    def forward(self, t):
        if self.use_fourier:
            # Transform t [N, 1] -> [N, n_freqs*2]
            t_f = torch.cat([torch.sin(self.omega * t), torch.cos(self.omega * t)], dim=1)
            return self.net(t_f)
        return self.net(t)

class PINNWorker(QThread):
    progress = pyqtSignal(int, float, dict) # epoch, loss, current_params
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, t_data, x_data, v_data, a_data, P, layers=5, neurons=64, 
                 adam_epochs=5000, lbfgs_epochs=1000, lr=1e-3, 
                 lambda_f=1.0, lambda_data=1.0, lambda_ic=1.0, lambda_reg=0.1,
                 use_fourier=False, omega_max=100.0, n_freqs=10, 
                 topology_mask=None, warmup_epochs=1000):
        super().__init__()
        self.t_data = torch.tensor(t_data, dtype=torch.float32).view(-1, 1)
        self.x_data = torch.tensor(x_data, dtype=torch.float32)
        self.v_data = torch.tensor(v_data, dtype=torch.float32)
        self.a_data = torch.tensor(a_data, dtype=torch.float32)
        self.P = P
        self.layers = layers
        self.neurons = neurons
        self.adam_epochs = adam_epochs
        self.lbfgs_epochs = lbfgs_epochs
        self.lr = lr
        
        self.lambda_f = lambda_f
        self.lambda_data = lambda_data
        self.lambda_ic = lambda_ic
        self.lambda_reg = lambda_reg
        
        self.use_fourier = use_fourier
        self.omega_max = omega_max
        self.n_freqs = n_freqs
        self.topology_mask = topology_mask
        self.warmup_epochs = warmup_epochs
        
        self.abort = False

    def stop(self):
        self.abort = True

    def physics_loss(self, model, t_col):
        t_col = t_col.clone().detach().requires_grad_(True)
        x_hat = model(t_col)
        
        M, C, K = model.get_matrices()
        
        v_hat = torch.zeros_like(x_hat)
        for i in range(self.P):
            v_hat[:, i:i+1] = torch.autograd.grad(x_hat[:, i:i+1], t_col, 
                                                 grad_outputs=torch.ones_like(x_hat[:, i:i+1]),
                                                 create_graph=True, retain_graph=True)[0]
            
        a_hat = torch.zeros_like(v_hat)
        for i in range(self.P):
            a_hat[:, i:i+1] = torch.autograd.grad(v_hat[:, i:i+1], t_col,
                                                 grad_outputs=torch.ones_like(v_hat[:, i:i+1]),
                                                 create_graph=True, retain_graph=True)[0]
            
        residual = a_hat @ M + v_hat @ C.T + x_hat @ K.T
        return torch.mean(residual**2)

    def run(self):
        try:
            model = GearboxPINN(
                P=self.P, layers=self.layers, neurons=self.neurons, 
                use_fourier=self.use_fourier, omega_max=self.omega_max, 
                n_freqs=self.n_freqs, topology_mask=self.topology_mask
            )
            optimizer_adam = torch.optim.Adam(model.parameters(), lr=self.lr)

            # Normalization
            t_min, t_max = self.t_data.min(), self.t_data.max()
            t_norm = (self.t_data - t_min) / (t_max - t_min)

            x_mean, x_std = self.x_data.mean(dim=0), self.x_data.std(dim=0)
            x_norm = (self.x_data - x_mean) / (x_std + 1e-8)
            v_norm = self.v_data / (x_std + 1e-8)
            a_norm = self.a_data / (x_std + 1e-8)

            loss_val = 0.0

            # Phase 1: Adam
            for epoch in range(self.adam_epochs):
                if self.abort: break

                optimizer_adam.zero_grad()

                t_norm.requires_grad_(True)
                x_pred = model(t_norm)

                # Compute predicted v and a for data loss
                v_pred = torch.zeros_like(x_pred)
                for i in range(self.P):
                    v_pred[:, i:i+1] = torch.autograd.grad(x_pred[:, i:i+1], t_norm, 
                                                         grad_outputs=torch.ones_like(x_pred[:, i:i+1]),
                                                         create_graph=True)[0]

                a_pred = torch.zeros_like(v_pred)
                for i in range(self.P):
                    a_pred[:, i:i+1] = torch.autograd.grad(v_pred[:, i:i+1], t_norm,
                                                         grad_outputs=torch.ones_like(v_pred[:, i:i+1]),
                                                         create_graph=True)[0]

                # Data Loss: x, v, a
                loss_x = torch.mean((x_pred - x_norm)**2)
                loss_v = torch.mean((v_pred - v_norm)**2)
                loss_a = torch.mean((a_pred - a_norm)**2)
                loss_data = self.lambda_data * (loss_x + loss_v + loss_a)

                # Apply physics loss after warmup
                if epoch >= self.warmup_epochs:
                    loss_phys = self.physics_loss(model, t_norm)
                    loss = loss_data + self.lambda_f * loss_phys
                else:
                    loss = loss_data

                loss.backward()
                optimizer_adam.step()
                loss_val = loss.item()
                if epoch % 50 == 0:
                    M, C, K = model.get_matrices()
                    self.progress.emit(epoch, loss_val, {
                        "M": M.detach().numpy().tolist(),
                        "C": C.detach().numpy().tolist(),
                        "K": K.detach().numpy().tolist()
                    })

            # Phase 2: L-BFGS
            if not self.abort and self.lbfgs_epochs > 0:
                optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), 
                                                   max_iter=self.lbfgs_epochs,
                                                   history_size=50,
                                                   line_search_fn='strong_wolfe')

                def closure():
                    optimizer_lbfgs.zero_grad()
                    x_pred = model(t_norm)
                    
                    # Compute predicted v and a for data loss
                    v_pred = torch.zeros_like(x_pred)
                    for i in range(self.P):
                        v_pred[:, i:i+1] = torch.autograd.grad(x_pred[:, i:i+1], t_norm, 
                                                             grad_outputs=torch.ones_like(x_pred[:, i:i+1]),
                                                             create_graph=True)[0]
                    
                    a_pred = torch.zeros_like(v_pred)
                    for i in range(self.P):
                        a_pred[:, i:i+1] = torch.autograd.grad(v_pred[:, i:i+1], t_norm,
                                                             grad_outputs=torch.ones_like(v_pred[:, i:i+1]),
                                                             create_graph=True)[0]

                    loss_x = torch.mean((x_pred - x_norm)**2)
                    loss_v = torch.mean((v_pred - v_norm)**2)
                    loss_a = torch.mean((a_pred - a_norm)**2)
                    loss_data = self.lambda_data * (loss_x + loss_v + loss_a)
                    
                    loss_phys = self.physics_loss(model, t_norm)
                    loss = loss_data + self.lambda_f * loss_phys
                    loss.backward()
                    return loss

                optimizer_lbfgs.step(closure)

                loss_val = closure().item()
                M, C, K = model.get_matrices()
                self.progress.emit(self.adam_epochs + self.lbfgs_epochs, loss_val, {
                    "M": M.detach().numpy().tolist(),
                    "C": C.detach().numpy().tolist(),
                    "K": K.detach().numpy().tolist()
                })

            # Final denormalization
            M, C, K = model.get_matrices()
            T = (t_max - t_min).item()
            M_final = M.detach().numpy() * (T**2)
            C_final = C.detach().numpy() * T
            K_final = K.detach().numpy()

            self.finished.emit({
                "M": M_final.tolist(),
                "C": C_final.tolist(),
                "K": K_final.tolist(),
                "loss": loss_val
            })

        except Exception as e:
            self.error.emit(str(e))