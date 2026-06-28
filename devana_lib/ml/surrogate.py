import numpy as np
from typing import List

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True

    class _SurrogateMLP(nn.Module):
        """
        A simple Multi-Layer Perceptron (Neural Network) for predicting fitness.
        Think of this as a mathematical "Brain" that learns patterns.
        """
        def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
            super().__init__()
            layers: List[nn.Module] = []
            d = input_dim
            # Create the layers of the brain
            for _ in range(max(0, num_layers)):
                layers.append(nn.Linear(d, hidden_dim))
                layers.append(nn.ReLU()) # Non-linear activation
                if dropout > 0.0:
                    layers.append(nn.Dropout(p=float(dropout)))
                d = hidden_dim
            # The final layer gives a single number (the fitness score)
            layers.append(nn.Linear(d, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x).squeeze(-1)

except ImportError:
    TORCH_AVAILABLE = False

class NeuralSurrogate:
    """
    A Neural Network Surrogate that replaces KNN for candidate screening.
    
    Analogy: The "Seasoned Expert" who predicts performance without doing the math.
    """
    def __init__(
        self,
        input_dim: int,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.1,
        epochs: int = 10,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: str = "cpu"
    ):
        self.input_dim = input_dim
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        
        self._device = device if TORCH_AVAILABLE else "cpu"
        self._model = None
        if TORCH_AVAILABLE:
            self._model = _SurrogateMLP(input_dim, hidden, layers, dropout).to(self._device)
            self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
            self._criterion = nn.MSELoss()

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the surrogate brain using real evaluated data.
        """
        if not TORCH_AVAILABLE or self._model is None or len(X) < 10:
            return

        self._model.train()
        X_t = torch.tensor(X, dtype=torch.float32).to(self._device)
        y_t = torch.tensor(y, dtype=torch.float32).to(self._device)
        
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        for _ in range(self.epochs):
            for xb, yb in loader:
                self._optimizer.zero_grad()
                pred = self._model(xb)
                loss = self._criterion(pred, yb)
                loss.backward()
                self._optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the fitness of new candidates instantly.
        """
        if not TORCH_AVAILABLE or self._model is None:
            return np.ones(len(X)) * 1e6 # Return bad score if no model
            
        self._model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self._device)
            preds = self._model(X_t)
            return preds.cpu().numpy()

    def get_fitness_gradient(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the gradient (slope) of the predicted fitness with respect to the input parameters.
        This is the core of the "Smart Mutation" (Physics-Guided Mutation).
        """
        if not TORCH_AVAILABLE or self._model is None:
            return np.zeros_like(X) # No gradient if no model
            
        self._model.eval()
        # Enable gradient tracking on the input tensor
        X_t = torch.tensor(X, dtype=torch.float32).to(self._device).requires_grad_(True)
        
        # Forward pass
        preds = self._model(X_t)
        
        # Calculate gradients for each item in the batch
        # Since preds is a batch of scalars, we can just sum them up and call backward once,
        # which will correctly route the gradients back to the respective rows in X_t.
        total_pred = preds.sum()
        total_pred.backward()
        
        # Return the computed gradient (the "Slope" of the fitness landscape)
        return X_t.grad.cpu().numpy()
