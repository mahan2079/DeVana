import time
import math
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset, random_split
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    from scipy.stats import qmc
    SCIPY_QMC_AVAILABLE = True
except Exception:
    SCIPY_QMC_AVAILABLE = False

try:
    from scipy.stats import norm as _norm
    SCIPY_NORM_AVAILABLE = True
except Exception:
    SCIPY_NORM_AVAILABLE = False


def _now_ms() -> float:
    return time.time() * 1000.0


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        d = input_dim
        for _ in range(max(0, num_layers)):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU())
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(p=float(dropout)))
            d = hidden_dim
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class NeuralSeeder:
    """
    Online-learning seeding via an ensemble of small MLPs.

    - Maintains dataset of evaluated (x, y). x provided in original scale; class normalizes to [0,1] over variable dims.
    - Trains a small ensemble each generation under a wall-clock time cap.
    - Proposes seeds via uncertainty-aware acquisition (UCB or EI) over a candidate pool.
    - Supports epsilon exploration and diversity filtering; optional gradient refinement.
    - Honors fixed parameters exactly and respects provided bounds when decoding.
    """

    def __init__(
        self,
        lows: np.ndarray,
        highs: np.ndarray,
        fixed_mask: np.ndarray,
        fixed_values: np.ndarray,
        ensemble_n: int = 3,
        hidden: int = 96,
        layers: int = 2,
        dropout: float = 0.1,
        weight_decay: float = 1e-4,
        epochs: int = 8,
        time_cap_ms: int = 750,
        pool_mult: float = 3.0,
        epsilon: float = 0.1,
        acq_type: str = "ucb",
        device: str = "cpu",
        seed: Optional[int] = None,
        diversity_min_dist: float = 0.03,
        enable_grad_refine: bool = False,
        grad_steps: int = 0,
    ) -> None:
        self.lows = lows.astype(float)
        self.highs = highs.astype(float)
        self.fixed_mask = fixed_mask.astype(bool)
        self.fixed_values = fixed_values.astype(float)
        self.var_indices = np.where(~self.fixed_mask)[0]
        self.input_dim = int(self.var_indices.size)

        self.ensemble_n = int(max(1, ensemble_n))
        self.hidden = int(max(8, hidden))
        self.layers = int(max(0, layers))
        self.dropout = float(max(0.0, min(0.9, dropout)))
        self.weight_decay = float(max(0.0, weight_decay))
        self.epochs = int(max(1, epochs))
        self.time_cap_ms = int(max(50, time_cap_ms))
        self.pool_mult = float(max(1.0, pool_mult))
        self.epsilon = float(max(0.0, min(0.9, epsilon)))
        self.acq_type = (acq_type or "ucb").lower()
        self.device = device
        self.seed = int(seed) if (seed is not None and seed >= 0) else None
        self.diversity_min_dist = float(max(0.0, diversity_min_dist))
        self.enable_grad_refine = bool(enable_grad_refine)
        self.grad_steps = int(max(0, grad_steps))

        self._X: List[np.ndarray] = []  # original scale
        self._y: List[float] = []
        self._models: List[_MLP] = []
        self._torch_ok = TORCH_AVAILABLE and self.input_dim > 0
        self._rng = np.random.default_rng(self.seed)

        if self._torch_ok:
            self._device = torch.device(self.device if torch.cuda.is_available() and self.device == "cuda" else "cpu")
        else:
            self._device = None

    def _to_z(self, X: np.ndarray) -> np.ndarray:
        # Normalize variable dims to [0,1], fixed dims are ignored.
        lows = self.lows[self.var_indices]
        highs = self.highs[self.var_indices]
        span = np.maximum(highs - lows, 1e-12)
        return (X[:, self.var_indices] - lows) / span

    def _from_z(self, Z: np.ndarray) -> np.ndarray:
        # Decode normalized var dims to original scale full vector
        X = np.zeros((Z.shape[0], self.lows.shape[0]), dtype=float)
        X[:, :] = self.fixed_values  # start with fixed values everywhere
        lows = self.lows[self.var_indices]
        highs = self.highs[self.var_indices]
        span = np.maximum(highs - lows, 0.0)
        X[:, self.var_indices] = lows + Z * span
        return X

    @property
    def size(self) -> int:
        return len(self._y)

    def add_data(self, X: List[List[float]], y: List[float]) -> None:
        if X is None or y is None or len(X) == 0:
            return
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        # Basic sanity: clip absurd values to stabilize learning
        y_arr = np.where(np.isfinite(y_arr), y_arr, 1e6)
        y_arr = np.clip(y_arr, -1e6, 1e6)
        for i in range(X_arr.shape[0]):
            self._X.append(X_arr[i].copy())
            self._y.append(float(y_arr[i]))

    def _train_torch(self) -> Tuple[float, int]:
        if not self._torch_ok or self.size < max(50, 5 * max(1, self.input_dim)):
            self._models = []
            return 0.0, 0

        start_ms = _now_ms()

        X = np.asarray(self._X, dtype=float)
        y = np.asarray(self._y, dtype=float)
        Z = self._to_z(X)
        # Targets normalization (optional): center and scale roughly
        y_mean = float(np.mean(y))
        y_std = float(np.std(y) + 1e-8)
        y_norm = (y - y_mean) / y_std

        X_tensor = torch.from_numpy(Z.astype(np.float32))
        y_tensor = torch.from_numpy(y_norm.astype(np.float32))

        dataset = TensorDataset(X_tensor, y_tensor)
        val_size = max(1, int(0.1 * len(dataset))) if len(dataset) > 10 else 1
        train_size = max(1, len(dataset) - val_size)
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        self._models = []
        epochs_done = 0
        for m_idx in range(self.ensemble_n):
            model = _MLP(self.input_dim, self.hidden, self.layers, self.dropout).to(self._device)
            if self.seed is not None:
                torch.manual_seed(self.seed + m_idx * 9973)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=self.weight_decay)
            criterion = nn.MSELoss()
            train_loader = DataLoader(train_ds, batch_size=min(128, len(train_ds)), shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

            best_val = float('inf')
            patience = 3
            bad = 0
            for epoch in range(self.epochs):
                model.train()
                for xb, yb in train_loader:
                    xb = xb.to(self._device)
                    yb = yb.to(self._device)
                    opt.zero_grad()
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    loss.backward()
                    opt.step()

                # validation
                model.eval()
                with torch.no_grad():
                    vals = []
                    for xb, yb in val_loader:
                        xb = xb.to(self._device)
                        yb = yb.to(self._device)
                        pred = model(xb)
                        v = criterion(pred, yb).item()
                        vals.append(v)
                    vloss = float(np.mean(vals)) if vals else 0.0
                if vloss + 1e-6 < best_val:
                    best_val = vloss
                    bad = 0
                else:
                    bad += 1
                epochs_done += 1
                if bad >= patience:
                    break
                if (_now_ms() - start_ms) >= self.time_cap_ms:
                    break

            # Attach normalization params
            model._y_mean = y_mean
            model._y_std = y_std
            self._models.append(model)

            if (_now_ms() - start_ms) >= self.time_cap_ms:
                break

        train_time = _now_ms() - start_ms
        return train_time, epochs_done

    def train(self) -> Tuple[float, int]:
        if not self._torch_ok:
            return 0.0, 0
        try:
            return self._train_torch()
        except Exception:
            self._models = []
            return 0.0, 0

    def _predict_mu_sigma(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self._torch_ok or not self._models:
            # Fallback: no model → uniform mean, large sigma
            mu = np.full((X.shape[0],), float(np.mean(self._y)) if self._y else 1e3, dtype=float)
            sigma = np.full((X.shape[0],), 1.0, dtype=float)
            return mu, sigma
        Z = self._to_z(X)
        Zt = torch.from_numpy(Z.astype(np.float32)).to(self._device)
        preds = []
        with torch.no_grad():
            for m in self._models:
                out = m(Zt)
                # de-normalize
                out = out * m._y_std + m._y_mean
                preds.append(out.detach().cpu().numpy())
        P = np.stack(preds, axis=0)
        mu = np.mean(P, axis=0)
        sigma = np.std(P, axis=0) + 1e-8
        return mu, sigma

    def _acq_scores(self, mu: np.ndarray, sigma: np.ndarray, best_y: Optional[float], beta: float) -> np.ndarray:
        if self.acq_type == "ei":
            # Expected Improvement for minimization (vectorized)
            if best_y is None or not np.isfinite(best_y):
                return mu  # fallback
            s = np.maximum(sigma, 1e-8)
            z = (best_y - mu) / s

            if SCIPY_NORM_AVAILABLE:
                cdf = _norm.cdf(z)
                pdf = _norm.pdf(z)
            else:
                # Vectorized erf approximation for CDF; exact PDF via exp
                x = z / np.sqrt(2.0)
                sign = np.sign(x)
                ax = np.abs(x)
                t = 1.0 / (1.0 + 0.3275911 * ax)
                a1 = 0.254829592
                a2 = -0.284496736
                a3 = 1.421413741
                a4 = -1.453152027
                a5 = 1.061405429
                poly = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t
                erf_approx = sign * (1.0 - poly * np.exp(-ax * ax))
                cdf = 0.5 * (1.0 + erf_approx)
                pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z * z)

            ei = (best_y - mu) * cdf + s * pdf
            # Higher EI better → convert to score where lower is better: use -EI
            return -ei
        # UCB for minimization: mu - beta * sigma (lower better)
        return mu - float(beta) * sigma

    def _diversity_filter(self, Z: np.ndarray, idx_sorted: np.ndarray, k: int) -> List[int]:
        if k <= 0 or idx_sorted.size == 0:
            return []
        chosen: List[int] = []
        for idx in idx_sorted:
            if len(chosen) >= k:
                break
            z = Z[idx]
            ok = True
            for j in chosen:
                if np.linalg.norm(z - Z[j]) < self.diversity_min_dist:
                    ok = False
                    break
            if ok:
                chosen.append(int(idx))
        # If we couldn't fill due to strict diversity, pad greedily
        i = 0
        while len(chosen) < k and i < idx_sorted.size:
            cand = int(idx_sorted[i])
            if cand not in chosen:
                chosen.append(cand)
            i += 1
        return chosen

    def propose(
        self,
        count: int,
        beta: float,
        best_y: Optional[float] = None,
        exploration_fraction: Optional[float] = None,
    ) -> List[List[float]]:
        if count <= 0:
            return []
        # Pool size
        pool_n = int(max(count, math.ceil(self.pool_mult * count)))

        # Generate normalized candidates in var space
        if SCIPY_QMC_AVAILABLE and self.input_dim > 0:
            engine = qmc.Sobol(d=self.input_dim, scramble=True, seed=self.seed)
            m = int(np.ceil(np.log2(max(1, pool_n))))
            Z = engine.random_base2(m=m)[:pool_n]
        else:
            Z = self._rng.random((pool_n, max(1, self.input_dim))) if self.input_dim > 0 else np.zeros((pool_n, 0))

        # Optionally gradient refine top few according to acquisition
        # We'll only refine if models exist and it's enabled
        if self.enable_grad_refine and self._torch_ok and self._models and self.grad_steps > 0 and self.input_dim > 0:
            try:
                Zt = torch.from_numpy(Z.astype(np.float32)).to(self._device)
                Zt.requires_grad_(True)
                opt = torch.optim.SGD([Zt], lr=0.05)
                for _ in range(self.grad_steps):
                    opt.zero_grad()
                    # Build full X from Zt for mu/sigma via models
                    # Concatenate fixed dims back in numpy after step; here approximate using models directly on Zt
                    preds = []
                    for m in self._models:
                        out = m(Zt)
                        out = out * m._y_std + m._y_mean
                        preds.append(out)
                    P = torch.stack(preds, dim=0)
                    mu = torch.mean(P, dim=0)
                    sigma = torch.std(P, dim=0) + 1e-8
                    if self.acq_type == "ei" and best_y is not None and math.isfinite(best_y):
                        s = sigma
                        z = (best_y - mu) / s
                        # approximate normal cdf and pdf via torch
                        cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
                        pdf = (1.0 / math.sqrt(2.0 * math.pi)) * torch.exp(-0.5 * z * z)
                        ei = (best_y - mu) * cdf + s * pdf
                        loss = -ei.mean()
                    else:
                        loss = (mu - float(beta) * sigma).mean()
                    loss.backward()
                    opt.step()
                    with torch.no_grad():
                        Zt.clamp_(0.0, 1.0)
                Z = Zt.detach().cpu().numpy()
            except Exception:
                pass

        # Decode to full X and score
        X_pool = self._from_z(Z)
        mu, sigma = self._predict_mu_sigma(X_pool)
        scores = self._acq_scores(mu, sigma, best_y, beta)

        # Diversity-aware top-k selection
        idx_sorted = np.argsort(scores)
        chosen_idx = self._diversity_filter(Z, idx_sorted, count)

        # Epsilon exploration: replace a fraction with random samples
        eps = self.epsilon if exploration_fraction is None else float(exploration_fraction)
        n_eps = int(max(0, math.floor(eps * count)))
        n_exploit = count - n_eps
        exploit_idx = chosen_idx[:n_exploit]
        X_sel = X_pool[exploit_idx] if len(exploit_idx) > 0 else np.zeros((0, self.lows.shape[0]))
        if n_eps > 0:
            if self.input_dim > 0:
                Z_eps = self._rng.random((n_eps, self.input_dim))
            else:
                Z_eps = np.zeros((n_eps, 0))
            X_eps = self._from_z(Z_eps)
            X_out = np.vstack([X_sel, X_eps]) if X_sel.size else X_eps
        else:
            X_out = X_sel

        return [list(row) for row in X_out]

    def predict_mean(self, X: List[List[float]]) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        mu, _ = self._predict_mu_sigma(X_arr)
        return mu

    def predict_mu_sigma(self, X: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
        """Public method to get ensemble mean and std for given points X (original scale)."""
        X_arr = np.asarray(X, dtype=float)
        mu, sigma = self._predict_mu_sigma(X_arr)
        return mu, sigma


