import json
import os
import math
import time
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, random_split
    TORCH_AVAILABLE = True

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


class MemorySeeder:
    """
    Lightweight AI-like memory seeder that learns and memorizes good seeds across runs.
    """

    def __init__(
        self,
        lows: np.ndarray,
        highs: np.ndarray,
        fixed_mask: np.ndarray,
        fixed_values: np.ndarray,
        max_size: int = 1000,
        top_k: int = 50,
        sigma_scale: float = 0.05,
        exploration_frac: float = 0.2,
        replay_frac: float = 0.2,
        file_path: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.lows = lows.astype(float)
        self.highs = highs.astype(float)
        self.fixed_mask = fixed_mask.astype(bool)
        self.fixed_values = fixed_values.astype(float)
        self.var_indices = np.where(~self.fixed_mask)[0]
        self.max_size = int(max(10, max_size))
        self.top_k = int(max(1, top_k))
        self.sigma_scale = float(max(0.0, sigma_scale))
        self.exploration_frac = float(min(1.0, max(0.0, exploration_frac)))
        self.replay_frac = float(min(1.0 - self.exploration_frac, max(0.0, replay_frac)))
        self.file_path = file_path
        self._rng = np.random.default_rng(seed)
        self._X: List[List[float]] = []
        self._y: List[float] = []
        self._load()

    @property
    def size(self) -> int:
        return len(self._y)

    def _load(self) -> None:
        try:
            if self.file_path and os.path.isfile(self.file_path):
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                X = data.get('X', [])
                y = data.get('y', [])
                if isinstance(X, list) and isinstance(y, list) and len(X) == len(y):
                    self._X = [list(map(float, row)) for row in X]
                    self._y = [float(v) for v in y]
        except Exception:
            self._X, self._y = [], []

    def _save(self) -> None:
        try:
            if not self.file_path:
                return
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'X': self._X,
                    'y': self._y,
                }, f)
        except Exception:
            pass

    def add_data(self, X: List[List[float]], y: List[float]) -> None:
        if not X or not y:
            return
        try:
            for xi, yi in zip(X, y):
                if not (isinstance(xi, (list, tuple)) and np.isfinite(yi)):
                    continue
                self._X.append([float(v) for v in xi])
                self._y.append(float(yi))
            idxs = list(range(len(self._y)))
            idxs.sort(key=lambda i: self._y[i])
            idxs = idxs[: self.max_size]
            self._X = [self._X[i] for i in idxs]
            self._y = [self._y[i] for i in idxs]
            self._save()
        except Exception:
            pass

    def _rand_var(self, n: int) -> np.ndarray:
        if self.var_indices.size == 0:
            return np.zeros((n, 0))
        Z = self._rng.random((n, self.var_indices.size))
        lows = self.lows[self.var_indices]
        highs = self.highs[self.var_indices]
        span = np.maximum(highs - lows, 0.0)
        X = np.zeros((n, self.lows.shape[0]), dtype=float)
        X[:, :] = self.fixed_values
        X[:, self.var_indices] = lows + Z * span
        return X

    def _jitter_around(self, bases: np.ndarray, n: int) -> np.ndarray:
        if n <= 0:
            return np.zeros((0, self.lows.shape[0]))
        if bases.size == 0 or self.var_indices.size == 0:
            return self._rand_var(n)
        lows = self.lows[self.var_indices]
        highs = self.highs[self.var_indices]
        span = np.maximum(highs - lows, 1e-12)
        idxs = self._rng.integers(0, bases.shape[0], size=n)
        base_sel = bases[idxs]
        sigma = self.sigma_scale * span
        noise = self._rng.normal(loc=0.0, scale=sigma, size=(n, self.var_indices.size))
        var_part = np.clip(base_sel[:, self.var_indices] + noise, lows, highs)
        out = np.zeros((n, self.lows.shape[0]))
        out[:, :] = self.fixed_values
        out[:, self.var_indices] = var_part
        return out

    def propose(self, count: int) -> List[List[float]]:
        if count <= 0:
            return []
        if self.size == 0:
            return [list(row) for row in self._rand_var(count)]
        n_replay = int(math.floor(self.replay_frac * count))
        n_explore = int(math.floor(self.exploration_frac * count))
        n_model = max(0, count - n_replay - n_explore)
        idxs = list(range(self.size))
        idxs.sort(key=lambda i: self._y[i])
        top = idxs[: min(self.top_k, len(idxs))]
        bases = np.asarray([self._X[i] for i in top], dtype=float)
        out = []
        if n_replay > 0:
            pick = self._rng.choice(len(top), size=min(n_replay, len(top)), replace=False)
            out.extend([list(bases[i]) for i in pick])
        if n_model > 0:
            out.extend([list(row) for row in self._jitter_around(bases, n_model)])
        if n_explore > 0:
            out.extend([list(row) for row in self._rand_var(n_explore)])
        while len(out) < count:
            out.append(list(self._rand_var(1)[0]))
        return out[:count]


class NeuralSeeder:
    """
    Online-learning seeding via an ensemble of small MLPs.
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
        self._X: List[np.ndarray] = []
        self._y: List[float] = []
        self._models: List[_MLP] = []
        self._torch_ok = TORCH_AVAILABLE and self.input_dim > 0
        self._rng = np.random.default_rng(self.seed)
        if self._torch_ok:
            self._device = torch.device(self.device if torch.cuda.is_available() and self.device == "cuda" else "cpu")
        else:
            self._device = None

    def _to_z(self, X: np.ndarray) -> np.ndarray:
        lows = self.lows[self.var_indices]
        highs = self.highs[self.var_indices]
        span = np.maximum(highs - lows, 1e-12)
        return (X[:, self.var_indices] - lows) / span

    def _from_z(self, Z: np.ndarray) -> np.ndarray:
        X = np.zeros((Z.shape[0], self.lows.shape[0]), dtype=float)
        X[:, :] = self.fixed_values
        lows = self.lows[self.var_indices]
        highs = self.highs[self.var_indices]
        span = np.maximum(highs - lows, 0.0)
        X[:, self.var_indices] = lows + Z * span
        return X

    @property
    def size(self) -> int:
        return len(self._y)

    def add_data(self, X: List[List[float]], y: List[float]) -> None:
        if X is None or y is None or len(X) == 0: return
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        y_arr = np.where(np.isfinite(y_arr), y_arr, 1e6)
        y_arr = np.clip(y_arr, -1e6, 1e6)
        for i in range(X_arr.shape[0]):
            self._X.append(X_arr[i].copy())
            self._y.append(float(y_arr[i]))

    def train(self) -> Tuple[float, int]:
        if not self._torch_ok or self.size < max(50, 5 * max(1, self.input_dim)):
            self._models = []
            return 0.0, 0
        start_ms = _now_ms()
        X, y = np.asarray(self._X, dtype=float), np.asarray(self._y, dtype=float)
        Z = self._to_z(X)
        y_mean, y_std = float(np.mean(y)), float(np.std(y) + 1e-8)
        y_norm = (y - y_mean) / y_std
        X_tensor, y_tensor = torch.from_numpy(Z.astype(np.float32)), torch.from_numpy(y_norm.astype(np.float32))
        dataset = TensorDataset(X_tensor, y_tensor)
        val_size = max(1, int(0.1 * len(dataset))) if len(dataset) > 10 else 1
        train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
        self._models = []
        epochs_done = 0
        for m_idx in range(self.ensemble_n):
            model = _MLP(self.input_dim, self.hidden, self.layers, self.dropout).to(self._device)
            if self.seed is not None: torch.manual_seed(self.seed + m_idx * 9973)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=self.weight_decay)
            criterion = nn.MSELoss()
            train_loader = DataLoader(train_ds, batch_size=min(128, len(train_ds)), shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)
            best_val, bad = float('inf'), 0
            for epoch in range(self.epochs):
                model.train()
                for xb, yb in train_loader:
                    xb, yb = xb.to(self._device), yb.to(self._device)
                    opt.zero_grad(); pred = model(xb); loss = criterion(pred, yb); loss.backward(); opt.step()
                model.eval()
                with torch.no_grad():
                    vals = [criterion(model(xb.to(self._device)), yb.to(self._device)).item() for xb, yb in val_loader]
                    vloss = float(np.mean(vals)) if vals else 0.0
                if vloss + 1e-6 < best_val: best_val, bad = vloss, 0
                else: bad += 1
                epochs_done += 1
                if bad >= 3 or (_now_ms() - start_ms) >= self.time_cap_ms: break
            model._y_mean, model._y_std = y_mean, y_std
            self._models.append(model)
            if (_now_ms() - start_ms) >= self.time_cap_ms: break
        return _now_ms() - start_ms, epochs_done

    def _predict_mu_sigma(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self._torch_ok or not self._models:
            return np.full((X.shape[0],), float(np.mean(self._y)) if self._y else 1e3), np.full((X.shape[0],), 1.0)
        Z = self._to_z(X)
        Zt = torch.from_numpy(Z.astype(np.float32)).to(self._device)
        preds = []
        with torch.no_grad():
            for m in self._models:
                preds.append((m(Zt) * m._y_std + m._y_mean).detach().cpu().numpy())
        P = np.stack(preds, axis=0)
        return np.mean(P, axis=0), np.std(P, axis=0) + 1e-8

    def _acq_scores(self, mu: np.ndarray, sigma: np.ndarray, best_y: Optional[float], beta: float) -> np.ndarray:
        if self.acq_type == "ei" and best_y is not None and np.isfinite(best_y):
            s = np.maximum(sigma, 1e-8)
            z = (best_y - mu) / s
            if SCIPY_NORM_AVAILABLE: cdf, pdf = _norm.cdf(z), _norm.pdf(z)
            else:
                x = z / np.sqrt(2.0); sign = np.sign(x); ax = np.abs(x); t = 1.0 / (1.0 + 0.3275911 * ax)
                poly = (((((1.061405429 * t + -1.453152027) * t) + 1.421413741) * t + -0.284496736) * t + 0.254829592) * t
                cdf = 0.5 * (1.0 + sign * (1.0 - poly * np.exp(-ax * ax))); pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z * z)
            return -((best_y - mu) * cdf + s * pdf)
        return mu - float(beta) * sigma

    def _diversity_filter(self, Z: np.ndarray, idx_sorted: np.ndarray, k: int) -> List[int]:
        chosen: List[int] = []
        for idx in idx_sorted:
            if len(chosen) >= k: break
            if all(np.linalg.norm(Z[idx] - Z[j]) >= self.diversity_min_dist for j in chosen): chosen.append(int(idx))
        i = 0
        while len(chosen) < k and i < idx_sorted.size:
            if int(idx_sorted[i]) not in chosen: chosen.append(int(idx_sorted[i]))
            i += 1
        return chosen

    def propose(self, count: int, beta: float, best_y: Optional[float] = None, exploration_fraction: Optional[float] = None) -> List[List[float]]:
        if count <= 0: return []
        pool_n = int(max(count, math.ceil(self.pool_mult * count)))
        if SCIPY_QMC_AVAILABLE and self.input_dim > 0:
            Z = qmc.Sobol(d=self.input_dim, scramble=True, seed=self.seed).random_base2(m=int(np.ceil(np.log2(max(1, pool_n)))))[:pool_n]
        else: Z = self._rng.random((pool_n, max(1, self.input_dim))) if self.input_dim > 0 else np.zeros((pool_n, 0))
        if self.enable_grad_refine and self._torch_ok and self._models and self.grad_steps > 0 and self.input_dim > 0:
            try:
                Zt = torch.from_numpy(Z.astype(np.float32)).to(self._device).requires_grad_(True)
                opt = torch.optim.SGD([Zt], lr=0.05)
                for _ in range(self.grad_steps):
                    opt.zero_grad(); P = torch.stack([(m(Zt) * m._y_std + m._y_mean) for m in self._models], dim=0)
                    mu, sigma = torch.mean(P, dim=0), torch.std(P, dim=0) + 1e-8
                    if self.acq_type == "ei" and best_y is not None and math.isfinite(best_y):
                        z = (best_y - mu) / sigma; cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0))); pdf = (1.0 / math.sqrt(2.0 * math.pi)) * torch.exp(-0.5 * z * z)
                        loss = -((best_y - mu) * cdf + sigma * pdf).mean()
                    else: loss = (mu - float(beta) * sigma).mean()
                    loss.backward(); opt.step() 
                    with torch.no_grad(): Zt.clamp_(0.0, 1.0)
                Z = Zt.detach().cpu().numpy()
            except Exception: pass
        X_pool = self._from_z(Z); mu, sigma = self._predict_mu_sigma(X_pool); scores = self._acq_scores(mu, sigma, best_y, beta)
        chosen_idx = self._diversity_filter(Z, np.argsort(scores), count)
        eps = self.epsilon if exploration_fraction is None else float(exploration_fraction)
        n_eps = int(max(0, math.floor(eps * count))); n_exploit = count - n_eps; exploit_idx = chosen_idx[:n_exploit]
        X_sel = X_pool[exploit_idx] if len(exploit_idx) > 0 else np.zeros((0, self.lows.shape[0]))
        if n_eps > 0:
            X_eps = self._from_z(self._rng.random((n_eps, self.input_dim)) if self.input_dim > 0 else np.zeros((n_eps, 0)))
            X_out = np.vstack([X_sel, X_eps]) if X_sel.size else X_eps
        else: X_out = X_sel
        return [list(row) for row in X_out]

    def predict_mean(self, X: List[List[float]]) -> np.ndarray:
        mu, _ = self._predict_mu_sigma(np.asarray(X, dtype=float))
        return mu

    def predict_mu_sigma(self, X: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
        return self._predict_mu_sigma(np.asarray(X, dtype=float))
