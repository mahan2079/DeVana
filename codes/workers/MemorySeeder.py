import json
import os
import math
from typing import List, Optional, Tuple

import numpy as np


class MemorySeeder:
    """
    Lightweight AI-like memory seeder that learns and memorizes good seeds across runs.

    - Keeps a bounded memory of the best parameter vectors (with lowest fitness)
    - Proposes new seeds via a mixture of: replaying top seeds, Gaussian jitter around top-K,
      and uniform exploration within bounds
    - Respects fixed parameters
    - Persists memory to disk (JSON) to improve over time
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
            # Ignore corrupt files
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
            # Keep only the best max_size entries
            idxs = list(range(len(self._y)))
            idxs.sort(key=lambda i: self._y[i])  # lower fitness is better
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
        # Choose base rows at random from top-K
        idxs = self._rng.integers(0, bases.shape[0], size=n)
        base_sel = bases[idxs]
        # Gaussian jitter in var space
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
        # If memory empty, random
        if self.size == 0:
            return [list(row) for row in self._rand_var(count)]

        # Determine mixture counts
        n_replay = int(math.floor(self.replay_frac * count))
        n_explore = int(math.floor(self.exploration_frac * count))
        n_model = max(0, count - n_replay - n_explore)

        # Sort memory by fitness
        idxs = list(range(self.size))
        idxs.sort(key=lambda i: self._y[i])
        top = idxs[: min(self.top_k, len(idxs))]
        bases = np.asarray([self._X[i] for i in top], dtype=float)

        out = []
        # Replay
        if n_replay > 0:
            pick = self._rng.choice(len(top), size=min(n_replay, len(top)), replace=False)
            out.extend([list(bases[i]) for i in pick])

        # Jitter around top-K
        if n_model > 0:
            out.extend([list(row) for row in self._jitter_around(bases, n_model)])

        # Exploration
        if n_explore > 0:
            out.extend([list(row) for row in self._rand_var(n_explore)])

        # If we could not fill exactly, pad with random
        while len(out) < count:
            out.append(list(self._rand_var(1)[0]))
        return out[:count]


