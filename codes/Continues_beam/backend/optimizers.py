from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .model import BeamModel, TargetSpecification


@dataclass
class Bounds:
    k_min: float = 0.0
    k_max: float = 1e7
    c_min: float = 0.0
    c_max: float = 1e5


def _clip(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(v, lo), hi)


def optimize_values_at_locations(
    model: BeamModel,
    spring_locations: List[float],
    damper_locations: List[float],
    targets: List[TargetSpecification],
    omega: np.ndarray,
    bounds: Bounds | None = None,
    max_iters: int = 200,
    population: int = 30,
    seed: int | None = None,
    force=None,
) -> Dict:
    """
    Optimize only the magnitudes (k, c) at user-specified locations.

    Decision vector x = [k_vals (len(spring_locations)), c_vals (len(damper_locations))]
    """
    if bounds is None:
        bounds = Bounds()
    if seed is not None:
        np.random.seed(seed)

    n_k = len(spring_locations)
    n_c = len(damper_locations)
    dim = n_k + n_c

    def decode(ind: np.ndarray):
        k_vals = _clip(ind[:n_k], bounds.k_min, bounds.k_max)
        c_vals = _clip(ind[n_k:], bounds.c_min, bounds.c_max)
        k_points = list(zip(spring_locations, k_vals.tolist()))
        c_points = list(zip(damper_locations, c_vals.tolist()))
        return k_points, c_points

    # Initialize population
    pop = np.zeros((population, dim))
    pop[:, :n_k] = bounds.k_min + (bounds.k_max - bounds.k_min) * np.random.rand(population, n_k)
    pop[:, n_k:] = bounds.c_min + (bounds.c_max - bounds.c_min) * np.random.rand(population, n_c)

    best = None
    best_val = np.inf
    hist = []

    for it in range(max_iters):
        objs = np.zeros(population)
        for i in range(population):
            k_points, c_points = decode(pop[i])
            objs[i] = model.objective_from_targets(k_points, c_points, targets, omega, force=force)

        idx = int(np.argmin(objs))
        if objs[idx] < best_val:
            best_val = float(objs[idx])
            best = pop[idx].copy()
        hist.append(best_val)

        # Differential evolution mutation + crossover
        F = 0.7
        CR = 0.9
        new_pop = pop.copy()
        for i in range(population):
            a, b, c = np.random.choice(population, 3, replace=False)
            mutant = pop[a] + F * (pop[b] - pop[c])
            mask = np.random.rand(dim) < CR
            trial = np.where(mask, mutant, pop[i])
            # enforce bounds by segment
            trial[:n_k] = _clip(trial[:n_k], bounds.k_min, bounds.k_max)
            trial[n_k:] = _clip(trial[n_k:], bounds.c_min, bounds.c_max)
            # accept if better
            k_points, c_points = decode(trial)
            f_trial = model.objective_from_targets(k_points, c_points, targets, omega, force=force)
            if f_trial <= objs[i]:
                new_pop[i] = trial
        pop = new_pop

    k_points, c_points = decode(best if best is not None else pop[0])
    return {
        "k_points": k_points,
        "c_points": c_points,
        "best_objective": best_val,
        "history": np.asarray(hist),
    }


def optimize_placement_and_values(
    model: BeamModel,
    num_springs: int,
    num_dampers: int,
    targets: List[TargetSpecification],
    omega: np.ndarray,
    bounds: Bounds | None = None,
    max_iters: int = 250,
    population: int = 40,
    min_separation: float | None = None,
    seed: int | None = None,
    force=None,
) -> Dict:
    """
    Optimize both placements (x in [0,L]) and magnitudes (k, c).

    Decision vector x = [xs_k (nk), ks (nk), xs_c (nc), cs (nc)]
    """
    if bounds is None:
        bounds = Bounds()
    if seed is not None:
        np.random.seed(seed)

    L = model.L
    nk, nc = int(num_springs), int(num_dampers)
    dim = (nk + nc) * 2

    def enforce_min_separation(xs: np.ndarray) -> np.ndarray:
        if min_separation is None or xs.size <= 1:
            return xs
        xs_sorted = np.sort(xs)
        for i in range(1, xs_sorted.size):
            if xs_sorted[i] - xs_sorted[i - 1] < min_separation:
                xs_sorted[i] = xs_sorted[i - 1] + min_separation
        # wrap back into [0,L]
        xs_sorted = np.clip(xs_sorted, 0.0, L)
        return xs_sorted

    def decode(ind: np.ndarray):
        xs_k = _clip(ind[:nk], 0.0, L)
        ks = _clip(ind[nk:2 * nk], bounds.k_min, bounds.k_max)
        xs_c = _clip(ind[2 * nk:2 * nk + nc], 0.0, L)
        cs = _clip(ind[2 * nk + nc:], bounds.c_min, bounds.c_max)
        if min_separation is not None:
            xs_k = enforce_min_separation(xs_k)
            xs_c = enforce_min_separation(xs_c)
        k_points = list(zip(xs_k.tolist(), ks.tolist()))
        c_points = list(zip(xs_c.tolist(), cs.tolist()))
        return k_points, c_points

    # Initialize population
    pop = np.zeros((population, dim))
    pop[:, :nk] = L * np.random.rand(population, nk)
    pop[:, nk:2 * nk] = bounds.k_min + (bounds.k_max - bounds.k_min) * np.random.rand(population, nk)
    pop[:, 2 * nk:2 * nk + nc] = L * np.random.rand(population, nc)
    pop[:, 2 * nk + nc:] = bounds.c_min + (bounds.c_max - bounds.c_min) * np.random.rand(population, nc)

    # PSO
    vel = np.zeros_like(pop)
    pbest = pop.copy()
    pbest_val = np.full((population,), np.inf)
    gbest = pop[0].copy()
    gbest_val = np.inf

    hist = []
    for it in range(max_iters):
        # evaluate
        for i in range(population):
            k_points, c_points = decode(pop[i])
            f = model.objective_from_targets(k_points, c_points, targets, omega, force=force)
            if f < pbest_val[i]:
                pbest_val[i] = f
                pbest[i] = pop[i].copy()
            if f < gbest_val:
                gbest_val = f
                gbest = pop[i].copy()

        hist.append(float(gbest_val))

        # update velocities/positions
        w, c1, c2 = 0.72, 1.4, 1.4
        r1 = np.random.rand(population, dim)
        r2 = np.random.rand(population, dim)
        vel = w * vel + c1 * r1 * (pbest - pop) + c2 * r2 * (gbest - pop)
        pop = pop + vel

        # clip by segments
        pop[:, :nk] = _clip(pop[:, :nk], 0.0, L)
        pop[:, nk:2 * nk] = _clip(pop[:, nk:2 * nk], bounds.k_min, bounds.k_max)
        pop[:, 2 * nk:2 * nk + nc] = _clip(pop[:, 2 * nk:2 * nk + nc], 0.0, L)
        pop[:, 2 * nk + nc:] = _clip(pop[:, 2 * nk + nc:], bounds.c_min, bounds.c_max)

        if min_separation is not None:
            for i in range(population):
                xs_k = enforce_min_separation(pop[i, :nk])
                xs_c = enforce_min_separation(pop[i, 2 * nk:2 * nk + nc])
                pop[i, :nk] = xs_k
                pop[i, 2 * nk:2 * nk + nc] = xs_c

    k_points, c_points = decode(gbest)
    return {
        "k_points": k_points,
        "c_points": c_points,
        "best_objective": float(gbest_val),
        "history": np.asarray(hist),
    }
