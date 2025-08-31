import numpy as np
from typing import List, Tuple, Dict, Callable
from dataclasses import dataclass
from .model import BeamModel, TargetSpecification


def _bounded_random(lower: float, upper: float, size: int) -> np.ndarray:
    return lower + (upper - lower) * np.random.rand(size)


def _clip_array(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return np.minimum(np.maximum(values, lower), upper)


def optimize_values_only(
    model: BeamModel,
    candidate_locations: List[float],
    num_springs: int,
    num_dampers: int,
    targets: List[TargetSpecification],
    omega: np.ndarray,
    k_bounds: Tuple[float, float] = (0.0, 1e7),
    c_bounds: Tuple[float, float] = (0.0, 1e5),
    max_iters: int = 150,
    population: int = 30,
) -> Dict:
    """
    Optimize magnitudes of springs and dampers at fixed locations.
    Locations are taken as the first num_springs and num_dampers entries from candidate_locations.
    Returns dict with optimal k_points, c_points, and objective history.
    """
    xlocs_k = candidate_locations[:num_springs]
    xlocs_c = candidate_locations[:num_dampers]

    def decode(ind: np.ndarray):
        k_vals = _clip_array(ind[:num_springs], *k_bounds)
        c_vals = _clip_array(ind[num_springs:], *c_bounds)
        k_points = list(zip(xlocs_k, k_vals.tolist()))
        c_points = list(zip(xlocs_c, c_vals.tolist()))
        return k_points, c_points

    dim = num_springs + num_dampers

    # Initialize population
    pop = np.zeros((population, dim))
    for i in range(population):
        pop[i, :num_springs] = _bounded_random(*k_bounds, num_springs)
        pop[i, num_springs:] = _bounded_random(*c_bounds, num_dampers)

    best = None
    best_obj = np.inf
    history = []

    for it in range(max_iters):
        objs = np.zeros(population)
        for i in range(population):
            k_points, c_points = decode(pop[i])
            objs[i] = model.objective_from_targets(k_points, c_points, targets, omega)

        # Selection
        idx = np.argmin(objs)
        if objs[idx] < best_obj:
            best_obj = float(objs[idx])
            best = pop[idx].copy()
        history.append(best_obj)

        # Differential evolution style update
        F = 0.7
        CR = 0.9
        new_pop = pop.copy()
        for i in range(population):
            a, b, c = np.random.choice(population, 3, replace=False)
            mutant = pop[a] + F * (pop[b] - pop[c])
            # Crossover
            cross = np.random.rand(dim) < CR
            trial = np.where(cross, mutant, pop[i])
            # Clip into bounds per segment
            trial[:num_springs] = _clip_array(trial[:num_springs], *k_bounds)
            trial[num_springs:] = _clip_array(trial[num_springs:], *c_bounds)

            # Accept if better
            k_points, c_points = decode(trial)
            f_trial = model.objective_from_targets(k_points, c_points, targets, omega)
            if f_trial <= objs[i]:
                new_pop[i] = trial
        pop = new_pop

    k_points, c_points = decode(best)
    return {
        'k_points': k_points,
        'c_points': c_points,
        'best_objective': best_obj,
        'history': np.array(history),
    }


def optimize_placement_and_values(
    model: BeamModel,
    num_springs: int,
    num_dampers: int,
    targets: List[TargetSpecification],
    omega: np.ndarray,
    k_bounds: Tuple[float, float] = (0.0, 1e7),
    c_bounds: Tuple[float, float] = (0.0, 1e5),
    max_iters: int = 200,
    population: int = 40,
) -> Dict:
    """
    Jointly optimize locations (continuous 0..L) and magnitudes for springs and dampers.
    Decision vector: [x_k (num_springs), k_values (num_springs), x_c (num_dampers), c_values (num_dampers)]
    Returns optimal k_points, c_points.
    """
    L = model.L
    dim = (num_springs + num_dampers) * 2

    def decode(ind: np.ndarray):
        xs_k = _clip_array(ind[:num_springs], 0.0, L)
        ks = _clip_array(ind[num_springs:2 * num_springs], *k_bounds)
        xs_c = _clip_array(ind[2 * num_springs:2 * num_springs + num_dampers], 0.0, L)
        cs = _clip_array(ind[2 * num_springs + num_dampers:], *c_bounds)
        k_points = list(zip(xs_k.tolist(), ks.tolist()))
        c_points = list(zip(xs_c.tolist(), cs.tolist()))
        return k_points, c_points

    # Initialize population
    pop = np.zeros((population, dim))
    for i in range(population):
        # random positions and values
        pop[i, :num_springs] = _bounded_random(0.0, L, num_springs)
        pop[i, num_springs:2 * num_springs] = _bounded_random(*k_bounds, num_springs)
        pop[i, 2 * num_springs:2 * num_springs + num_dampers] = _bounded_random(0.0, L, num_dampers)
        pop[i, 2 * num_springs + num_dampers:] = _bounded_random(*c_bounds, num_dampers)

    best = None
    best_obj = np.inf
    history = []

    for it in range(max_iters):
        objs = np.zeros(population)
        for i in range(population):
            k_points, c_points = decode(pop[i])
            objs[i] = model.objective_from_targets(k_points, c_points, targets, omega)

        # Track best
        idx = np.argmin(objs)
        if objs[idx] < best_obj:
            best_obj = float(objs[idx])
            best = pop[idx].copy()
        history.append(best_obj)

        # PSO-like position/velocity update for diversity
        if it == 0:
            vel = np.zeros_like(pop)
            pbest = pop.copy()
            pbest_val = objs.copy()
            gbest = pop[idx].copy()
        else:
            w = 0.7
            c1 = 1.4
            c2 = 1.4
            r1 = np.random.rand(population, dim)
            r2 = np.random.rand(population, dim)
            vel = w * vel + c1 * r1 * (pbest - pop) + c2 * r2 * (gbest - pop)
            pop = pop + vel

            # Clip by segments
            pop[:, :num_springs] = _clip_array(pop[:, :num_springs], 0.0, L)
            pop[:, num_springs:2 * num_springs] = _clip_array(pop[:, num_springs:2 * num_springs], *k_bounds)
            pop[:, 2 * num_springs:2 * num_springs + num_dampers] = _clip_array(pop[:, 2 * num_springs:2 * num_springs + num_dampers], 0.0, L)
            pop[:, 2 * num_springs + num_dampers:] = _clip_array(pop[:, 2 * num_springs + num_dampers:], *c_bounds)

            # Update personal and global bests
            new_objs = np.zeros(population)
            for i in range(population):
                k_points, c_points = decode(pop[i])
                new_objs[i] = model.objective_from_targets(k_points, c_points, targets, omega)
            improve = new_objs < pbest_val
            pbest[improve] = pop[improve]
            pbest_val[improve] = new_objs[improve]
            gidx = int(np.argmin(pbest_val))
            gbest = pbest[gidx].copy()

    k_points, c_points = decode(best)
    return {
        'k_points': k_points,
        'c_points': c_points,
        'best_objective': best_obj,
        'history': np.array(history),
    }



