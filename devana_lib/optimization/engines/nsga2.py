import os
import json
import time
import random
import psutil
import numpy as np
from typing import Callable, Any, Dict, List, Tuple, Optional
from deap import base, creator, tools

from devana_lib.optimization.config import OptimizationConfig
from devana_lib.optimization.controller import ExecutionController
from devana_lib.optimization.utils import safe_deap_operation
from devana_lib.physics.frf import frf

class NSGA2Engine:
    """
    Headless NSGA-II optimization engine decoupled from Qt.
    """

    def _evaluate(
        self, individual: List[float], config: OptimizationConfig
    ) -> Tuple[float, float, float]:
        try:
            target_weights = config.physics.target_values_weights

            def get_target_values(idx: int) -> List[float]: 
                return target_weights[idx].values if len(target_weights) > idx else []

            def get_weights(idx: int) -> List[float]: 
                return target_weights[idx].weights if len(target_weights) > idx else []

            f_range = config.physics.frequency_range
            results = frf(
                main_system_parameters=config.physics.main_system_parameters,
                dva_parameters=tuple(individual),
                omega_start=f_range.start,
                omega_end=f_range.end,
                omega_points=f_range.points,
                target_values_mass1=get_target_values(0), weights_mass1=get_weights(0),
                target_values_mass2=get_target_values(1), weights_mass2=get_weights(1),
                target_values_mass3=get_target_values(2), weights_mass3=get_weights(2),
                target_values_mass4=get_target_values(3), weights_mass4=get_weights(3),
                target_values_mass5=get_target_values(4), weights_mass5=get_weights(4),
                plot_figure=False,
                show_peaks=False,
                show_slopes=False
            )
            f1 = results.get('singular_response', 1e6)
            if not np.isfinite(f1):
                f1 = 1e6
        except Exception:
            f1 = 1e6

        n_active = np.sum(np.array(individual) > config.sparsity.tau)
        f2 = (config.sparsity.alpha * n_active + 
              config.sparsity.beta * np.sum(np.abs(individual)))

        cost_coeffs = [p.cost_coeff for p in config.physics.dva_parameters]
        f3 = np.sum(np.array(individual) * np.array(cost_coeffs))

        return f1, f2, f3

    @safe_deap_operation
    def run(
        self, 
        config: OptimizationConfig, 
        controller: ExecutionController,
        on_progress: Callable[[int, int, int, Dict[str, Any]], None], 
        on_finished: Callable[[List[Dict[str, Any]]], None], 
        on_error: Callable[[str], None],
        random_seed: Optional[int] = None,
        convergence_epsilon: float = 0.001,
        convergence_window: int = 50,
        convergence_min_gen: int = 500,
        hv_ref_point: Optional[List[float]] = None
    ) -> None:
        try:
            hv_ref_point = hv_ref_point if hv_ref_point else [1.0, 100.0, 100.0]
            
            all_runs_data = []
            process = psutil.Process()

            results_dir = "nsga2_results"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            dva_params = config.physics.dva_parameters
            low_bounds = [p.low for p in dva_params]
            high_bounds = [p.high for p in dva_params]
            fixed_params = {
                i: p.fixed_value for i, p in enumerate(dva_params) if p.is_fixed
            }

            for run_idx in range(config.hyperparameters.num_runs):
                if controller.should_abort(): 
                    return
                
                seed = random_seed + run_idx if random_seed is not None else None
                if seed is not None:
                    random.seed(seed)
                    np.random.seed(seed)

                if not hasattr(creator, "FitnessMulti"):
                    creator.create(
                        "FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0)
                    )
                if not hasattr(creator, "Individual"):
                    creator.create(
                        "Individual", list, fitness=creator.FitnessMulti
                    )

                toolbox = base.Toolbox()
                def attr_float(i: int) -> float:
                    if i in fixed_params: 
                        return fixed_params[i]
                    return random.uniform(low_bounds[i], high_bounds[i])
                
                toolbox.register("attr_float", attr_float)
                toolbox.register(
                    "individual", tools.initIterate, creator.Individual, 
                    lambda: [toolbox.attr_float(i) for i in range(len(low_bounds))]
                )
                toolbox.register(
                    "population", tools.initRepeat, list, toolbox.individual
                )
                toolbox.register("evaluate", lambda ind: self._evaluate(ind, config))
                
                toolbox.register(
                    "mate", tools.cxSimulatedBinaryBounded, 
                    low=low_bounds, up=high_bounds, 
                    eta=config.hyperparameters.eta_c
                )
                toolbox.register(
                    "mutate", tools.mutPolynomialBounded, 
                    low=low_bounds, up=high_bounds, 
                    eta=config.hyperparameters.eta_m, 
                    indpb=config.hyperparameters.indpb
                )
                toolbox.register("select", tools.selNSGA2)

                pop = toolbox.population(n=config.hyperparameters.pop_size)
                fitnesses = list(map(toolbox.evaluate, pop))
                for ind, fit in zip(pop, fitnesses): 
                    ind.fitness.values = fit

                # DEAP's selTournamentDCD requires crowding_dist. 
                pop = toolbox.select(pop, len(pop))

                generation_metrics = []
                start_time = time.time()

                for gen in range(config.hyperparameters.generations):
                    controller.wait_if_paused()
                    if controller.should_abort(): 
                        return

                    gen_start = time.time()

                    offspring = tools.selTournamentDCD(pop, len(pop))
                    offspring = [toolbox.clone(ind) for ind in offspring]

                    for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                        if random.random() < config.hyperparameters.cxpb:
                            toolbox.mate(ind1, ind2)
                            del ind1.fitness.values
                            del ind2.fitness.values

                    for ind in offspring:
                        if random.random() < config.hyperparameters.mutpb:
                            toolbox.mutate(ind)
                            del ind.fitness.values

                    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                    fitnesses = map(toolbox.evaluate, invalid_ind)
                    for ind, fit in zip(invalid_ind, fitnesses): 
                        ind.fitness.values = fit

                    pop = toolbox.select(
                        pop + offspring, config.hyperparameters.pop_size
                    )
                    
                    pareto_front = tools.sortNondominated(
                        pop, len(pop), first_front_only=True
                    )[0]
                    try:
                        from deap.tools._hypervolume import hv
                        objs = [ind.fitness.values for ind in pareto_front]
                        hypervolume_val = hv.hypervolume(objs, hv_ref_point)
                    except Exception:
                        hypervolume_val = 0.0

                    metrics = {
                        "Gen": gen + 1,
                        "HV": hypervolume_val,
                        "IGD+": 0.0, 
                        "GD": 0.0, 
                        "Spread": 0.0, 
                        "N_Pareto": len(pareto_front),
                        "Diversity": 0.0, 
                        "Time (s)": time.time() - start_time,
                        "Memory (MB)": process.memory_info().rss / 1024 / 1024,
                        "Rank Diversity": 0.0, 
                        "Best Fitness (f1,f2,f3)": str(
                            min([ind.fitness.values for ind in pop])
                        ),
                        "time_gen": time.time() - gen_start
                    }
                    generation_metrics.append(metrics)
                    on_progress(
                        run_idx, gen + 1, config.hyperparameters.generations, metrics
                    )

                    if (gen + 1) > convergence_min_gen and (gen + 1) % convergence_window == 0:
                        recent_hvs = [
                            m['HV'] for m in generation_metrics[-convergence_window:]
                        ]
                        if np.max(recent_hvs) - np.min(recent_hvs) < convergence_epsilon:
                            break

                final_pareto = tools.sortNondominated(
                    pop, len(pop), first_front_only=True
                )[0]
                run_data = {
                    "run_id": run_idx + 1,
                    "total_time_hours": (time.time() - start_time) / 3600,
                    "final_HV": generation_metrics[-1]['HV'] if generation_metrics else 0,
                    "pareto_size": len(final_pareto),
                    "generation_metrics": generation_metrics,
                    "final_pareto_front_objectives": [
                        list(ind.fitness.values) for ind in final_pareto
                    ],
                    "final_population_parameters": [list(ind) for ind in final_pareto],
                    "convergence_gen": len(generation_metrics)
                }
                all_runs_data.append(run_data)

                file_path = os.path.join(results_dir, f"nsga2_run_{run_idx + 1}.json")
                with open(file_path, 'w') as f:
                    json.dump(run_data, f, indent=4)

            on_finished(all_runs_data)
        except Exception as e:
            on_error(str(e))
