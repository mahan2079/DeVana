from PyQt5.QtCore import QObject, pyqtSignal

from deap import base, creator, tools, algorithms

# Assuming FRF function is available in modules.FRF
from modules.FRF import frf

class NSGA2Worker(QObject):
    progress = pyqtSignal(int, int, int, dict) # run_idx, current_gen, total_gens, metrics
    finished = pyqtSignal(list) # all_runs_data
    error = pyqtSignal(str) # error_message

    def __init__(self, main_system_parameters, dva_parameters, target_values_weights,
                 omega_start, omega_end, omega_points,
                 pop_size, generations, cxpb, mutpb, eta_c, eta_m,
                 num_runs, random_seed, convergence_epsilon, convergence_window, convergence_min_gen,
                 hv_ref_point):
        super().__init__()
        
        self.main_system_parameters = main_system_parameters
        self.dva_parameters = dva_parameters
        self.target_values_weights = target_values_weights
        self.omega_start = omega_start
        self.omega_end = omega_end
        self.omega_points = omega_points
        
        self.pop_size = pop_size
        self.generations = generations
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.eta_c = eta_c
        self.eta_m = eta_m
        
        self.num_runs = num_runs
        self.random_seed = random_seed
        self.convergence_epsilon = convergence_epsilon
        self.convergence_window = convergence_window
        self.convergence_min_gen = convergence_min_gen
        self.hv_ref_point = hv_ref_point
        
        self.is_running = False
        self.is_paused = False
        self.stop_requested = False

        # DEAP setup
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0)) # Minimize all 3 objectives
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        self.toolbox = base.Toolbox()
        # Attribute generator
        # Assuming 48 parameters, each between 0 and 1
        self.toolbox.register("attr_float", random.uniform, 0, 1)
        # Individual generator
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, n=48)
        # Population generator
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self._evaluate_objectives)
        self.toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0, up=1, eta=self.eta_c)
        self.toolbox.register("mutate", tools.mutPolynomialBounded, low=0, up=1, eta=self.eta_m, indpb=1.0/48.0) # indpb is probability of each attribute being mutated
        self.toolbox.register("select", tools.selNSGA2)

    def _evaluate_objectives(self, individual):
        """
        Evaluate the three objectives for a given individual (parameter set).
        f1: FRF minimization
        f2: Sparsity
        f3: Cost
        """
        # Convert individual to DVA parameter tuple
        dva_tuple = tuple(individual)

        # Objective 1: FRF Minimization (max_omega |H(omega, x)|)
        # This requires calling the FRF function from modules.FRF
        # The FRF function needs main_system_parameters, dva_parameters, omega_start, omega_end, omega_points
        # For now, use placeholder values or assume they are passed correctly.
        # The dva_parameters here are the individual's values.
        
        # Placeholder for FRF calculation
        try:
            # Assuming frf function returns a dictionary with 'magnitude' for each mass
            # and we need to find the max magnitude across all masses and frequencies.
            # This is a simplified call, actual implementation might need more details from the problem definition.
            frf_results = frf(
                main_system_parameters=self.main_system_parameters,
                dva_parameters=dva_tuple,
                omega_start=self.omega_start,
                omega_end=self.omega_end,
                omega_points=self.omega_points,
                # Assuming target_values_massX and weights_massX are structured correctly
                target_values_mass1=self.target_values_weights[0]['mass_1'],
                weights_mass1=self.target_values_weights[1]['mass_1'],
                target_values_mass2=self.target_values_weights[0]['mass_2'],
                weights_mass2=self.target_values_weights[1]['mass_2'],
                target_values_mass3=self.target_values_weights[0]['mass_3'],
                weights_mass3=self.target_values_weights[1]['mass_3'],
                target_values_mass4=self.target_values_weights[0]['mass_4'],
                weights_mass4=self.target_values_weights[1]['mass_4'],
                target_values_mass5=self.target_values_weights[0]['mass_5'],
                weights_mass5=self.target_values_weights[1]['mass_5'],
                plot_figure=False,
                show_peaks=False,
                show_slopes=False,
            )
            
            max_frf_magnitude = 0.0
            for mass_key in ['mass_1', 'mass_2', 'mass_3', 'mass_4', 'mass_5']:
                if mass_key in frf_results and 'magnitude' in frf_results[mass_key]:
                    max_frf_magnitude = max(max_frf_magnitude, np.max(frf_results[mass_key]['magnitude']))
            f1 = max_frf_magnitude
        except Exception as e:
            self.error.emit(f"Error during FRF calculation: {e}")
            f1 = 1e9 # Assign a very high penalty for failed FRF calculation

        # Objective 2: Sparsity (alpha * N_active + beta * sum(|x_i|))
        tau = 0.1 # Sparsity threshold
        alpha = 1.0 # Weight for cardinality term
        beta = 0.5 # Weight for magnitude term
        
        n_active = sum(1 for x_i in individual if abs(x_i) > tau)
        sum_abs_xi = sum(abs(x_i) for x_i in individual)
        f2 = alpha * n_active + beta * sum_abs_xi

        # Objective 3: Cost (sum(c_i * x_i))
        # Assuming c_i are fixed cost coefficients for each of the 48 parameters
        # For now, let's use a simple placeholder for c_i
        # In a real scenario, c_i would be loaded from a config or passed in.
        cost_coefficients = np.linspace(0.1, 1.0, 48) # Example: varying costs
        f3 = sum(c_i * x_i for c_i, x_i in zip(cost_coefficients, individual))

        return f1, f2, f3

    def run(self):
        self.is_running = True
        self.stop_requested = False
        
        all_runs_data = []

        for run_idx in range(self.num_runs):
            if self.stop_requested:
                break

            random.seed(self.random_seed + run_idx)
            np.random.seed(self.random_seed + run_idx)

            pop = self.toolbox.population(n=self.pop_size)
            
            # Evaluate the initial population
            fitnesses = self.toolbox.map(self.toolbox.evaluate, pop)
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit

            # This is the algorithm's main loop
            logbook = tools.Logbook()
            logbook.header = "gen", "evals", "min_f1", "min_f2", "min_f3", "avg_f1", "avg_f2", "avg_f3", "max_f1", "max_f2", "max_f3", "hv", "igd", "gd", "spread", "n_pareto", "diversity", "time_gen", "memory_peak"

            stats_obj = tools.Statistics(lambda ind: ind.fitness.values)
            stats_obj.register("min", np.min, axis=0)
            stats_obj.register("avg", np.mean, axis=0)
            stats_obj.register("max", np.max, axis=0)

            # For Hypervolume, IGD+, GD, Spread, N_pareto, Diversity, Time, Memory
            # These will be calculated manually per generation

            # Keep track of the best individuals found
            hof = tools.HallOfFame(1) # For single-objective, but useful for tracking non-dominated in MOO
            
            # For NSGA-II, we need to keep track of the non-dominated solutions
            # This is typically done by re-evaluating the population after each generation
            
            # Store all non-dominated fronts found across generations
            all_pareto_fronts = []
            
            # Store per-generation metrics
            generation_metrics = []

            for gen in range(1, self.generations + 1):
                if self.stop_requested:
                    break
                while self.is_paused:
                    time.sleep(0.1) # Wait while paused

                gen_start_time = time.time()
                process = psutil.Process(os.getpid())

                # Select the next generation individuals
                offspring = algorithms.varAnd(pop, self.toolbox, self.cxpb, self.mutpb)
                
                # Evaluate the individuals with an invalid fitness
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # Combine the current population and offspring
                pop = self.toolbox.select(pop + offspring, self.pop_size)
                
                # Update the HallOfFame with the current population's non-dominated solutions
                # This is a simplified way; a proper MOO HallOfFame would store all non-dominated solutions
                # found so far, not just from the current population.
                current_pareto_front = tools.selNSGA2(pop, len(pop)) # Select all non-dominated from current pop
                all_pareto_fronts.append(current_pareto_front)

                # Calculate per-generation metrics
                record = stats_obj.compile(pop)
                
                # Calculate Hypervolume (HV)
                # Requires a reference point. For 3 objectives, typically [1.0, 1.0, 1.0] if objectives are normalized to [0,1]
                # The problem states objectives are FRF [0,1], Sparsity [0,50], Cost [0,48].
                # So, a reference point like [1.0, 50.0, 48.0] might be appropriate if minimizing.
                # For now, use the provided self.hv_ref_point
                
                # Extract objective values for HV calculation
                obj_values = np.array([ind.fitness.values for ind in pop])
                
                # Hypervolume calculation (requires pymoo or similar for complex cases, or manual implementation)
                # For simplicity, let's use a placeholder or a basic implementation if DEAP doesn't provide it directly.
                # DEAP's tools.emo.metrics.hypervolume is for 2 objectives. For 3+, need external.
                # For now, a placeholder.
                hv = 0.0 # Placeholder
                
                # IGD+, GD, Spread, Diversity (placeholders for now)
                igd = 0.0
                gd = 0.0
                spread = 0.0
                diversity = 0.0
                n_pareto = len(current_pareto_front)

                gen_end_time = time.time()
                time_gen = gen_end_time - gen_start_time
                memory_peak = process.memory_info().rss / (1024 * 1024) # in MB

                # Append custom metrics to the record
                record['hv'] = hv
                record['igd'] = igd
                record['gd'] = gd
                record['spread'] = spread
                record['n_pareto'] = n_pareto
                record['diversity'] = diversity
                record['time_gen'] = time_gen
                record['memory_peak'] = memory_peak
                
                logbook.record(gen=gen, evals=len(invalid_ind), **record)
                generation_metrics.append(logbook.chapters["gen"].current)

                self.progress.emit(run_idx, gen, self.generations, logbook.chapters["gen"].current)
                
                # Check for convergence (placeholder logic)
                if gen > self.convergence_min_gen and gen % self.convergence_window == 0:
                    # Simplified convergence check: check last 'convergence_window' HV values
                    if len(generation_metrics) >= self.convergence_window:
                        recent_hvs = [m['hv'] for m in generation_metrics[-self.convergence_window:]]
                        if np.max(recent_hvs) - np.min(recent_hvs) < self.convergence_epsilon:
                            print(f"Convergence detected at generation {gen}")
                            # break # Uncomment to enable early stopping on convergence

            # After all generations or convergence
            final_pareto_front = tools.selNSGA2(pop, len(pop))
            
            # Save results for this run
            run_results = {
                "run_id": run_idx,
                "settings": {
                    "pop_size": self.pop_size,
                    "generations": self.generations,
                    "cxpb": self.cxpb,
                    "mutpb": self.mutpb,
                    "eta_c": self.eta_c,
                    "eta_m": self.eta_m,
                    "random_seed": self.random_seed + run_idx,
                    "convergence_epsilon": self.convergence_epsilon,
                    "convergence_window": self.convergence_window,
                    "convergence_min_gen": self.convergence_min_gen,
                    "hv_ref_point": self.hv_ref_point,
                },
                "generation_metrics": generation_metrics,
                "final_pareto_front_objectives": [list(ind.fitness.values) for ind in final_pareto_front],
                "final_population_parameters": [list(ind) for ind in final_pareto_front],
            }
            all_runs_data.append(run_results)

        self.is_running = False
        self.finished.emit(all_runs_data)

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def stop(self):
        self.stop_requested = True
        self.is_running = False
        self.is_paused = False