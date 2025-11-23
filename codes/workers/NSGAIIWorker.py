"""
This module contains the NSGAIIWorker class for multi-objective optimization.
"""
import sys
import numpy as np
import os
import time
import random
import traceback
from deap import base, creator, tools
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition
from modules.FRF import frf

def safe_deap_operation(func):
    """
    Decorator to safely execute functions that use DEAP, with error recovery and retries.
    """
    def wrapper(*args, **kwargs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt < max_retries - 1:
                    try:
                        if hasattr(creator, "FitnessMulti"):
                            delattr(creator, "FitnessMulti")
                        if hasattr(creator, "Individual"):
                            delattr(creator, "Individual")
                    except Exception:
                        pass
                else:
                    raise
    return wrapper

class NSGAIIWorker(QThread):
    """
    Background worker thread that executes the NSGA-II multi-objective optimization.
    """
    finished = pyqtSignal(list, list)  # Emits Pareto front and parameter names
    error = pyqtSignal(str)
    update = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, main_params, target_values_dict, weights_dict, omega_start, omega_end, omega_points,
                 pop_size, num_generations, cxpb, mutpb, parameter_data, cost_threshold):
        super().__init__()
        self.main_params = main_params
        self.target_values_dict = target_values_dict
        self.weights_dict = weights_dict
        self.omega_start = omega_start
        self.omega_end = omega_end
        self.omega_points = omega_points
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.parameter_data = parameter_data
        self.cost_threshold = cost_threshold

        self.mutex = QMutex()
        self.condition = QWaitCondition()
        self.abort = False
        self.paused = False

    def __del__(self):
        self.mutex.lock()
        self.abort = True
        self.condition.wakeAll()
        self.mutex.unlock()
        self.wait()

    def pause(self):
        self.mutex.lock()
        self.paused = True
        self.mutex.unlock()

    def resume(self):
        self.mutex.lock()
        self.paused = False
        self.condition.wakeAll()
        self.mutex.unlock()

    def stop(self):
        self.mutex.lock()
        self.abort = True
        self.paused = False
        self.condition.wakeAll()
        self.mutex.unlock()

    def _check_pause_abort(self):
        self.mutex.lock()
        while self.paused and not self.abort:
            self.condition.wait(self.mutex)
        aborted = self.abort
        self.mutex.unlock()
        return aborted

    @safe_deap_operation
    def run(self):
        """
        Main execution method for the NSGA-II optimization.
        """
        try:
            self.update.emit("Setting up NSGA-II optimization...")

            parameter_names = [p[0] for p in self.parameter_data]
            bounds_low = [p[1] for p in self.parameter_data]
            bounds_up = [p[2] for p in self.parameter_data]
            dva_costs = [p[3] for p in self.parameter_data]

            # Define a multi-objective fitness function (4 objectives to minimize)
            creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))
            creator.create("Individual", list, fitness=creator.FitnessMulti)

            toolbox = base.Toolbox()
            
            # Attribute generator
            def attr_float(i):
                return random.uniform(bounds_low[i], bounds_up[i])

            toolbox.register("attr_float", attr_float)
            toolbox.register("individual", tools.initIterate, creator.Individual, 
                             lambda: [attr_float(i) for i in range(len(parameter_names))])
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            def evaluate_individual(individual):
                dva_params = tuple(individual)
                try:
                    results = frf(
                        main_system_parameters=self.main_params,
                        dva_parameters=dva_params,
                        omega_start=self.omega_start,
                        omega_end=self.omega_end,
                        omega_points=self.omega_points,
                        target_values_mass1=self.target_values_dict['mass_1'],
                        weights_mass1=self.weights_dict['mass_1'],
                        target_values_mass2=self.target_values_dict['mass_2'],
                        weights_mass2=self.weights_dict['mass_2'],
                        target_values_mass3=self.target_values_dict['mass_3'],
                        weights_mass3=self.weights_dict['mass_3'],
                        target_values_mass4=self.target_values_dict['mass_4'],
                        weights_mass4=self.weights_dict['mass_4'],
                        target_values_mass5=self.target_values_dict['mass_5'],
                        weights_mass5=self.weights_dict['mass_5'],
                        plot_figure=False
                    )

                    if 'singular_response' not in results or not np.isfinite(results['singular_response']):
                        return (1e6, 1e6, 1e6, 1e6)

                    obj1_performance = abs(results['singular_response'] - 1.0)
                    obj2_sparsity = sum(abs(p) for p in dva_params)
                    
                    obj3_cost = sum(dva_costs[i] for i, p in enumerate(individual) if abs(p) > self.cost_threshold)
                    
                    percentage_error_sum = 0.0
                    if "percentage_differences" in results:
                        for mass_key, pdiffs in results["percentage_differences"].items():
                            for criterion, percent_diff in pdiffs.items():
                                if np.isfinite(percent_diff):
                                    percentage_error_sum += abs(percent_diff)
                    obj4_error = percentage_error_sum

                    return (obj1_performance, obj2_sparsity, obj3_cost, obj4_error)

                except Exception as e:
                    self.update.emit(f"Warning: FRF evaluation failed: {str(e)}")
                    return (1e6, 1e6, 1e6, 1e6)

            toolbox.register("evaluate", evaluate_individual)
            toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=bounds_low, up=bounds_up, eta=20.0)
            toolbox.register("mutate", tools.mutPolynomialBounded, low=bounds_low, up=bounds_up, eta=20.0, indpb=1.0/len(parameter_names))
            toolbox.register("select", tools.selNSGA2)

            self.update.emit("Initializing population...")
            
            # Ensure population size is a multiple of 4 for selTournamentDCD
            if self.pop_size % 4 != 0:
                original_size = self.pop_size
                self.pop_size = (self.pop_size // 4) * 4
                if self.pop_size == 0: self.pop_size = 4
                self.update.emit(f"Warning: Population size must be a multiple of 4. Adjusted from {original_size} to {self.pop_size}.")

            population = toolbox.population(n=self.pop_size)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # This is just to assign the crowding distance to the individuals
            # no actual selection is done
            population = toolbox.select(population, len(population))

            self.update.emit("Starting NSGA-II evolution...")
            for gen in range(1, self.num_generations + 1):
                if self._check_pause_abort():
                    self.update.emit("Optimization aborted.")
                    return

                self.progress.emit(int((gen / self.num_generations) * 100))
                self.update.emit(f"Generation {gen}/{self.num_generations}")

                # Vary the population
                offspring = tools.selTournamentDCD(population, len(population))
                offspring = [toolbox.clone(ind) for ind in offspring]
                
                for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() <= self.cxpb:
                        toolbox.mate(ind1, ind2)
                    toolbox.mutate(ind1)
                    toolbox.mutate(ind2)
                    del ind1.fitness.values, ind2.fitness.values
                
                # Evaluate the individuals with an invalid fitness
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # Select the next generation population
                population = toolbox.select(population + offspring, self.pop_size)

                # Log stats for the first front
                front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
                min_obj1 = min(ind.fitness.values[0] for ind in front)
                min_obj2 = min(ind.fitness.values[1] for ind in front)
                self.update.emit(f"  Front size: {len(front)}, Min Perf: {min_obj1:.4f}, Min Sparsity: {min_obj2:.4f}")


            self.progress.emit(100)
            self.update.emit("Optimization finished. Extracting Pareto front...")

            pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
            
            # Prepare results for emitting
            results_list = []
            for ind in pareto_front:
                results_list.append({
                    'parameters': list(ind),
                    'objectives': ind.fitness.values
                })

            self.finished.emit(results_list, parameter_names)

        except Exception as e:
            error_msg = f"NSGA-II optimization error: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)
        finally:
            if hasattr(creator, "FitnessMulti"):
                delattr(creator, "FitnessMulti")
            if hasattr(creator, "Individual"):
                delattr(creator, "Individual")
