import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from devana_lib.optimization.config import (
    OptimizationConfig, PhysicsConfig, SparsityConfig, 
    NSGA2Hyperparameters, TargetWeight, FrequencyRange, DVAParameter
)
from devana_lib.optimization.controller import ExecutionController
from devana_lib.optimization.engines.nsga2 import NSGA2Engine

class NSGA2Worker(QThread):
    progress = pyqtSignal(int, int, int, dict)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, main_params, dva_params, target_values_weights, omega_start, omega_end, omega_points,
                 pop_size, generations, cxpb, mutpb, eta_c, eta_m, indpb, sparsity_tau, sparsity_alpha, sparsity_beta,
                 num_runs=1, random_seed=None, convergence_epsilon=0.001, convergence_window=50, convergence_min_gen=500,
                 hv_ref_point=None, parent=None):
        super().__init__(parent)
        
        self.main_params = main_params
        self.dva_params = dva_params
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
        self.indpb = indpb
        self.sparsity_tau = sparsity_tau
        self.sparsity_alpha = sparsity_alpha
        self.sparsity_beta = sparsity_beta
        self.num_runs = num_runs
        self.random_seed = random_seed
        self.convergence_epsilon = convergence_epsilon
        self.convergence_window = convergence_window
        self.convergence_min_gen = convergence_min_gen
        self.hv_ref_point = hv_ref_point
        
        self.controller = ExecutionController()

    def stop(self):
        self.controller.abort()

    def pause(self):
        self.controller.pause()

    def resume(self):
        self.controller.resume()

    def run(self):
        # Build Configs in the worker thread
        dva_config = [
            DVAParameter(name=p[0], low=p[1], high=p[2], is_fixed=p[3], fixed_value=p[4], cost_coeff=p[5]) 
            for p in self.dva_params
        ]
        
        target_weights = [
            TargetWeight(values=tvw[0], weights=tvw[1]) for tvw in self.target_values_weights
        ]
        
        physics = PhysicsConfig(
            main_system_parameters=self.main_params,
            target_values_weights=target_weights,
            frequency_range=FrequencyRange(self.omega_start, self.omega_end, self.omega_points),
            dva_parameters=dva_config
        )
        
        sparsity = SparsityConfig(tau=self.sparsity_tau, alpha=self.sparsity_alpha, beta=self.sparsity_beta)
        
        hyper = NSGA2Hyperparameters(
            pop_size=self.pop_size, generations=self.generations, cxpb=self.cxpb, mutpb=self.mutpb,
            eta_c=self.eta_c, eta_m=self.eta_m, indpb=self.indpb, num_runs=self.num_runs
        )
        
        config = OptimizationConfig(physics=physics, sparsity=sparsity, hyperparameters=hyper)
        engine = NSGA2Engine()

        def on_progress(run_idx, gen, total_gens, metrics):
            self.progress.emit(run_idx, gen, total_gens, metrics)
            
        def on_finished(data):
            if not self.controller.should_abort():
                self.finished.emit(data)
                
        def on_error(err):
            self.error.emit(err)

        engine.run(
            config=config,
            controller=self.controller,
            on_progress=on_progress,
            on_finished=on_finished,
            on_error=on_error,
            random_seed=self.random_seed,
            convergence_epsilon=self.convergence_epsilon,
            convergence_window=self.convergence_window,
            convergence_min_gen=self.convergence_min_gen,
            hv_ref_point=self.hv_ref_point
        )
