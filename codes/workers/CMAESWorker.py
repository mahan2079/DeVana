import sys
import numpy as np
import random
import time
import platform
import psutil
import cma  # Make sure the cma package is installed
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from modules.FRF import frf
from modules.sobol_sensitivity import format_parameter_name

class CMAESWorker(QThread):
    # Emits: finished(final_results, best_candidate, parameter_names, best_fitness)
    finished = pyqtSignal(dict, list, list, float)
    error = pyqtSignal(str)
    update = pyqtSignal(str)
    progress = pyqtSignal(int)
    benchmark_data = pyqtSignal(dict)
    generation_metrics = pyqtSignal(dict)

    def __init__(self, 
                 main_params,
                 target_values_dict,
                 weights_dict,
                 omega_start,
                 omega_end,
                 omega_points,
                 cma_initial_sigma,   # Scalar: initial standard deviation for search
                 cma_max_iter,        # Maximum number of iterations/generations
                 cma_tol,             # Tolerance to stop the search
                 cma_parameter_data,  # List of tuples: (name, lower bound, upper bound, fixed flag)
                 alpha=0.01,          # Sparsity penalty factor
                 percentage_error_scale=1000.0,
                 track_metrics=True,
                 use_ml_adaptive=False,
                 ml_ucb_c=0.6,
                 use_rl_controller=False,
                 rl_alpha=0.1,
                 rl_gamma=0.9,
                 rl_epsilon=0.2,
                 rl_epsilon_decay=0.95,
                 sigma_scale=1.0):         # base sigma scale multiplier
        super().__init__()
        self.main_params = main_params
        self.target_values_dict = target_values_dict
        self.weights_dict = weights_dict
        self.omega_start = omega_start
        self.omega_end = omega_end
        self.omega_points = omega_points
        self.cma_initial_sigma = cma_initial_sigma
        self.cma_max_iter = cma_max_iter
        self.cma_tol = cma_tol
        self.cma_parameter_data = cma_parameter_data
        self.alpha = alpha
        self.percentage_error_scale = percentage_error_scale if percentage_error_scale is not None else 1000.0
        # Controllers and metrics
        self.track_metrics = bool(track_metrics)
        self.use_ml_adaptive = bool(use_ml_adaptive)
        self.ml_ucb_c = float(ml_ucb_c)
        self.use_rl_controller = bool(use_rl_controller)
        self.rl_alpha = float(rl_alpha)
        self.rl_gamma = float(rl_gamma)
        self.rl_epsilon = float(rl_epsilon)
        self.rl_epsilon_decay = float(rl_epsilon_decay)
        self.sigma_scale = float(sigma_scale)

        self.metrics = {
            'start_time': None,
            'end_time': None,
            'total_duration': None,
            'cpu_usage': [],
            'memory_usage': [],
            'cpu_per_core': [],
            'memory_details': [],
            'thread_count': [],
            'system_info': {},
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'generation_times': [],
            'time_per_generation_breakdown': [],
            'evaluation_times': [],
            'fitness_history': [],
            'mean_fitness_history': [],
            'std_fitness_history': [],
            'best_fitness_per_gen': [],
            'best_individual_per_gen': [],
            'convergence_rate': [],
            'rates_history': [],  # sigma, popsize
            'controller': 'fixed'
        }
        self._metrics_timer = QTimer(); self._metrics_timer.setSingleShot(False)
        self._metrics_interval = 500
        self._watchdog = QTimer(); self._watchdog.setSingleShot(True); self._watchdog.timeout.connect(self._handle_timeout)

    def run(self):
        try:
            # Controller descriptor
            try:
                if self.use_rl_controller:
                    self.metrics['controller'] = 'rl'
                elif self.use_ml_adaptive:
                    self.metrics['controller'] = 'ml_bandit'
                else:
                    self.metrics['controller'] = 'fixed'
                self.update.emit(f"DEBUG: CMA-ES controller is set to: {self.metrics['controller']}")
                if self.use_ml_adaptive:
                    self.update.emit(f"DEBUG: ML params: UCB c={self.ml_ucb_c:.2f}")
                if self.use_rl_controller:
                    self.update.emit(f"DEBUG: RL params: alpha={self.rl_alpha:.3f}, gamma={self.rl_gamma:.3f}, epsilon={self.rl_epsilon:.3f}, decay={self.rl_epsilon_decay:.3f}")
            except Exception:
                pass
            if self.track_metrics:
                self._start_metrics_tracking()
            try:
                self._watchdog.start(600000)
            except Exception:
                pass
            # Extract parameter names, bounds, and fixed parameters.
            parameter_names = []
            parameter_bounds = []
            fixed_parameters = {}  # key: index, value: fixed value
            for idx, (name, low, high, fixed) in enumerate(self.cma_parameter_data):
                parameter_names.append(name)
                if fixed:
                    parameter_bounds.append((low, low))
                    fixed_parameters[idx] = low
                else:
                    parameter_bounds.append((low, high))
            num_params = len(parameter_bounds)

            # Build initial candidate x0.
            x0 = []
            for j in range(num_params):
                low, high = parameter_bounds[j]
                if j in fixed_parameters:
                    x0.append(fixed_parameters[j])
                else:
                    x0.append(random.uniform(low, high))

            # Use provided cma_initial_sigma as the initial standard deviation (scaled).
            sigma0 = self.cma_initial_sigma * max(1e-6, self.sigma_scale)

            # Build lower and upper bound arrays.
            lower_bounds = [lb for lb, ub in parameter_bounds]
            upper_bounds = [ub for lb, ub in parameter_bounds]

            # Define the objective function.
            def objective(x):
                # For dimensions with fixed values, force them.
                for idx, fixed_val in fixed_parameters.items():
                    x[idx] = fixed_val
                try:
                    results = frf(
                        main_system_parameters=self.main_params,
                        dva_parameters=tuple(x),
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
                        plot_figure=False,
                        show_peaks=False,
                        show_slopes=False
                    )
                    singular_response = results.get('singular_response', None)
                    if singular_response is None or not np.isfinite(singular_response):
                        return 1e6
                    else:
                        primary_objective = abs(singular_response - 1)
                        sparsity_penalty = self.alpha * sum(abs(xi) for xi in x)
                        # Percentage error component to match GA
                        percentage_error_sum = 0.0
                        if "percentage_differences" in results:
                            for mass_key, pdiffs in results["percentage_differences"].items():
                                for criterion, percent_diff in pdiffs.items():
                                    percentage_error_sum += abs(percent_diff)
                        return primary_objective + sparsity_penalty + percentage_error_sum / self.percentage_error_scale
                except Exception as e:
                    return 1e6

            # Set up options for CMA-ES, including bounds.
            options = {
                'bounds': [lower_bounds, upper_bounds],
                'maxiter': self.cma_max_iter,
                'verb_disp': 0,  # We handle our own logging
                'tolx': self.cma_tol
            }
            es = cma.CMAEvolutionStrategy(x0, sigma0, options)

            iter_count = 0
            best_fitness = float('inf')
            best_candidate = None

            # CMA-ES main loop.
            # Optional ML/RL controller setup to modulate sigma
            if self.use_ml_adaptive:
                deltas = [-0.5, -0.25, 0.0, 0.25, 0.5]
                ml_actions = [d for d in deltas]
                ml_counts = [0 for _ in ml_actions]
                ml_sums = [0.0 for _ in ml_actions]
                ml_t = 0
                def ml_select(cur_sigma):
                    nonlocal ml_t
                    ml_t += 1
                    scores = []
                    for i in range(len(ml_actions)):
                        if ml_counts[i] == 0:
                            scores.append((float('inf'), i))
                        else:
                            avg = ml_sums[i] / ml_counts[i]
                            bonus = self.ml_ucb_c * np.sqrt(np.log(max(ml_t, 1)) / ml_counts[i])
                            scores.append((avg + bonus, i))
                    scores.sort(key=lambda t: t[0], reverse=True)
                    _, idx = scores[0]
                    d = ml_actions[idx]
                    new_sigma = max(1e-8, cur_sigma * (1.0 + d))
                    return idx, new_sigma
                def ml_update(idx, reward):
                    ml_counts[idx] += 1
                    ml_sums[idx] += float(reward)
            if self.use_rl_controller:
                deltas = [-0.5, -0.25, 0.0, 0.25, 0.5]
                rl_actions = [d for d in deltas]
                rl_q = {0: [0.0 for _ in rl_actions], 1: [0.0 for _ in rl_actions]}
                rl_state = 0
                def rl_select(cur_sigma):
                    if random.random() < self.rl_epsilon:
                        idx = random.randrange(len(rl_actions))
                    else:
                        values = rl_q[rl_state]
                        idx = int(np.argmax(values)) if values else 0
                    d = rl_actions[idx]
                    new_sigma = max(1e-8, cur_sigma * (1.0 + d))
                    return idx, new_sigma
                def rl_update(state, action_idx, reward, next_state):
                    q_old = rl_q[state][action_idx]
                    q_next = max(rl_q[next_state]) if rl_q[next_state] else 0.0
                    rl_q[state][action_idx] = q_old + self.rl_alpha * (reward + self.rl_gamma * q_next - q_old)

            while not es.stop():
                iter_count += 1
                iter_start = time.time()
                try:
                    self.progress.emit(int(min(100, (iter_count / max(1, self.cma_max_iter)) * 100)))
                except Exception:
                    pass
                # Controller: adjust sigma if possible
                if hasattr(es, 'sigma'):
                    try:
                        cur_sigma = float(es.sigma)
                        if self.use_ml_adaptive:
                            ml_idx, new_sigma = ml_select(cur_sigma)
                            es.sigma = new_sigma
                        elif self.use_rl_controller:
                            rl_idx, new_sigma = rl_select(cur_sigma)
                            es.sigma = new_sigma
                    except Exception:
                        pass
                solutions = es.ask()
                eval_t0 = time.time()
                fitnesses = [objective(x) for x in solutions]
                eval_time = time.time() - eval_t0
                es.tell(solutions, fitnesses)
                current_best = min(fitnesses)
                if current_best < best_fitness:
                    best_fitness = current_best
                    best_candidate = solutions[fitnesses.index(current_best)]
                self.update.emit(f"Iteration {iter_count}: Best fitness = {best_fitness:.6f}")
                if best_fitness <= self.cma_tol:
                    self.update.emit(f"[INFO] Convergence reached at iteration {iter_count}")
                    break

                # Metrics per generation
                if self.track_metrics:
                    gen_time = time.time() - iter_start
                    self.metrics['generation_times'].append(gen_time)
                    self.metrics['time_per_generation_breakdown'].append({'total': gen_time, 'evaluation': eval_time})
                    self.metrics['evaluation_times'].append(eval_time)
                    self.metrics['fitness_history'].append(fitnesses[:])
                    self.metrics['mean_fitness_history'].append(float(np.mean(fitnesses)))
                    self.metrics['std_fitness_history'].append(float(np.std(fitnesses)))
                    self.metrics['best_fitness_per_gen'].append(best_fitness)
                    if hasattr(es, 'sigma'):
                        try:
                            self.metrics['rates_history'].append({'iteration': iter_count, 'sigma': float(es.sigma), 'popsize': getattr(es, 'popsize', None)})
                        except Exception:
                            self.metrics['rates_history'].append({'iteration': iter_count, 'sigma': None, 'popsize': getattr(es, 'popsize', None)})
                    # Controller reward
                    if self.use_ml_adaptive or self.use_rl_controller:
                        last_best = self.metrics['best_fitness_per_gen'][-2] if len(self.metrics['best_fitness_per_gen']) > 1 else None
                        imp = (last_best - best_fitness) if (last_best is not None and last_best > best_fitness) else 0.0
                        reward = (imp / max(gen_time, 1e-6))
                        if self.use_ml_adaptive:
                            try:
                                ml_update(ml_idx, reward)
                            except Exception:
                                pass
                            self.metrics.setdefault('ml_controller_history', []).append({'iteration': iter_count, 'best_fitness': best_fitness, 'reward': reward})
                        elif self.use_rl_controller:
                            try:
                                next_state = 1 if imp > 0 else 0
                                rl_update(rl_state, rl_idx, reward, next_state)
                                rl_state = next_state
                                self.rl_epsilon *= self.rl_epsilon_decay
                            except Exception:
                                pass
                            self.metrics.setdefault('rl_controller_history', []).append({'iteration': iter_count, 'best_fitness': best_fitness, 'reward': reward, 'epsilon': self.rl_epsilon})

            if best_candidate is None:
                best_candidate = es.result.xbest

            # Final evaluation using the best candidate.
            try:
                final_results = frf(
                    main_system_parameters=self.main_params,
                    dva_parameters=tuple(best_candidate),
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
                    plot_figure=False,
                    show_peaks=False,
                    show_slopes=False
                )
            except Exception as e:
                final_results = {"Error": str(e)}

            if self.track_metrics:
                self._stop_metrics_tracking()
                if not self.metrics.get('system_info'):
                    self.metrics['system_info'] = self._get_system_info()
                self.metrics['total_duration'] = (self.metrics['end_time'] - self.metrics['start_time']) if (self.metrics.get('end_time') and self.metrics.get('start_time')) else None
                final_results['benchmark_metrics'] = self.metrics
                try:
                    self.benchmark_data.emit(self.metrics)
                except Exception:
                    pass
            self.finished.emit(final_results, best_candidate, parameter_names, best_fitness)

        except Exception as e:
            self.error.emit(str(e))
        finally:
            try:
                if self._watchdog.isActive():
                    self._watchdog.stop()
            except Exception:
                pass

    # Metrics helpers
    def _get_system_info(self):
        try:
            system_info = {
                'platform': platform.system(),
                'platform_release': platform.release(),
                'platform_version': platform.version(),
                'architecture': platform.machine(),
                'processor': platform.processor(),
                'physical_cores': psutil.cpu_count(logical=False),
                'total_cores': psutil.cpu_count(logical=True),
                'total_memory': round(psutil.virtual_memory().total / (1024.0 ** 3), 2),
                'python_version': platform.python_version(),
            }
            try:
                cpu_freq = psutil.cpu_freq()
                if cpu_freq:
                    system_info['cpu_max_freq'] = cpu_freq.max
                    system_info['cpu_min_freq'] = cpu_freq.min
                    system_info['cpu_current_freq'] = cpu_freq.current
            except Exception:
                pass
            return system_info
        except Exception as e:
            try:
                self.update.emit(f"Warning: Could not collect complete system info: {str(e)}")
            except Exception:
                pass
            return {'error': str(e)}

    def _update_resource_metrics(self):
        if not self.track_metrics:
            return
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            process = psutil.Process()
            mem = process.memory_info()
            memory_usage_mb = mem.rss / (1024 * 1024)
            self.metrics['cpu_usage'].append(cpu_percent)
            self.metrics['memory_usage'].append(memory_usage_mb)
            per_core = psutil.cpu_percent(interval=None, percpu=True)
            self.metrics['cpu_per_core'].append(per_core)
            mem_details = {
                'rss': mem.rss / (1024 * 1024),
                'vms': mem.vms / (1024 * 1024),
                'shared': getattr(mem, 'shared', 0) / (1024 * 1024),
                'system_total': psutil.virtual_memory().total / (1024 * 1024),
                'system_available': psutil.virtual_memory().available / (1024 * 1024),
                'system_percent': psutil.virtual_memory().percent,
            }
            self.metrics['memory_details'].append(mem_details)
            self.metrics['thread_count'].append(process.num_threads())
            current_metrics = {
                'cpu': cpu_percent,
                'cpu_per_core': per_core,
                'memory': memory_usage_mb,
                'memory_details': mem_details,
                'thread_count': process.num_threads(),
                'time': time.time() - self.metrics['start_time'] if self.metrics.get('start_time') else 0
            }
            self.generation_metrics.emit(current_metrics)
        except Exception as e:
            try:
                self.update.emit(f"Warning: Failed to update resource metrics: {str(e)}")
            except Exception:
                pass

    def _start_metrics_tracking(self):
        if not self.track_metrics:
            return
        self.metrics['start_time'] = time.time()
        if not self.metrics.get('system_info'):
            self.metrics['system_info'] = self._get_system_info()
        self._metrics_timer.timeout.connect(self._update_resource_metrics)
        self._metrics_timer.start(self._metrics_interval)
        try:
            self.update.emit(f"Started metrics tracking with interval: {self._metrics_interval}ms")
        except Exception:
            pass

    def _stop_metrics_tracking(self):
        if not self.track_metrics:
            return
        try:
            self._metrics_timer.stop()
        except Exception:
            pass
        self.metrics['end_time'] = time.time()
        if self.metrics.get('start_time'):
            self.metrics['total_duration'] = self.metrics['end_time'] - self.metrics['start_time']
        try:
            if not self.metrics.get('cpu_usage'):
                self._update_resource_metrics()
        except Exception:
            pass

    def _handle_timeout(self):
        try:
            self.update.emit("CMA-ES optimization timed out. The operation was taking too long.")
        except Exception:
            pass
