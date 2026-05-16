import sys
import numpy as np
import random
import time
import platform
import psutil
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from modules.FRF import frf
from modules.sobol_sensitivity import format_parameter_name

class SAWorker(QThread):
    # Signals: finished(final_results, best_candidate, parameter_names, best_fitness), error(str), update(str)
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
                 sa_initial_temp,      # Initial temperature
                 sa_cooling_rate,      # Cooling factor (e.g., 0.95)
                 sa_num_iterations,    # Maximum number of iterations
                 sa_tol,               # Tolerance to stop (if best fitness is below this value)
                 sa_parameter_data,    # List of tuples: (name, lower bound, upper bound, fixed flag)
                 alpha=0.01,           # Sparsity penalty factor
                 percentage_error_scale=1000.0,
                 track_metrics=True,
                 use_ml_adaptive=False,
                 ml_ucb_c=0.6,
                 ml_accept_target=0.3,
                 use_rl_controller=False,
                 rl_alpha=0.1,
                 rl_gamma=0.9,
                 rl_epsilon=0.2,
                 rl_epsilon_decay=0.95,
                 step_scale=0.1):
        super().__init__()
        self.main_params = main_params
        self.target_values_dict = target_values_dict
        self.weights_dict = weights_dict
        self.omega_start = omega_start
        self.omega_end = omega_end
        self.omega_points = omega_points
        self.sa_initial_temp = sa_initial_temp
        self.sa_cooling_rate = sa_cooling_rate
        self.sa_num_iterations = sa_num_iterations
        self.sa_tol = sa_tol
        self.sa_parameter_data = sa_parameter_data
        self.alpha = alpha
        self.percentage_error_scale = percentage_error_scale if percentage_error_scale is not None else 1000.0
        # Controllers/metrics
        self.track_metrics = bool(track_metrics)
        self.use_ml_adaptive = bool(use_ml_adaptive)
        self.ml_ucb_c = float(ml_ucb_c)
        self.ml_accept_target = float(ml_accept_target)
        self.use_rl_controller = bool(use_rl_controller)
        self.rl_alpha = float(rl_alpha)
        self.rl_gamma = float(rl_gamma)
        self.rl_epsilon = float(rl_epsilon)
        self.rl_epsilon_decay = float(rl_epsilon_decay)
        self.step_scale = float(step_scale)
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
            'best_fitness_per_gen': [],
            'mean_fitness_history': [],
            'std_fitness_history': [],
            'fitness_history': [],
            'evaluation_count': 0,
            'temperature_history': [],
            'cooling_rate_history': [],
            'step_scale_history': [],
            'acceptance_rate_history': [],
            'controller': 'fixed',
            'rates_history': []
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
                self.update.emit(f"DEBUG: SA controller is set to: {self.metrics['controller']}")
                if self.use_ml_adaptive:
                    self.update.emit(f"DEBUG: ML params: UCB c={self.ml_ucb_c:.2f}, accept_target={self.ml_accept_target:.2f}")
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
            # Extract parameter names, bounds, and fixed parameters
            parameter_names = []
            parameter_bounds = []
            fixed_parameters = {}  # key: index, value: fixed value
            for idx, (name, low, high, fixed) in enumerate(self.sa_parameter_data):
                parameter_names.append(name)
                if fixed:
                    parameter_bounds.append((low, low))
                    fixed_parameters[idx] = low
                else:
                    parameter_bounds.append((low, high))
            num_params = len(parameter_bounds)

            # Initialize candidate solution (current state)
            current_candidate = []
            for j in range(num_params):
                low, high = parameter_bounds[j]
                if j in fixed_parameters:
                    current_candidate.append(fixed_parameters[j])
                else:
                    current_candidate.append(random.uniform(low, high))
            current_fitness = self.evaluate_candidate(current_candidate)
            best_candidate = current_candidate[:]
            best_fitness = current_fitness

            # Initialize temperature
            T = self.sa_initial_temp
            initial_temp = self.sa_initial_temp
            cooling_rate = float(self.sa_cooling_rate)
            base_step_scale = float(0.1 if self.step_scale is None else self.step_scale)
            accept_count = 0
            attempt_count = 0

            # ML bandit setup
            if self.use_ml_adaptive:
                deltas = [-0.5, -0.25, 0.0, 0.25, 0.5]
                temp_mults = [0.75, 1.0, 1.25]
                ml_actions = [(dS, dC, tm) for dS in deltas for dC in deltas for tm in temp_mults]
                ml_counts = [0 for _ in ml_actions]
                ml_sums = [0.0 for _ in ml_actions]
                ml_t = 0
                def ml_select(cur_step, cur_cooling, cur_T):
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
                    dS, dC, tm = ml_actions[idx]
                    new_step = max(1e-6, cur_step * (1.0 + dS))
                    new_cooling = min(0.9999, max(1e-6, cur_cooling * (1.0 + dC)))
                    new_T = max(1e-12, cur_T * tm)
                    return idx, new_step, new_cooling, new_T
                def ml_update(idx, reward):
                    ml_counts[idx] += 1
                    ml_sums[idx] += float(reward)

            # RL setup
            if self.use_rl_controller:
                deltas = [-0.5, -0.25, 0.0, 0.25, 0.5]
                temp_mults = [0.75, 1.0, 1.25]
                rl_actions = [(dS, dC, tm) for dS in deltas for dC in deltas for tm in temp_mults]
                rl_q = {0: [0.0 for _ in rl_actions], 1: [0.0 for _ in rl_actions]}
                rl_state = 0
                def rl_select(cur_step, cur_cooling, cur_T):
                    if random.random() < self.rl_epsilon:
                        idx = random.randrange(len(rl_actions))
                    else:
                        values = rl_q[rl_state]
                        idx = int(np.argmax(values)) if values else 0
                    dS, dC, tm = rl_actions[idx]
                    new_step = max(1e-6, cur_step * (1.0 + dS))
                    new_cooling = min(0.9999, max(1e-6, cur_cooling * (1.0 + dC)))
                    new_T = max(1e-12, cur_T * tm)
                    return idx, new_step, new_cooling, new_T
                def rl_update(state, action_idx, reward, next_state):
                    q_old = rl_q[state][action_idx]
                    q_next = max(rl_q[next_state]) if rl_q[next_state] else 0.0
                    rl_q[state][action_idx] = q_old + self.rl_alpha * (reward + self.rl_gamma * q_next - q_old)

            # Simulated Annealing main loop
            for iteration in range(1, self.sa_num_iterations + 1):
                iter_start = time.time()
                try:
                    self.progress.emit(int((iteration / max(1, self.sa_num_iterations)) * 100))
                except Exception:
                    pass
                # Controller adjustments
                if self.use_ml_adaptive:
                    ml_idx, base_step_scale, cooling_rate, T = ml_select(base_step_scale, cooling_rate, T)
                elif self.use_rl_controller:
                    rl_idx, base_step_scale, cooling_rate, T = rl_select(base_step_scale, cooling_rate, T)
                # Generate a new candidate by perturbing the current candidate
                new_candidate = []
                for j in range(num_params):
                    if j in fixed_parameters:
                        new_candidate.append(fixed_parameters[j])
                    else:
                        low, high = parameter_bounds[j]
                        base_scale = (high - low) * base_step_scale
                        perturbation = random.gauss(0, base_scale) * (T / max(1e-12, initial_temp))
                        new_val = current_candidate[j] + perturbation
                        # Ensure new value remains within bounds
                        new_val = max(low, min(new_val, high))
                        new_candidate.append(new_val)
                _t0 = time.time()
                new_fitness = self.evaluate_candidate(new_candidate)
                if self.track_metrics:
                    self.metrics['evaluation_times'].append(time.time() - _t0)
                delta_fitness = new_fitness - current_fitness

                # Accept new candidate if it is better or with a probability if worse
                if delta_fitness < 0:
                    current_candidate = new_candidate
                    current_fitness = new_fitness
                    accept = 1
                else:
                    acceptance_probability = np.exp(-delta_fitness / max(1e-12, T))
                    if random.random() < acceptance_probability:
                        current_candidate = new_candidate
                        current_fitness = new_fitness
                        accept = 1
                    else:
                        accept = 0
                attempt_count += 1
                accept_count += accept

                # Update the best candidate found so far
                if current_fitness < best_fitness:
                    best_candidate = current_candidate[:]
                    best_fitness = current_fitness

                self.update.emit(f"Iteration {iteration}: Current={current_fitness:.6f}, Best={best_fitness:.6f}, T={T:.6f}, step={base_step_scale:.4f}, cool={cooling_rate:.5f}")

                # Update temperature
                T = T * cooling_rate

                # Check convergence criterion
                if best_fitness <= self.sa_tol:
                    self.update.emit(f"[INFO] Convergence reached at iteration {iteration}")
                    break
                # Metrics
                if self.track_metrics:
                    iter_time = time.time() - iter_start
                    self.metrics['generation_times'].append(iter_time)
                    self.metrics['time_per_generation_breakdown'].append({'total': iter_time})
                    self.metrics['best_fitness_per_gen'].append(best_fitness)
                    self.metrics['fitness_history'].append([current_fitness])
                    self.metrics['mean_fitness_history'].append(current_fitness)
                    self.metrics['std_fitness_history'].append(0.0)
                    self.metrics['temperature_history'].append(T)
                    self.metrics['cooling_rate_history'].append(cooling_rate)
                    self.metrics['step_scale_history'].append(base_step_scale)
                    if attempt_count > 0 and iteration % 5 == 0:
                        acc_rate = accept_count / max(1, attempt_count)
                        self.metrics['acceptance_rate_history'].append(acc_rate)
                        accept_count = 0
                        attempt_count = 0
                    self.metrics['rates_history'].append({'iteration': iteration, 'T': T, 'cooling': cooling_rate, 'step': base_step_scale})
                    # Controller reward
                    if self.use_ml_adaptive or self.use_rl_controller:
                        last_best = self.metrics['best_fitness_per_gen'][-2] if len(self.metrics['best_fitness_per_gen']) > 1 else None
                        imp = (last_best - best_fitness) if (last_best is not None and last_best > best_fitness) else 0.0
                        acc_rate = self.metrics['acceptance_rate_history'][-1] if self.metrics['acceptance_rate_history'] else 0.0
                        reward = (imp / max(iter_time, 1e-6)) - abs(acc_rate - self.ml_accept_target)
                        if self.use_ml_adaptive:
                            try:
                                ml_update(ml_idx, reward)
                            except Exception:
                                pass
                            self.metrics.setdefault('ml_controller_history', []).append({'iteration': iteration, 'T': T, 'cooling': cooling_rate, 'step': base_step_scale, 'best_fitness': best_fitness, 'reward': reward})
                        elif self.use_rl_controller:
                            try:
                                next_state = 1 if imp > 0 else 0
                                rl_update(rl_state, rl_idx, reward, next_state)
                                rl_state = next_state
                                self.rl_epsilon *= self.rl_epsilon_decay
                            except Exception:
                                pass
                            self.metrics.setdefault('rl_controller_history', []).append({'iteration': iteration, 'T': T, 'cooling': cooling_rate, 'step': base_step_scale, 'best_fitness': best_fitness, 'reward': reward, 'epsilon': self.rl_epsilon})

            # Final evaluation using best candidate
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

    def evaluate_candidate(self, candidate):
        """
        Evaluate the fitness of a candidate solution using the FRF function.
        The fitness is defined as the absolute difference between the singular response
        and 1 plus a sparsity penalty.
        """
        try:
            results = frf(
                main_system_parameters=self.main_params,
                dva_parameters=tuple(candidate),
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
                sparsity_penalty = self.alpha * sum(abs(x) for x in candidate)
                percentage_error_sum = 0.0
                if "percentage_differences" in results:
                    for mass_key, pdiffs in results["percentage_differences"].items():
                        for criterion, percent_diff in pdiffs.items():
                            percentage_error_sum += abs(percent_diff)
                return primary_objective + sparsity_penalty + percentage_error_sum / self.percentage_error_scale
        except Exception as e:
            return 1e6

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
            self.update.emit("SA optimization timed out. The operation was taking too long.")
        except Exception:
            pass
