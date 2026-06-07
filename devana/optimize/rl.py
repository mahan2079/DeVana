import numpy as np
import random
import pickle
import os
from .base import Solver
from ..sensitivity.sobol import perform_sobol_analysis

class RLSolver(Solver):
    """
    Reinforcement Learning Solver for DVA parameter optimization.
    
    This implementation uses a Deep Deterministic Policy Gradient (DDPG)-inspired 
    approach adapted for continuous parameter spaces.
    """
    def __init__(self, config, evaluate_fn=None, callback=None):
        super().__init__(config, evaluate_fn, callback)
        
        # RL Hyperparameters
        self.num_episodes = config.get('num_episodes', 100)
        self.max_steps = config.get('max_steps', 50)
        self.alpha = config.get('alpha', 0.01)        # Learning rate
        self.gamma = config.get('gamma', 0.95)        # Discount factor
        self.epsilon = config.get('epsilon', 1.0)      # Initial exploration rate
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay_type = config.get('epsilon_decay_type', 'exponential')
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        
        # Additional RL parameters
        self.replay_buffer_size = config.get('replay_buffer_size', 10000)
        self.batch_size = config.get('batch_size', 32)
        self.noise_std = config.get('noise_std', 0.1)
        
        # Epsilon decay parameters
        self.linear_decay_step = config.get('linear_decay_step', None)
        self.inverse_decay_coefficient = config.get('inverse_decay_coefficient', 1.0)
        self.step_interval = config.get('step_interval', 10)
        self.step_decay_amount = config.get('step_decay_amount', self.epsilon_decay)
        self.cosine_decay_amplitude = config.get('cosine_decay_amplitude', 1.0)
        
        # Sobol settings
        self.sobol_settings = config.get('sobol_settings', {})
        
        # Initialize policy network (simple linear policy)
        self.policy_weights = np.random.randn(self.num_parameters) * 0.1
        self.policy_bias = np.zeros(self.num_parameters)
        
        # Experience replay buffer
        self.experience_buffer = []
        
        # Experience saving/loading
        self.experience_save_path = config.get('experience_save_path', None)
        if config.get('load_existing_experience', False) and self.experience_save_path:
            self._load_experience()

    def _load_experience(self):
        """Load existing experience from file"""
        if os.path.exists(self.experience_save_path):
            try:
                with open(self.experience_save_path, 'rb') as f:
                    data = pickle.load(f)
                    self.experience_buffer = data.get('experience_buffer', [])
                    self.policy_weights = data.get('policy_weights', self.policy_weights)
                    self.policy_bias = data.get('policy_bias', self.policy_bias)
            except Exception:
                pass

    def _save_experience(self):
        """Save experience to file"""
        if self.experience_save_path:
            try:
                data = {
                    'experience_buffer': self.experience_buffer,
                    'policy_weights': self.policy_weights,
                    'policy_bias': self.policy_bias
                }
                with open(self.experience_save_path, 'wb') as f:
                    pickle.dump(data, f)
            except Exception:
                pass

    def generate_parameters(self, add_noise=True):
        """Generate DVA parameters using current policy."""
        raw_params = self.policy_weights + self.policy_bias
        
        if add_noise and self.epsilon > 0:
            noise = np.random.normal(0, self.noise_std * self.epsilon, self.num_parameters)
            raw_params += noise
        
        bounded_params = []
        for i, (low, high) in enumerate(self.parameter_bounds):
            if i in self.fixed_parameters:
                bounded_params.append(self.fixed_parameters[i])
            else:
                normalized = 1 / (1 + np.exp(-raw_params[i]))  # Sigmoid activation
                scaled = low + normalized * (high - low)
                bounded_params.append(scaled)
        
        return bounded_params

    def update_policy(self, experiences):
        """Update the policy based on collected experiences."""
        if len(experiences) < self.batch_size:
            return
        
        batch = random.sample(experiences, min(self.batch_size, len(experiences)))
        
        policy_gradient_weights = np.zeros_like(self.policy_weights)
        policy_gradient_bias = np.zeros_like(self.policy_bias)
        
        for params, fitness in batch:
            advantage = -fitness  # Lower fitness is better
            
            for i in range(self.num_parameters):
                if i not in self.fixed_parameters:
                    policy_gradient_weights[i] += advantage * params[i] * self.alpha
                    policy_gradient_bias[i] += advantage * self.alpha
        
        self.policy_weights += policy_gradient_weights / len(batch)
        self.policy_bias += policy_gradient_bias / len(batch)
        
        # Regularization
        self.policy_weights *= 0.999
        self.policy_bias *= 0.999

    def update_epsilon(self, episode):
        """Update exploration rate."""
        if self.epsilon_decay_type == 'exponential':
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        elif self.epsilon_decay_type == 'linear':
            if self.linear_decay_step is None:
                step = (self.epsilon - self.epsilon_min) / self.num_episodes
            else:
                step = self.linear_decay_step
            self.epsilon = max(self.epsilon_min, self.epsilon - step)
        elif self.epsilon_decay_type == 'inverse':
            self.epsilon = self.epsilon_min + (self.epsilon - self.epsilon_min) / (1 + self.inverse_decay_coefficient * episode)
        elif self.epsilon_decay_type == 'step':
            if episode % self.step_interval == 0:
                self.epsilon = max(self.epsilon_min, self.epsilon - self.step_decay_amount)
        elif self.epsilon_decay_type == 'cosine':
            cosine_term = (1 + np.cos(np.pi * episode / self.num_episodes)) / 2
            self.epsilon = self.epsilon_min + (self.epsilon - self.epsilon_min) * (1 + self.cosine_decay_amplitude * cosine_term) / 2

    def solve(self):
        """Main RL training loop."""
        # Optional Sobol Analysis for parameter hierarchy
        if self.sobol_settings.get("use_sobol", False):
            sample_size = self.sobol_settings.get("sample_size", 32)
            sobol_results, _ = perform_sobol_analysis(
                main_system_parameters=self.config.get('main_params'),
                dva_parameters_bounds=self.parameter_data,
                dva_parameter_order=self.parameter_names,
                omega_start=self.config.get('omega_start'),
                omega_end=self.config.get('omega_end'),
                omega_points=self.config.get('omega_points'),
                num_samples_list=[sample_size],
                target_values_dict=self.config.get('target_values_dict'),
                weights_dict=self.config.get('weights_dict'),
                visualize=False,
                n_jobs=1
            )
            
            # Reorder parameters by sensitivity
            last_ST = np.array(sobol_results['ST'][-1])
            sorted_indices = np.argsort(last_ST)[::-1]
            ranking = [self.parameter_names[i] for i in sorted_indices]
            
            # Rebuild parameter mappings
            new_parameter_data = []
            for param_name in ranking:
                for item in self.parameter_data:
                    if item[0] == param_name:
                        new_parameter_data.append(item)
                        break
            self.parameter_data = new_parameter_data
            
            # Re-initialize based on new order
            self.__init__(self.config, self.evaluate_fn, self.callback)

        best_fitness = float('inf')
        best_solution = None
        
        for episode in range(1, self.num_episodes + 1):
            if self.stop_requested:
                break
                
            episode_best_fitness = float('inf')
            episode_experiences = []

            for step in range(self.max_steps):
                if self.stop_requested:
                    break
                    
                params = self.generate_parameters(add_noise=True)
                fitness = self.evaluate(params)
                
                episode_experiences.append((params, fitness))
                
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = params.copy()
                
                if fitness < episode_best_fitness:
                    episode_best_fitness = fitness

            # Update replay buffer and policy
            self.experience_buffer.extend(episode_experiences)
            if len(self.experience_buffer) > self.replay_buffer_size:
                self.experience_buffer = self.experience_buffer[-self.replay_buffer_size:]
            
            self.update_policy(self.experience_buffer)
            self.update_epsilon(episode)
            
            self._report_progress(episode, best_fitness, best_solution, {
                'epsilon': self.epsilon,
                'episode_best': episode_best_fitness
            })

        self._save_experience()
        
        return {
            'best_fitness': best_fitness,
            'best_solution': best_solution,
            'parameter_names': self.parameter_names
        }
