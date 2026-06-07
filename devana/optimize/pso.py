import random
import numpy as np
from devana.optimize.base import Solver

class PSOSolver(Solver):
    """
    Particle Swarm Optimization (PSO) Solver.
    
    Ported from PSOWorker.py, removing PyQt5 dependencies.
    """
    def __init__(self, config, evaluate_fn=None, callback=None):
        super().__init__(config, evaluate_fn, callback)
        
        # PSO specific configuration
        self.w = config.get('w', 0.729)      # Inertia weight
        self.c1 = config.get('c1', 1.49445)   # Cognitive coefficient
        self.c2 = config.get('c2', 1.49445)   # Social coefficient
        self.max_velocity_factor = config.get('max_velocity_factor', 0.1)
        
        # Boundary handling
        self.boundary_handling = config.get('boundary_handling', 'absorbing')

    def solve(self):
        """Execute the PSO optimization."""
        num_particles = self.pop_size
        num_params = self.num_parameters
        
        # Calculate max velocities
        max_velocities = []
        for j in range(num_params):
            low, high = self.parameter_bounds[j]
            max_velocities.append((high - low) * self.max_velocity_factor)
            
        # Initialize swarm
        swarm = []
        for i in range(num_particles):
            position = []
            velocity = []
            for j in range(num_params):
                low, high = self.parameter_bounds[j]
                if j in self.fixed_parameters:
                    pos = self.fixed_parameters[j]
                    vel = 0.0
                else:
                    pos = random.uniform(low, high)
                    max_vel = max_velocities[j]
                    vel = random.uniform(-max_vel, max_vel)
                position.append(pos)
                velocity.append(vel)
            
            fitness = self.evaluate(position)
            
            particle = {
                'position': np.array(position),
                'velocity': np.array(velocity),
                'best_position': np.array(position),
                'best_fitness': fitness,
                'fitness': fitness
            }
            swarm.append(particle)
            
        # Global best
        global_best_idx = min(range(num_particles), key=lambda i: swarm[i]['best_fitness'])
        global_best_position = swarm[global_best_idx]['best_position'].copy()
        global_best_fitness = swarm[global_best_idx]['best_fitness']
        
        metrics = {
            'best_fitness_history': []
        }
        
        for it in range(1, self.num_generations + 1):
            if self.stop_requested:
                break
                
            for i in range(num_particles):
                particle = swarm[i]
                
                # Update velocity
                r1 = random.random()
                r2 = random.random()
                
                cognitive = self.c1 * r1 * (particle['best_position'] - particle['position'])
                social = self.c2 * r2 * (global_best_position - particle['position'])
                
                new_velocity = self.w * particle['velocity'] + cognitive + social
                
                # Clamping and Fixed parameters
                for j in range(num_params):
                    if j in self.fixed_parameters:
                        new_velocity[j] = 0.0
                    else:
                        new_velocity[j] = max(-max_velocities[j], min(max_velocities[j], new_velocity[j]))
                
                particle['velocity'] = new_velocity
                
                # Update position
                new_position = particle['position'] + particle['velocity']
                
                # Boundary handling
                for j in range(num_params):
                    if j in self.fixed_parameters:
                        new_position[j] = self.fixed_parameters[j]
                    else:
                        low, high = self.parameter_bounds[j]
                        if new_position[j] < low or new_position[j] > high:
                            if self.boundary_handling == 'absorbing':
                                new_position[j] = max(low, min(high, new_position[j]))
                                particle['velocity'][j] = 0.0
                            elif self.boundary_handling == 'reflecting':
                                if new_position[j] < low:
                                    new_position[j] = 2*low - new_position[j]
                                    particle['velocity'][j] = -particle['velocity'][j]
                                else:
                                    new_position[j] = 2*high - new_position[j]
                                    particle['velocity'][j] = -particle['velocity'][j]
                                # Re-clamp in case reflection still outside
                                new_position[j] = max(low, min(high, new_position[j]))
                
                particle['position'] = new_position
                
                # Evaluate
                fitness = self.evaluate(particle['position'].tolist())
                particle['fitness'] = fitness
                
                # Update personal best
                if fitness < particle['best_fitness']:
                    particle['best_fitness'] = fitness
                    particle['best_position'] = particle['position'].copy()
                    
                    # Update global best
                    if fitness < global_best_fitness:
                        global_best_fitness = fitness
                        global_best_position = particle['best_position'].copy()
            
            metrics['best_fitness_history'].append(global_best_fitness)
            self._report_progress(it, global_best_fitness, global_best_position.tolist(), metrics)
            
            if global_best_fitness <= self.tolerance:
                break
                
        return {
            'best_individual': global_best_position.tolist(),
            'best_fitness': global_best_fitness,
            'metrics': metrics,
            'parameter_names': self.parameter_names
        }
