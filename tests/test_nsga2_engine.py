import unittest
from unittest.mock import patch, MagicMock

from devana_lib.optimization.config import (
    OptimizationConfig, PhysicsConfig, SparsityConfig, 
    NSGA2Hyperparameters, TargetWeight, FrequencyRange, DVAParameter
)
from devana_lib.optimization.controller import ExecutionController
from devana_lib.optimization.engines.nsga2 import NSGA2Engine

class TestNSGA2Engine(unittest.TestCase):
    def setUp(self):
        freq = FrequencyRange(0.1, 10.0, 10)
        dva_params = [
            DVAParameter("p1", 0.0, 1.0, False, 0.0, 1.0),
            DVAParameter("p2", 0.0, 1.0, False, 0.0, 1.0)
        ]
        physics = PhysicsConfig([1.0], [TargetWeight([1.0], [1.0])], freq, dva_params)
        sparsity = SparsityConfig(0.1, 0.2, 0.3)
        self.hyper = NSGA2Hyperparameters(pop_size=4, generations=2, cxpb=0.9, mutpb=0.1, eta_c=15.0, eta_m=20.0, indpb=0.1, num_runs=1)
        self.config = OptimizationConfig(physics=physics, sparsity=sparsity, hyperparameters=self.hyper)

    @patch('devana_lib.optimization.engines.nsga2.frf')
    def test_nsga2_engine_runs_successfully(self, mock_frf):
        mock_frf.return_value = {'singular_response': 5.0}

        engine = NSGA2Engine()
        controller = ExecutionController()

        progress_calls = []
        finished_calls = []
        error_calls = []

        def on_progress(run_idx, gen, total_gens, metrics):
            progress_calls.append((run_idx, gen))
        def on_finished(data):
            finished_calls.append(data)
        def on_error(err):
            error_calls.append(err)

        engine.run(
            config=self.config,
            controller=controller,
            on_progress=on_progress,
            on_finished=on_finished,
            on_error=on_error,
            random_seed=42
        )

        self.assertEqual(len(error_calls), 0, f"Engine failed with errors: {error_calls}")
        self.assertEqual(len(finished_calls), 1)
        self.assertEqual(len(progress_calls), self.hyper.generations)

if __name__ == '__main__':
    unittest.main()
