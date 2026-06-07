import unittest
import numpy as np
import sys
import os
import time
from PyQt5.QtCore import QCoreApplication

# Add 'codes' directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../codes')))

from workers.PINNSolver import PINNSolver
from workers.GAWorker import GAWorker
from workers.NeuralSurrogate import TORCH_AVAILABLE

class TestPINNSolver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not QCoreApplication.instance():
            cls.app = QCoreApplication(sys.argv)
        else:
            cls.app = QCoreApplication.instance()

    def setUp(self):
        self.main_params = [
            1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75, 0.75,
            0.05, 0.95, 100.0, 100.0, 100.0, 0.01
        ]
        self.dva_parameter_order = [f"param_{i}" for i in range(48)]
        self.dva_bounds = [(name, 0.01, 1.0, False) for name in self.dva_parameter_order]
        self.targets = {f"mass_{i}": {"peak_value_1": 1.0} for i in range(1, 6)}
        self.weights = {f"mass_{i}": {"peak_value_1": 1.0} for i in range(1, 6)}

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not installed")
    def test_pinn_training_and_acceleration(self):
        """Test the PINNSolver speedup and basic logic"""
        solver = PINNSolver(param_dim=48)
        
        # 1. Simulate a study session
        X = np.random.rand(20, 48)
        omegas = np.random.rand(20)
        y = np.random.rand(20)
        
        start_train = time.time()
        for _ in range(5):
            solver.train_step(X, omegas, y)
        train_duration = time.time() - start_train
        
        # 2. Test prediction (The Acceleration)
        X_test = np.random.rand(48)
        omega_range = np.linspace(0, 100, 500)
        
        start_pred = time.time()
        preds = solver.predict(X_test, omega_range)
        pred_duration = time.time() - start_pred
        
        self.assertEqual(len(preds), 500)
        print(f"\nPINN Performance: Prediction for 500 freqs took {pred_duration:.6f}s")
        self.assertLess(pred_duration, 0.1, "PINN prediction is too slow")

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not installed")
    def test_ga_worker_with_pinn_bypass(self):
        """Test if GAWorker correctly switches to PINN solver after threshold"""
        worker = GAWorker(
            self.main_params, 
            self.targets, self.weights,
            0, 200, 20, 
            ga_pop_size=20, # More individuals
            ga_num_generations=2, # More generations
            ga_cxpb=0.9, 
            ga_mutpb=0.5, # Much higher mutation to force invalid_ind
            ga_tol=0.01,
            ga_parameter_data=self.dva_bounds,
            use_pinn_solver=True # Enable PINN Solver!
        )
        
        # Manually cross the 50-point threshold
        # We also need to set use_surrogate=True for some internal normalization to be ready
        worker.use_surrogate = True 
        X = [list(np.random.rand(48)) for _ in range(60)]
        y = [float(val) for val in np.random.rand(60)]
        worker._update_and_train_surrogate(X, y) 
        
        results = []
        worker.update.connect(lambda msg: results.append(msg))
        
        # Run
        worker.run()
        
        # Log results for debugging
        # print("\nMessages received from worker:", results)
        
        # Check if "PINN Solver:" was used
        self.assertTrue(any("PINN Solver:" in r for r in results), 
                        "GAWorker did not use PINN-accelerated evaluation")

if __name__ == '__main__':
    unittest.main()
