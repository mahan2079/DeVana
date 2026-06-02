import unittest
import numpy as np
import sys
import os
import time
from PyQt5.QtCore import QCoreApplication

# Add 'codes' directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../codes')))

try:
    from workers.NSGA2Worker import NSGA2Worker
    NSGA2_AVAILABLE = True
except ImportError:
    NSGA2_AVAILABLE = False

@unittest.skipIf(not NSGA2_AVAILABLE, "DEAP not installed")
class TestNSGA2Worker(unittest.TestCase):
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
        # (name, low, high, is_fixed, fixed_value, cost_coeff)
        self.dva_bounds = [
            (f"param_{i}", 0.01, 1.0, False, 0.0, 1.0) for i in range(48)
        ]
        
        # (target_values_dict, weights_dict)
        targets = {"peak_value_1": 1.0}
        weights = {"peak_value_1": 1.0}
        target_dict = {f"mass_{i}": targets for i in range(1, 6)}
        weight_dict = {f"mass_{i}": weights for i in range(1, 6)}
        self.target_values_weights = (target_dict, weight_dict)

    def test_nsga2_worker_run(self):
        """Test if NSGA2Worker can run and emit progress"""
        worker = NSGA2Worker(
            main_params=self.main_params, 
            dva_params=self.dva_bounds,
            target_values_weights=self.target_values_weights,
            omega_start=0, 
            omega_end=200, 
            omega_points=20,
            pop_size=10,
            generations=2,
            cxpb=0.7,
            mutpb=0.2,
            eta_c=20,
            eta_m=20,
            indpb=0.05,
            sparsity_tau=0.1,
            sparsity_alpha=0.01,
            sparsity_beta=0.01
        )
        
        progress_calls = []
        def capture_progress(run_idx, current_gen, total_gens, metrics):
            progress_calls.append((run_idx, current_gen, total_gens, metrics))
            
        worker.progress.connect(capture_progress)
        
        worker.start()
        timeout = time.time() + 30
        while worker.isRunning() and time.time() < timeout:
            self.app.processEvents()
            time.sleep(0.1)
        
        if worker.isRunning():
            worker.stop()
            worker.wait()
            self.fail("NSGA2Worker timed out")

        self.assertTrue(len(progress_calls) > 0, "NSGA2Worker did not emit any progress")
        self.assertEqual(progress_calls[0][1], 1, "First generation should be 1")

if __name__ == '__main__':
    unittest.main()
