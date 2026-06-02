import unittest
import numpy as np
import sys
import os
import time
from PyQt5.QtCore import QCoreApplication

# Add 'codes' directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../codes')))

try:
    from workers.MOGAWorker import MOGAWorker
    MOGA_AVAILABLE = True
except ImportError:
    MOGA_AVAILABLE = False

@unittest.skipIf(not MOGA_AVAILABLE, "DEAP not installed")
class TestMOGAWorker(unittest.TestCase):
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
        
        # [[targets_mass1, weights_mass1], ...]
        targets = {"peak_value_1": 1.0}
        weights = {"peak_value_1": 1.0}
        self.target_values_weights = [[targets, weights] for _ in range(5)]

    def test_moga_worker_run(self):
        """Test if MOGAWorker can run and emit progress"""
        worker = MOGAWorker(
            self.main_params, 
            self.dva_bounds,
            self.target_values_weights,
            0, 200, 20,
            pop_size=10,
            generations=2,
            cxpb=0.7,
            mutpb=0.2,
            eta_c=20.0,
            eta_m=20.0,
            indpb=0.05,
            sparsity_tau=0.1,
            sparsity_alpha=0.01,
            sparsity_beta=0.01,
            num_runs=1,
            random_seed=42
        )
        
        results = []
        def capture_progress(run_idx, gen, total_gens, metrics):
            results.append(gen)
            
        worker.progress.connect(capture_progress)
        
        worker.start()
        timeout = time.time() + 30
        while worker.isRunning() and time.time() < timeout:
            self.app.processEvents()
            time.sleep(0.1)
        
        if worker.isRunning():
            worker.stop()
            worker.wait()
            self.fail("MOGAWorker timed out")

        self.assertTrue(len(results) > 0, "MOGAWorker did not emit any progress")

if __name__ == '__main__':
    unittest.main()
