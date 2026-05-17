import unittest
import numpy as np
import sys
import os
import time
from PyQt5.QtCore import QCoreApplication

# Add 'codes' directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../codes')))

from workers.GAWorker import GAWorker
from workers.PSOWorker import PSOWorker

class TestOptimizationWorkers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not QCoreApplication.instance():
            cls.app = QCoreApplication(sys.argv)
        else:
            cls.app = QCoreApplication.instance()

    def setUp(self):
        # MU, LANDA_1-5, NU_1-5, A_LOW, A_UPP, F_1, F_2, OMEGA_DC, ZETA_DC
        self.main_params = [
            1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75, 0.75,
            0.05, 0.95, 100.0, 100.0, 100.0, 0.01
        ]
        self.dva_parameter_order = [f"b{i}" for i in range(1,16)] + \
                                   [f"k{i}" for i in range(1,16)] + \
                                   [f"mu{i}" for i in range(1,4)] + \
                                   [f"c{i}" for i in range(1,16)]
        
        self.dva_bounds = []
        for name in self.dva_parameter_order:
            self.dva_bounds.append((name, 0.01, 1.0, False))

        self.targets = {f"mass_{i}": {"peak_value_1": 1.0} for i in range(1, 6)}
        self.weights = {f"mass_{i}": {"peak_value_1": 1.0} for i in range(1, 6)}

    def test_ga_worker_run(self):
        """Test if GAWorker can run at least one generation and emit progress"""
        worker = GAWorker(
            self.main_params, 
            self.targets, self.weights,
            0, 200, 20, 10, 1, 0.7, 0.2, 0.5,
            self.dva_bounds
        )
        
        results = []
        def capture_progress(gen, best_fit, mean_fit, stats):
            results.append(best_fit)
            
        worker.progress.connect(capture_progress)
        
        # Start and wait briefly
        worker.start()
        # Give it some time to finish 1 generation
        timeout = time.time() + 10
        while worker.isRunning() and time.time() < timeout:
            self.app.processEvents()
            time.sleep(0.1)
        
        worker.stop()
        worker.wait()
        
        self.assertTrue(len(results) > 0, "GAWorker did not emit any progress")
        self.assertLess(results[0], 1e6)

    def test_pso_worker_evaluation(self):
        """Test basic fitness evaluation for PSO worker"""
        worker = PSOWorker(
            self.main_params,
            self.targets, self.weights,
            0, 200, 20,
            pso_swarm_size=10,
            pso_num_iterations=1,
            pso_parameter_data=self.dva_bounds
        )
        
        ind = np.array([0.5] * 48)
        # evaluate_particle(self, position, parameter_bounds)
        fitness = worker.evaluate_particle(ind, self.dva_bounds)
        self.assertIsInstance(fitness, float)
        self.assertGreater(fitness, 0)

if __name__ == '__main__':
    unittest.main()
