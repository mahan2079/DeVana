import unittest
import sys
import os
import time
from PyQt5.QtCore import QCoreApplication

# Add 'codes' directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../codes')))

from workers.DEWorker import DEWorker

class TestDEWorker(unittest.TestCase):
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

    def test_de_worker_run(self):
        """Test if DEWorker can run and emit progress"""
        worker = DEWorker(
            self.main_params, 
            self.targets, self.weights,
            0, 200, 20,
            de_pop_size=10,
            de_num_generations=2,
            de_F=0.8,
            de_CR=0.9,
            de_parameter_data=self.dva_bounds
        )
        
        results = []
        def capture_progress(gen, best_fit, diversity):
            results.append(best_fit)
            
        worker.progress.connect(capture_progress)
        
        worker.start()
        timeout = time.time() + 20
        while worker.isRunning() and time.time() < timeout:
            self.app.processEvents()
            time.sleep(0.1)
        
        if worker.isRunning():
            worker.terminate()
            worker.wait()
            self.fail("DEWorker timed out")

        self.assertTrue(len(results) > 0, "DEWorker did not emit any progress")
        self.assertLess(results[-1], 1e6)

if __name__ == '__main__':
    unittest.main()
