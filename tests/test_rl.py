import unittest
import numpy as np
import sys
import os
import time
from PyQt5.QtCore import QCoreApplication

# Add 'codes' directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../codes')))

from RL.RLWorker import RLWorker

class TestRLWorker(unittest.TestCase):
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
        
        self.dva_bounds = []
        for name in self.dva_parameter_order:
            self.dva_bounds.append((name, 0.01, 1.0, False))

        self.targets = {f"mass_{i}": {"peak_value_1": 1.0} for i in range(1, 6)}
        self.weights = {f"mass_{i}": {"peak_value_1": 1.0} for i in range(1, 6)}

    def test_rl_worker_run(self):
        """Test if RLWorker can run and emit progress"""
        worker = RLWorker(
            self.main_params, 
            self.targets, self.weights,
            0, 200, 20,
            rl_num_episodes=2,
            rl_max_steps=5,
            rl_alpha=0.01,
            rl_gamma=0.9,
            rl_epsilon=0.1,
            rl_epsilon_min=0.01,
            rl_epsilon_decay_type="linear",
            rl_epsilon_decay=0.99,
            rl_parameter_data=self.dva_bounds
        )
        
        results = []
        def capture_progress(prog):
            results.append(prog)
            
        worker.progress.connect(capture_progress)
        
        worker.start()
        timeout = time.time() + 30
        while worker.isRunning() and time.time() < timeout:
            self.app.processEvents()
            time.sleep(0.1)
        
        if worker.isRunning():
            worker.terminate()
            worker.wait()
            self.fail("RLWorker timed out")

        self.assertTrue(len(results) > 0, "RLWorker did not emit any progress")

if __name__ == '__main__':
    unittest.main()
