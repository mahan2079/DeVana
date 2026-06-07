import unittest
import numpy as np
import sys
import os
from PyQt5.QtCore import QCoreApplication

# Add 'codes' directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../codes')))

from workers.GAWorker import GAWorker
from workers.NeuralSurrogate import TORCH_AVAILABLE

class TestSmartMutation(unittest.TestCase):
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
    def test_ga_worker_with_smart_mutation(self):
        """Test if GAWorker correctly applies smart mutation without errors"""
        worker = GAWorker(
            self.main_params, 
            self.targets, self.weights,
            0, 200, 20, 
            ga_pop_size=10, 
            ga_num_generations=2, 
            ga_cxpb=0.1,  # Low crossover to favor mutation
            ga_mutpb=1.0, # Force mutation on every individual
            ga_tol=0.01,
            ga_parameter_data=self.dva_bounds,
            use_surrogate=True, # Need surrogate for smart mutation
            use_smart_mutation=True, # Enable smart mutation
            smart_mutation_eta=0.1
        )
        
        # Manually cross the 30-point threshold to ensure surrogate is trained
        X = [list(np.random.rand(48)) for _ in range(40)]
        y = [float(val) for val in np.random.rand(40)]
        worker._update_and_train_surrogate(X, y)
        
        results = []
        worker.update.connect(lambda msg: results.append(msg))
        
        # Run one generation
        # Since mutpb is 1.0, smart mutation should be executed for every individual
        # We just check that it runs without crashing
        try:
            worker.run()
            success = True
        except Exception as e:
            success = False
            print(f"GAWorker failed with smart mutation: {e}")
            
        self.assertTrue(success, "GAWorker crashed during smart mutation")

if __name__ == '__main__':
    unittest.main()
