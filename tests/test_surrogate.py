import unittest
import numpy as np
import sys
import os
from PyQt5.QtCore import QCoreApplication

# Add 'codes' directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../codes')))

from workers.GAWorker import GAWorker
from workers.NeuralSurrogate import NeuralSurrogate, TORCH_AVAILABLE

class TestNeuralSurrogate(unittest.TestCase):
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
    def test_surrogate_training_and_prediction(self):
        """Test the standalone NeuralSurrogate class"""
        surrogate = NeuralSurrogate(input_dim=48, epochs=5)
        
        # Create dummy data
        X = np.random.rand(50, 48)
        y = np.random.rand(50)
        
        # Train
        surrogate.train(X, y)
        
        # Predict
        X_test = np.random.rand(10, 48)
        preds = surrogate.predict(X_test)
        
        self.assertEqual(len(preds), 10)
        self.assertTrue(np.all(preds >= 0))

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not installed")
    def test_ga_worker_with_neural_surrogate(self):
        """Test if GAWorker correctly uses the NeuralSurrogate during a run"""
        worker = GAWorker(
            self.main_params, 
            self.targets, self.weights,
            0, 200, 20, 
            ga_pop_size=10, 
            ga_num_generations=1, 
            ga_cxpb=0.7, 
            ga_mutpb=0.2, 
            ga_tol=0.01,
            ga_parameter_data=self.dva_bounds,
            use_surrogate=True, # Enable surrogate!
            surrogate_pool_factor=2.0
        )
        
        # We need to manually add some data to the surrogate to trigger screening
        # GAWorker won't use surrogate until it has enough data
        X = [list(np.random.rand(48)) for _ in range(30)]
        y = [float(val) for val in np.random.rand(30)]
        worker._update_and_train_surrogate(X, y)
        
        results = []
        def capture_update(msg):
            if "Neural Surrogate:" in msg:
                results.append(msg)
        
        worker.update.connect(capture_update)
        
        # Run one generation
        # Use run() instead of start() to avoid thread-related init issues in unit tests if possible,
        # but GAWorker is a QThread and run() is the entry point.
        # Actually, let's use run() directly for the test to avoid event loop issues.
        worker.run()
        
        # Check if the surrogate message was emitted
        self.assertTrue(any("Neural Surrogate:" in r for r in results), 
                        "GAWorker did not use Neural Surrogate screening")

if __name__ == '__main__':
    unittest.main()
