import unittest
import sys
import os
from PyQt5.QtCore import QCoreApplication

# Add 'codes' directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../codes')))

try:
    from workers.AdaVEAWorker import AdaVEAWorker
    ADAVEA_AVAILABLE = True
except ImportError:
    ADAVEA_AVAILABLE = False

@unittest.skipIf(not ADAVEA_AVAILABLE, "DEAP or dependencies not installed")
class TestAdaVEAWorker(unittest.TestCase):
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
        # (target_values_dict, weights_dict)
        targets = {f"mass_{i}": {"peak_value_1": 1.0} for i in range(1, 6)}
        weights = {f"mass_{i}": {"peak_value_1": 1.0} for i in range(1, 6)}
        self.target_values_weights = (targets, weights)

    def test_adavea_worker_run(self):
        """Test if AdaVEAWorker can run and emit progress"""
        worker = AdaVEAWorker(
            main_system_parameters=self.main_params,
            dva_parameters=[
                (f"param_{i}", 0.0, 1.0, False, 0.0, 1.0) for i in range(48)
            ],
            target_values_weights=self.target_values_weights,
            omega_start=0, 
            omega_end=200, 
            omega_points=20,
            pop_size=10,
            generations=2,
            cxpb=0.7,
            mutpb=0.2,
            eta_c=20.0,
            eta_m=20.0,
            num_runs=1,
            random_seed=42,
            convergence_epsilon=0.001,
            convergence_window=10,
            convergence_min_gen=1,
            hv_ref_point=(1.0, 72.0, 48.0),
            heuristic_init_ratio=0.1
        )
        
        results = []
        def capture_progress(run_idx, current_gen, total_gens, metrics):
            results.append(current_gen)
            
        worker.progress.connect(capture_progress)
        
        # AdaVEAWorker.run() is synchronous within its thread context
        worker.run()

        self.assertTrue(len(results) > 0, "AdaVEAWorker did not emit any progress")
        self.assertEqual(results[0], 1, "First generation should be 1")

if __name__ == '__main__':
    unittest.main()
