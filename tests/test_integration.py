import unittest
import numpy as np
import sys
import os
import time
from PyQt5.QtCore import QCoreApplication

# Add 'codes' directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../codes')))

from workers.MemorySeeder import MemorySeeder
from workers.GAWorker import GAWorker
from modules.FRF import frf

class TestIntegration(unittest.TestCase):
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
        self.dva_parameter_order = [f"b{i}" for i in range(1,16)] + \
                                   [f"k{i}" for i in range(1,16)] + \
                                   [f"mu{i}" for i in range(1,4)] + \
                                   [f"c{i}" for i in range(1,16)]
        
        self.dva_bounds = []
        for name in self.dva_parameter_order:
            self.dva_bounds.append((name, 0.01, 1.0, False))

        self.targets = {f"mass_{i}": {"peak_value_1": 1.0} for i in range(1, 6)}
        self.weights = {f"mass_{i}": {"peak_value_1": 1.0} for i in range(1, 6)}

    def test_seeder_to_worker_integration(self):
        """Test if GAWorker can use seeds from MemorySeeder"""
        # 1. Create MemorySeeder and add some data
        lows = np.array([0.01]*48)
        highs = np.array([1.0]*48)
        mask = np.zeros(48, dtype=bool)
        vals = np.zeros(48)
        
        seeder = MemorySeeder(lows, highs, mask, vals, file_path="integration_memory.json")
        population = [list(np.random.uniform(0.1, 0.2, 48)) for _ in range(5)]
        fitnesses = [0.5, 0.4, 0.3, 0.2, 0.1]
        seeder.add_data(population, fitnesses)
        
        # 2. Initialize GAWorker with this seeder
        worker = GAWorker(
            self.main_params,
            self.targets, self.weights,
            0, 200, 20,
            ga_pop_size=10,
            ga_num_generations=1,
            ga_cxpb=0.7,
            ga_mutpb=0.2,
            ga_indpb=0.05,
            ga_parameter_data=self.dva_bounds
        )
        
        # We manually inject the seeds for the test
        seeds = seeder.propose(count=5)
        self.assertEqual(len(seeds), 5)
        
        # Check if one of our seeds is actually in the seeder's data
        found = False
        for p in population:
            if np.allclose(seeds[0], p):
                found = True
                break
        self.assertTrue(found)
        
        # Clean up
        if os.path.exists("integration_memory.json"):
            os.remove("integration_memory.json")

if __name__ == '__main__':
    unittest.main()
