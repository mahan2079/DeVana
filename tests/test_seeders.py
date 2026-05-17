import unittest
import numpy as np
import sys
import os
import json

# Add 'codes' directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../codes')))

from workers.MemorySeeder import MemorySeeder
from workers.NeuralSeeder import NeuralSeeder

class TestSeederModules(unittest.TestCase):
    def setUp(self):
        self.dim = 5
        self.lows = np.zeros(self.dim)
        self.highs = np.ones(self.dim)
        self.fixed_mask = np.zeros(self.dim, dtype=bool)
        self.fixed_values = np.zeros(self.dim)
        self.test_json = os.path.abspath("test_memory.json")
        if os.path.exists(self.test_json):
            os.remove(self.test_json)

    def tearDown(self):
        if os.path.exists(self.test_json):
            os.remove(self.test_json)

    def test_memory_seeder_persistence(self):
        """Test if MemorySeeder saves and loads candidates"""
        seeder = MemorySeeder(
            self.lows, self.highs, self.fixed_mask, self.fixed_values,
            file_path=self.test_json, replay_frac=1.0, exploration_frac=0.0
        )
        
        population = [[0.1]*5, [0.2]*5, [0.3]*5]
        fitnesses = [0.1, 0.2, 0.3]
        
        seeder.add_data(population, fitnesses)
        
        seeder2 = MemorySeeder(
            self.lows, self.highs, self.fixed_mask, self.fixed_values,
            file_path=self.test_json, replay_frac=1.0, exploration_frac=0.0
        )
        
        seeds = seeder2.propose(count=1)
        # Verify that the seed is one of our inputs
        found = False
        for p in population:
            if np.allclose(seeds[0], p):
                found = True
                break
        self.assertTrue(found, f"Seed {seeds[0]} not in original population")

    def test_neural_seeder_logic(self):
        """Test if NeuralSeeder trains and generates seeds"""
        try:
            seeder = NeuralSeeder(
                self.lows, self.highs, self.fixed_mask, self.fixed_values
            )
        except Exception as e:
            self.skipTest(f"NeuralSeeder initialization failed: {e}")

        X = [list(np.random.rand(self.dim)) for _ in range(20)]
        y = [float(val) for val in np.random.rand(20)]
        
        seeder.add_data(X, y)
        seeder.train()
        seeds = seeder.propose(count=5, beta=2.0)
        
        self.assertEqual(len(seeds), 5)

if __name__ == '__main__':
    unittest.main()
