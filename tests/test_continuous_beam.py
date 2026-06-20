import unittest
import numpy as np
import sys
import os

# Add 'codes' directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../codes')))

from Continues_beam.backend.model import BeamModel, LayerSpec, TargetSpecification
from Continues_beam.backend.optimizers import optimize_values_at_locations

class TestContinuousBeam(unittest.TestCase):
    def setUp(self):
        self.length = 1.0
        self.width = 0.05
        self.thickness = 0.01
        self.E = 210e9
        self.rho = 7800
        self.model = BeamModel(self.length, self.width, self.thickness, self.E, self.rho, num_elements=10)

    def test_beam_model_init(self):
        self.assertEqual(self.model.L, 1.0)
        self.assertEqual(self.model.N, 10)
        self.assertEqual(self.model.M.shape, (22, 22)) # 11 nodes, 2 DOFs per node
        self.assertEqual(self.model.K.shape, (22, 22))

    def test_composite_beam(self):
        layers = [
            LayerSpec(0.005, 210e9, 7800),
            LayerSpec(0.005, 70e9, 2700)
        ]
        model = BeamModel(self.length, self.width, layers=layers, num_elements=10)
        self.assertGreater(model.EI, 0)
        self.assertGreater(model.m_line, 0)

    def test_optimization(self):
        spring_locs = [0.5, 1.0]
        damper_locs = [0.5, 1.0]
        targets = [
            TargetSpecification(
                quantity="displacement",
                locations=[1.0],
                weights=[1.0],
                target_values=[0.0]
            )
        ]
        omega = np.linspace(10, 100, 5)
        
        results = optimize_values_at_locations(
            self.model, spring_locs, damper_locs, targets, omega,
            max_iters=2, population=5, seed=42
        )
        
        self.assertIn("k_points", results)
        self.assertIn("c_points", results)
        self.assertEqual(len(results["k_points"]), 2)
        self.assertEqual(len(results["c_points"]), 2)

if __name__ == '__main__':
    unittest.main()
