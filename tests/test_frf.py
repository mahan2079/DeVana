import unittest
import numpy as np
import sys
import os

# Add 'codes' directory to sys.path to allow importing modules correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../codes')))

from modules.FRF import frf, remove_zero_mass_dofs

class TestFRFModule(unittest.TestCase):
    def setUp(self):
        # Main System Parameters (17 values)
        # MU, LANDA_1-5, NU_1-5, A_LOW, A_UPP, F_1, F_2, OMEGA_DC, ZETA_DC
        self.main_params = [
            1.0, # MU
            1.0, 1.0, 0.5, 0.5, 0.5, # LANDA_1-5
            0.75, 0.75, 0.75, 0.75, 0.75, # NU_1-5
            0.05, 0.95, # A_LOW, A_UPP
            100.0, 100.0, # F_1, F_2
            100.0, 0.01 # OMEGA_DC, ZETA_DC
        ]
        
        # DVA Parameters (48 values)
        # beta_1-15, lambda_1-15, mu_1-3, nu_1-15
        self.dva_params = [0.01]*15 + [0.2]*15 + [0.1]*3 + [0.05]*15

        self.targets = {"peak_value_1": 1.0}
        self.weights = {"peak_value_1": 1.0}

    def test_frf_computation(self):
        """Test if FRF returns expected dictionary structure and non-zero response"""
        result = frf(
            self.main_params, self.dva_params, 0, 200, 50,
            self.targets, self.weights,
            self.targets, self.weights,
            self.targets, self.weights,
            self.targets, self.weights,
            self.targets, self.weights
        )
        
        self.assertIn("mass_1", result)
        self.assertIn("singular_response", result)
        self.assertGreater(result["singular_response"], 0)
        self.assertEqual(len(result["mass_1"]["magnitude"]), 50)

    def test_dof_elimination(self):
        """Test removal of DOFs with zero mass/stiffness/damping"""
        M = np.diag([1.0, 0.0, 1.0])
        C = np.diag([0.1, 0.0, 0.1])
        K = np.diag([100.0, 0.0, 100.0])
        F = np.array([1, 0, 1])
        
        M_red, C_red, K_red, F_red, active_dofs = remove_zero_mass_dofs(M, C, K, F)
        
        active_indices = np.where(active_dofs)[0]
        self.assertEqual(len(active_indices), 2)
        self.assertEqual(active_indices.tolist(), [0, 2])
        self.assertEqual(M_red.shape, (2, 2))
        self.assertEqual(F_red.shape, (2,))

    def test_interpolation_consistency(self):
        """Test if different interpolation methods provide stable singular response"""
        methods = ["cubic", "akima", "linear"]
        responses = []
        for m in methods:
            res = frf(
                self.main_params, self.dva_params, 0, 200, 50,
                self.targets, self.weights,
                self.targets, self.weights,
                self.targets, self.weights,
                self.targets, self.weights,
                self.targets, self.weights,
                interpolation_method=m
            )
            responses.append(res["singular_response"])
        
        self.assertLess(np.std(responses), 1.0)

if __name__ == '__main__':
    unittest.main()
