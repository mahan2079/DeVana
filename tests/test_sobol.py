import unittest
import sys
import os

# Add 'codes' directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../codes')))

from modules.sobol_sensitivity import perform_sobol_analysis

class TestSobolModule(unittest.TestCase):
    def setUp(self):
        # MU, LANDA_1-5, NU_1-5, A_LOW, A_UPP, F_1, F_2, OMEGA_DC, ZETA_DC
        self.main_params = [
            1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75, 0.75,
            0.05, 0.95, 100.0, 100.0, 100.0, 0.01
        ]
        
        # Exact order expected by FRF module:
        self.dva_parameter_order = [f"b{i}" for i in range(1,16)] + \
                                   [f"k{i}" for i in range(1,16)] + \
                                   [f"mu{i}" for i in range(1,4)] + \
                                   [f"c{i}" for i in range(1,16)]
        
        self.targets = {f"mass_{i}": {"peak_value_1": 1.0} for i in range(1, 6)}
        self.weights = {f"mass_{i}": {"peak_value_1": 1.0} for i in range(1, 6)}

    def test_sobol_dict_input(self):
        """Test if Sobol analysis runs with dict input"""
        dva_bounds_dict = {}
        for name in self.dva_parameter_order:
            if name in ["mu1", "k1", "b1"]:
                if name.startswith("mu"): dva_bounds_dict[name] = (0.01, 0.2)
                elif name.startswith("k"): dva_bounds_dict[name] = (0.01, 1.0)
                elif name.startswith("b"): dva_bounds_dict[name] = (0.001, 0.05)
            else:
                dva_bounds_dict[name] = 0.1

        num_samples = [4]
        results, warnings = perform_sobol_analysis(
            self.main_params, dva_bounds_dict, self.dva_parameter_order, 0, 200, 20,
            num_samples, self.targets, self.weights, visualize=False
        )
        self.assertEqual(results['S1'][0].shape[0], 3)

    def test_sobol_list_input(self):
        """Test if Sobol analysis runs with list input (verifies the bug fix)"""
        dva_bounds_list = []
        for name in self.dva_parameter_order:
            if name in ["mu1", "k1", "b1"]:
                if name.startswith("mu"): dva_bounds_list.append((name, 0.01, 0.2, False))
                elif name.startswith("k"): dva_bounds_list.append((name, 0.01, 1.0, False))
                elif name.startswith("b"): dva_bounds_list.append((name, 0.001, 0.05, False))
            else:
                dva_bounds_list.append((name, 0.1, 0.1, True))

        num_samples = [4]
        results, warnings = perform_sobol_analysis(
            self.main_params, dva_bounds_list, None, 0, 200, 20,
            num_samples, self.targets, self.weights, visualize=False
        )
        self.assertEqual(results['S1'][0].shape[0], 3)

if __name__ == '__main__':
    unittest.main()
