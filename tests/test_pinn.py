import unittest
import numpy as np
import sys
import os
import time
from PyQt5.QtCore import QCoreApplication

# Add 'codes' directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../codes')))

from workers.PINNWorker import PINNWorker, TORCH_AVAILABLE

class TestPINNIdentification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not QCoreApplication.instance():
            cls.app = QCoreApplication(sys.argv)
        else:
            cls.app = QCoreApplication.instance()

    def generate_realistic_data(self, P=2):
        """
        Generate realistic vibration data from a known M-C-K system.
        M x'' + C x' + K x = 0
        """
        t = np.linspace(0, 5, 500)
        
        # Known 2-DOF System
        # M = [1, 0; 0, 1]
        # K = [20, -10; -10, 10]
        # C = [1, -0.5; -0.5, 0.5]
        
        M_true = np.eye(P)
        K_true = np.array([[20, -10], [-10, 10]])
        C_true = np.array([[1, -0.5], [-0.5, 0.5]])
        
        # State-space formulation for simulation
        # q = [x1, x2, v1, v2]
        # q' = [v1, v2, a1, a2]
        # a = M^-1 (-Kx - Cv)
        
        from scipy.integrate import odeint
        
        def system_dynamics(q, t):
            x = q[:P]
            v = q[P:]
            a = np.linalg.inv(M_true) @ (-K_true @ x - C_true @ v)
            return np.concatenate([v, a])
        
        q0 = [1.0, 0.0, 0.0, 0.0] # Initial displacement at node 1
        sol = odeint(system_dynamics, q0, t)
        
        x_data = sol[:, :P]
        v_data = sol[:, P:]
        
        # Calculate acceleration
        a_data = np.zeros_like(x_data)
        for i in range(len(t)):
            a_data[i] = np.linalg.inv(M_true) @ (-K_true @ x_data[i] - C_true @ v_data[i])
            
        # Add some slight noise to make it realistic
        noise_level = 0.01
        x_data += noise_level * np.random.randn(*x_data.shape)
        v_data += noise_level * np.random.randn(*v_data.shape)
        a_data += noise_level * np.random.randn(*a_data.shape)
        
        return t, x_data, v_data, a_data, M_true, C_true, K_true

    def test_pinn_worker_execution(self):
        """Verify that PINNWorker executes and produces plausible results."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not installed. Skipping PINN unit test.")
            
        P = 2
        t, x, v, a, M_gt, C_true, K_gt = self.generate_realistic_data(P)
        
        # Use high epochs for a high-accuracy convergence check
        worker = PINNWorker(
            t, x, v, a, P,
            layers=7, neurons=64,
            adam_epochs=8000, 
            lbfgs_epochs=3000, 
            warmup_epochs=2000,
            topology_mask=np.ones((P, P)) # Dense
        )
        
        results = []
        def capture_finished(res):
            results.append(res)
            
        worker.finished.connect(capture_finished)
        
        # Run synchronously for unit test reliability
        worker.run()
            
        self.assertEqual(len(results), 1, "PINNWorker did not finish correctly")
        res = results[0]
        
        M_id = np.array(res["M"])
        C_id = np.array(res["C"])
        K_id = np.array(res["K"])
        
        print("\n" + "="*30)
        print("PINN IDENTIFICATION RESULTS")
        print("="*30)
        
        def calculate_percentage_diff(identified, true):
            diff = np.zeros_like(true, dtype=float)
            # Avoid division by zero
            mask = np.abs(true) > 1e-8
            diff[mask] = np.abs((identified[mask] - true[mask]) / true[mask]) * 100
            
            # For near-zero true values, just show the absolute difference as a percentage-like metric if the identified value is large
            zero_mask = ~mask
            diff[zero_mask] = np.abs(identified[zero_mask]) * 100 
            return diff
            
        M_diff = calculate_percentage_diff(M_id, M_gt)
        K_diff = calculate_percentage_diff(K_id, K_gt)
        C_diff = calculate_percentage_diff(C_id, C_true)

        print("MASS MATRIX (M):")
        print("True:\n", M_gt)
        print("Identified:\n", M_id)
        print("Difference (%):\n", np.round(M_diff, 2))
        
        print("\nSTIFFNESS MATRIX (K):")
        print("True:\n", K_gt)
        print("Identified:\n", K_id)
        print("Difference (%):\n", np.round(K_diff, 2))
        
        print("\nDAMPING MATRIX (C):")
        print("True:\n", C_true)
        print("Identified:\n", C_id)
        print("Difference (%):\n", np.round(C_diff, 2))
        print("="*30)
        
        # Assertions on physical properties
        # 1. Matrices should be symmetric (within tolerance)
        np.testing.assert_allclose(K_id, K_id.T, atol=1e-5)
        
        # 2. Mass should be diagonal (by implementation)
        self.assertTrue(np.all(np.abs(M_id - np.diag(np.diag(M_id))) < 1e-10))
        
        # 3. Check for positivity of mass and stiffness (diagonals)
        self.assertTrue(np.all(np.diag(M_id) > 0))
        self.assertTrue(np.all(np.diag(K_id) > 0))
        
        # 4. Identified matrices should have correct shape
        self.assertEqual(M_id.shape, (P, P))
        self.assertEqual(K_id.shape, (P, P))

    def test_pinn_topology_constraint(self):
        """Verify that the topology mask is strictly enforced in the identified matrices."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not installed.")
            
        P = 2
        t, x, v, a, _, _, _ = self.generate_realistic_data(P)
        
        # Force a mask where node 1 and 2 are DISCONNECTED (K[0,1] must be 0)
        mask = np.eye(P) # Only ground connections
        
        worker = PINNWorker(
            t, x, v, a, P,
            adam_epochs=10, # Very few just to check structure
            lbfgs_epochs=0,
            topology_mask=mask
        )
        
        results = []
        worker.finished.connect(lambda res: results.append(res))
        worker.run()
            
        res = results[0]
        K_id = np.array(res["K"])
        C_id = np.array(res["C"])
        
        # Check that off-diagonals are exactly zero as per mask
        self.assertEqual(K_id[0, 1], 0.0)
        self.assertEqual(K_id[1, 0], 0.0)
        self.assertEqual(C_id[0, 1], 0.0)
        self.assertEqual(C_id[1, 0], 0.0)

if __name__ == '__main__':
    unittest.main()
