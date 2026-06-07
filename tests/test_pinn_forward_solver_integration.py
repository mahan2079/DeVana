import unittest
import numpy as np
import os
import sys
import tempfile

# Ensure codes directory is in path
sys.path.append(os.path.join(os.getcwd(), 'codes'))

from workers.PINNSolver import PINNSolver

class TestPINNForwardSolverIntegration(unittest.TestCase):
    def setUp(self):
        self.main_params = (
            1.0,  # MU
            1.0, 1.0, 0.5, 0.5, 0.5,  # LANDA
            0.75, 0.75, 0.75, 0.75, 0.75,  # NU
            0.05, 0.05,  # A_LOW, A_UPP
            100.0, 100.0,  # F_1, F_2
            5000.0, 0.01  # OMEGA_DC, ZETA_DC
        )
        self.dva_params = [0.1] * 48

    def test_pinn_solver_init(self):
        """Test if PINNSolver initializes correctly."""
        solver = PINNSolver(param_dim=48, device="cpu")
        self.assertIsNotNone(solver.model)
        self.assertEqual(solver.device, "cpu")

    def test_pinn_prediction_speed(self):
        """Test if PINN prediction is fast."""
        solver = PINNSolver(param_dim=48, device="cpu")
        
        omega = np.linspace(0, 100, 1000)
        import time
        
        start_pinn = time.time()
        preds = solver.predict(np.array(self.dva_params), omega)
        end_pinn = time.time()
        
        self.assertEqual(len(preds), 1000)
        print(f"PINN Prediction time for 1000 points: {end_pinn - start_pinn:.6f}s")
        self.assertLess(end_pinn - start_pinn, 0.1) # Should be very fast

    def test_pinn_training_step(self):
        """Test the training step of the PINN solver."""
        solver = PINNSolver(param_dim=48, device="cpu")
        
        params = np.random.rand(5, 48)
        omega = np.zeros(5)
        targets = np.random.rand(5)
        
        loss = solver.train_step(params, omega, targets)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0)

    def test_save_load_weights(self):
        """Test saving and loading PINN weights."""
        solver = PINNSolver(param_dim=48, device="cpu")
        
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "PINN_Forward_FRF.pt")
            
            # Save weights
            self.assertTrue(solver.save_weights(model_path))
            self.assertTrue(os.path.exists(model_path))
            
            # Create a new solver and load weights
            new_solver = PINNSolver(param_dim=48, device="cpu")
            self.assertTrue(new_solver.load_weights(model_path))

    def test_online_learning_logic_mock(self):
        """Mock test to verify the logic of online learning refinement."""
        solver = PINNSolver(param_dim=48, device="cpu")
        
        # Simulate an optimization step
        candidate = np.array(self.dva_params)
        
        # 1. Predict with PINN
        initial_pred = solver.predict(candidate, np.array([0.0]))[0]
        
        # 2. Get ground truth (simplified)
        ground_truth = 0.5
        
        # 3. Train PINN
        loss_before = solver.train_step(candidate.reshape(1, -1), np.array([0.0]), np.array([ground_truth]))
        
        # 4. Train multiple times to see improvement
        for _ in range(50):
            solver.train_step(candidate.reshape(1, -1), np.array([0.0]), np.array([ground_truth]))
            
        final_pred = solver.predict(candidate, np.array([0.0]))[0]
        
        # Check if prediction moved closer to ground truth
        self.assertLess(abs(final_pred - ground_truth), abs(initial_pred - ground_truth))

if __name__ == '__main__':
    unittest.main()
