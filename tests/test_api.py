import unittest
from fastapi.testclient import TestClient
import sys
import os

# Ensure codes directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../codes')))

from api.main import app
from api.security import APIKeyManager

class TestDeVanaAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        # Generate a test key
        cls.test_key = "test_api_key_123"
        APIKeyManager.save_key("test_user", cls.test_key)
        cls.headers = {"X-API-Key": cls.test_key}

    def test_root_endpoint_unauthorized(self):
        """Test root endpoint without API key."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 401)

    def test_root_endpoint_authorized(self):
        """Test root endpoint with valid API key."""
        response = self.client.get("/", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "online")

    def test_physics_frf_endpoint(self):
        """Test FRF calculation endpoint."""
        payload = {
            "dva_params": {
                "mu_1": 0.1, "mu_2": 0.1, "mu_3": 0.1,
                "lambda_1_15": [1.0] * 15,
                "nu_1_15": [0.01] * 15,
                "beta_1_15": [0.0] * 15
            },
            "main_system_params": [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75, 0.75, 0.05, 0.05, 100.0, 100.0, 5000.0, 0.01],
            "omega_range": [0.1, 5.0, 50],
            "target_masses": [1, 2]
        }
        response = self.client.post("/physics/calculate-frf", json=payload, headers=self.headers)
        if response.status_code != 200:
            print(f"FRF Error: {response.json()}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("1", data["results"])
        self.assertIn("peak_positions", data["results"]["1"])
        self.assertIn("singular_response", data["results"]["1"])

    def test_optimization_start_endpoint(self):
        """Test starting an optimization task."""
        payload = {
            "algorithm": "GA",
            "pop_size": 10,
            "generations": 2,
            "dva_bounds": {"mu_1": [0.01, 0.5]},
            "fixed_parameters": []
        }
        response = self.client.post("/optimization/start", json=payload, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertIn("task_id", response.json())
        
        task_id = response.json()["task_id"]
        # Check status
        status_response = self.client.get(f"/optimization/status/{task_id}", headers=self.headers)
        self.assertEqual(status_response.status_code, 200)
        self.assertIn("status", status_response.json())

    def test_invalid_api_key(self):
        """Test endpoint with an invalid API key."""
        invalid_headers = {"X-API-Key": "wrong_key"}
        response = self.client.get("/", headers=invalid_headers)
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"], "Could not validate API Key")
