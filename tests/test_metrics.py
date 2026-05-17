import unittest
import sys
import os

# Add 'codes' directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../codes')))

import computational_metrics_new as metrics

class TestMetricsModule(unittest.TestCase):
    def test_metrics_collection(self):
        """Test basic hardware profile and resource usage collection"""
        profile = metrics.get_hardware_profile()
        self.assertIn("cpu_count", profile)
        self.assertGreater(profile["cpu_count"], 0)
        
        usage = metrics.get_resource_usage()
        self.assertIn("cpu_percent", usage)
        self.assertIn("memory_rss", usage)

if __name__ == '__main__':
    unittest.main()
