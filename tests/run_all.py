import unittest
import sys
import os

# Ensure 'codes' is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../codes')))

def check_dependencies():
    """Check if all required packages are installed."""
    required = [
        'numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn', 
        'PyQt5', 'SALib', 'deap', 'cma', 'psutil', 'sklearn'
    ]
    missing = []
    for pkg in required:
        try:
            if pkg == 'sklearn':
                import sklearn
            else:
                __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print("\nWARNING: Missing dependencies: ", ", ".join(missing))
        print("Some tests will be skipped. Run 'pip install -r requirements_full.txt' to install them.\n")
    return missing

def run_suite():
    print("="*60)
    print("DeVana: Comprehensive Unit Test Suite")
    print("="*60)
    
    check_dependencies()
    
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print(f"Tests Run: {result.testsRun}")
    print(f"Errors: {len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*60)
    
    if result.wasSuccessful():
        print("\nPASSED: All tests passed successfully!")
        return 0
    else:
        print("\nFAILED: Some tests failed or had errors.")
        return 1

if __name__ == '__main__':
    sys.exit(run_suite())
