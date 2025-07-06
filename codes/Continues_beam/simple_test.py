import numpy as np
from beam.solver import solve_beam_vibration

# Simple test
width = 0.05
layers = [{'height': 0.01, 'E': lambda T=0: 210e9, 'rho': lambda T=0: 7800}]
L = 1.0
k_spring = 0
num_elems = 10

try:
    results = solve_beam_vibration(width, layers, L, k_spring, num_elems, t_span=(0, 0.5), num_time_points=50)
    print(f'Natural frequencies: {results["natural_frequencies_hz"][:3]}')
    print('Test completed successfully!')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc() 