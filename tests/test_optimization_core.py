import pytest
import threading
import time

from devana_lib.optimization.config import (
    PhysicsConfig, SparsityConfig, OptimizationConfig, 
    NSGAIIHyperparameters, TargetWeight, FrequencyRange
)
from devana_lib.optimization.controller import ExecutionController

def test_physics_config():
    freq = FrequencyRange(start=0.1, end=10.0, points=100)
    config = PhysicsConfig(
        main_system_parameters=[1.0, 2.0],
        target_values_weights=[TargetWeight([10.0], [0.5])],
        frequency_range=freq
    )
    assert config.frequency_range.points == 100

def test_optimization_config():
    freq = FrequencyRange(0.1, 10.0, 100)
    physics = PhysicsConfig([1], [TargetWeight([1], [1])], freq)
    sparsity = SparsityConfig(0.1, 0.2, 0.3)
    hyper = NSGAIIHyperparameters(pop_size=50, generations=100, cxpb=0.9, mutpb=0.1, eta_c=15.0, eta_m=20.0, indpb=0.1, num_runs=1)
    
    opt = OptimizationConfig(physics=physics, sparsity=sparsity, hyperparameters=hyper)
    assert opt.physics.frequency_range.end == 10.0
    assert opt.sparsity.tau == 0.1
    assert opt.hyperparameters.pop_size == 50

def test_execution_controller_abort():
    controller = ExecutionController()
    assert not controller.should_abort()
    controller.abort()
    assert controller.should_abort()

def test_execution_controller_pause():
    controller = ExecutionController()
    
    # Run a thread that waits if paused
    result = []
    
    def worker():
        controller.wait_if_paused()
        result.append("done")
        
    controller.pause()
    t = threading.Thread(target=worker)
    t.start()
    
    time.sleep(0.1)
    assert len(result) == 0 # Worker should be blocked
    
    controller.resume()
    t.join(timeout=1.0)
    assert len(result) == 1
    assert result[0] == "done"

def test_execution_controller_timeout():
    controller = ExecutionController()
    controller.pause()
    with pytest.raises(TimeoutError):
        controller.wait_if_paused(timeout=0.1)
