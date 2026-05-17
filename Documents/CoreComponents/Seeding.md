# Intelligent Seeding Strategies

## Overview
Population initialization (seeding) drastically impacts the convergence speed and final quality of evolutionary algorithms. DeVana offers a hierarchy of advanced seeding options to bypass pure random search and focus on high-potential regions immediately.

## Seeding Options Hierarchy

```mermaid
graph TD
    Main["DeVana Seeding Options"] --> Random["Uniform Random"]
    Main --> Sobol["Sobol Sequence (Quasi-Random)"]
    Main --> LHS["Latin Hypercube Sampling (LHS)"]
    Main --> Neural["Neural Seeder (Machine Learning)"]
    Main --> Mem["Memory Seeder (Historical)"]
    
    Neural --> UCB["Upper Confidence Bound (UCB)"]
    Neural --> EI["Expected Improvement (EI)"]
```

---

## 1. Uniform Random Seeding
The simplest approach. Values are drawn from a uniform distribution across the allowable bounds. Fast, but risks poor coverage in high-dimensional spaces.

```mermaid
flowchart TD
    Start["Start Uniform Random"] --> Bounds["Read Parameter Bounds"]
    Bounds --> Generate["Generate random numbers from uniform distribution"]
    Generate --> Map["Map numbers to physical parameter ranges"]
    Map --> Output["Return Initial Population"]
```

## 2. Sobol Sequence Seeding
Utilizes low-discrepancy Sobol sequences to cover the parameter space much more uniformly than pseudo-random numbers, preventing "clusters" and "gaps" in the initial population.

```mermaid
flowchart TD
    Start["Start Sobol"] --> Init["Initialize Sobol Generator for N dimensions"]
    Init --> Sample["Generate base-2 samples"]
    Sample --> Scale["Scale [0,1] samples to actual bounds"]
    Scale --> Enforce["Enforce fixed parameter constraints"]
    Enforce --> Output["Return Quasi-Random Population"]
```

## 3. Latin Hypercube Sampling (LHS)
Divides the range of each parameter into $N$ equal intervals (where $N$ is population size) and ensures exactly one sample is drawn from each interval. This guarantees perfectly uniform marginal distributions for every individual parameter.

```mermaid
flowchart TD
    Start["Start LHS"] --> Divide["Divide each parameter range into N intervals"]
    Divide --> Sample["Draw one random point per interval"]
    Sample --> Shuffle["Randomly shuffle and pair points across dimensions"]
    Shuffle --> Output["Return LHS Population"]
```

## 4. Neural Seeding (Online ML)
An advanced surrogate-assisted approach. An ensemble of Multi-Layer Perceptrons (MLPs) learns the fitness landscape from past evaluations. It generates a massive pool of random candidates, predicts their performance, and selects the best ones using an acquisition function.

```mermaid
flowchart TD
    Start["Start Neural Seeder"] --> Train["Train MLP Ensemble on historical (X, Y) data"]
    Train --> Pool["Generate large random candidate pool"]
    Pool --> Predict["Predict Mean (mu) and Uncertainty (sigma) for each candidate"]
    Predict --> Acq["Calculate Acquisition Score (UCB or EI)"]
    Acq --> Filter["Apply Diversity Filter (ensure distinct seeds)"]
    Filter --> Refine["(Optional) Gradient Refinement of top candidates"]
    Refine --> Output["Return ML-Optimized Population"]
```

## 5. Memory-Based Seeding
Learns across different optimization runs by saving the best performing solutions to a JSON file. Reuses top candidates with a slight Gaussian jitter to explore near proven local optima.
