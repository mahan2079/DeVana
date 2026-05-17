# Particle Swarm Optimization (PSO)

## Overview
The Particle Swarm Optimization (PSO) module (`PSOWorker.py`) mimics the social behavior of bird flocking to find the optimal DVA parameters. Particles navigate the continuous parameter space influenced by their personal best experience and the global (or neighborhood) best experience.

## Advanced Features
- **Adaptive Parameters**: Inertia weight (`w`) and acceleration coefficients (`c1`, `c2`) dynamically adjust over time based on linear/nonlinear decay, swarm fitness, and diversity.
- **Topologies**: Supports multiple neighborhood topologies (Global, Ring, Von Neumann, Random) to balance exploration and exploitation.
- **Velocity Clamping**: Prevents parameter explosion by restricting maximum velocities.
- **Boundary Handling**: Multiple strategies ("absorbing", "reflecting", "invisible") when particles hit the search space bounds.
- **Quasi-Random Initialization**: Utilizes Sobol sequences for robust search space coverage during initialization.
- **Stagnation Recovery**: Detects stagnant particles and reinitializes them to prevent premature convergence.
- **Controllers**: Integrated ML Bandit and Reinforcement Learning (RL) controllers to adapt swarm behavior dynamically.

## Algorithm Flowchart

```mermaid
flowchart TD
    Start(["Start PSO"]) --> InitSwarm["Initialize Swarm <br/> (Quasi-random/Sobol or Random)"]
    InitSwarm --> InitVels["Initialize Velocities & Personal Bests"]
    InitVels --> Topology["Establish Neighborhood Topology"]
    Topology --> IterLoop{"Max Iterations <br/> Reached?"}
    
    IterLoop -- No --> CheckTermination["Check Termination Flags"]
    CheckTermination --> Adapt["Adapt w, c1, c2 <br/> (Adaptive, ML, or RL)"]
    
    Adapt --> ParticleLoop["For Each Particle"]
    
    ParticleLoop --> CalcVel["Calculate Velocity <br/> (Cognitive + Social components)"]
    CalcVel --> ClampVel["Apply Velocity Clamping & Constriction"]
    ClampVel --> UpdatePos["Update Position"]
    UpdatePos --> Bounds["Apply Boundary Handling <br/> (Absorbing, Reflecting, etc.)"]
    Bounds --> Mutate["Apply Mutation if Diversity is Low"]
    
    Mutate --> EvalParticle["Evaluate Fitness via FRF"]
    EvalParticle --> UpdatePersonalBest["Update Personal Best"]
    UpdatePersonalBest --> NextParticle{"More Particles?"}
    
    NextParticle -- Yes --> ParticleLoop
    NextParticle -- No --> UpdateGlobalBest["Update Global/Neighborhood Bests"]
    
    UpdateGlobalBest --> CheckStagnation["Check Particle Stagnation <br/> Reinitialize if needed"]
    CheckStagnation --> CalcDiversity["Calculate Swarm Diversity"]
    CalcDiversity --> UpdateControllers["Update ML/RL Controllers"]
    UpdateControllers --> EarlyStop{"Convergence or <br/> Early Stop?"}
    
    EarlyStop -- No --> IterLoop
    EarlyStop -- Yes --> OutputBest
    IterLoop -- Yes --> OutputBest
    
    OutputBest["Output Best Position & Metrics"] --> End(["End PSO"])
```
