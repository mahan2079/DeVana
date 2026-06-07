# DeVana GUI & Architecture Guild

## Overview
This directory contains hyper-exhaustive documentation for the DeVana Graphical User Interface (GUI) and its underlying Mixin-based architecture. DeVana utilizes a highly modular design where the `MainWindow` is composed of multiple specialized Mixins, each handling a specific domain of functionality (optimization algorithms, analysis tools, UI components).

## Master Index

### [Core Architecture: MainWindow](MainWindow.md)
Detailed breakdown of the central `MainWindow` class, its initialization sequence, and how it coordinates various Mixins and Worker threads.

### UI Mixins (Domain Specific)
Each Mixin is documented with its Purpose, UI Components, Logic, and Signal/Slot connections.

#### Foundation Mixins
- [IntroductionMixin](Mixins/introduction_mixin.md) - Welcome screen and overview.
- [MenuMixin](Mixins/menu_mixin.md) - Main menu, toolbar, and status bar orchestration.
- [SidebarMixin](Mixins/sidebar_mixin.md) - Navigation and application state switching.
- [ThemeMixin](Mixins/theme_mixin.md) - Dark/Light mode management and CSS styling.
- [InputTabsMixin](Mixins/input_mixin.md) - Global system parameters and DVA configuration tabs.

#### Optimization Mixins
- [GAOptimizationMixin](Mixins/ga_mixin.md) - Genetic Algorithm orchestration.
- [PSOMixin](Mixins/pso_mixin.md) - Particle Swarm Optimization.
- [DEOptimizationMixin](Mixins/de_mixin.md) - Differential Evolution.
- [SAOptimizationMixin](Mixins/sa_mixin.md) - Simulated Annealing.
- [CMAESOptimizationMixin](Mixins/cmaes_mixin.md) - CMA-ES Algorithm.
- [MOGAOptimizationMixin](Mixins/moga_mixin.md) - Multi-Objective Genetic Algorithms (NSGA-II).
- [AdaVEAOptimizationMixin](Mixins/adavea_mixin.md) - Adaptive Vibration Evolutionary Algorithm.
- [RLOptimizationMixin](Mixins/rl_mixin.md) - Reinforcement Learning based optimization.

#### Analysis & Machine Learning Mixins
- [FRFMixin](Mixins/frf_mixin.md) - Frequency Response Function calculations and visualization.
- [SobolAnalysisMixin](Mixins/sobol_mixin.md) - Global sensitivity analysis using Sobol indices.
- [OmegaSensitivityMixin](Mixins/omega_sensitivity_mixin.md) - Frequency-domain sensitivity analysis.
- [PINNIdentificationMixin](Mixins/pinn_mixin.md) - Physics-Informed Neural Networks for system identification.
- [AIAssistantMixin](Mixins/ai_assistant_mixin.md) - Integrated LLM-based engineering assistant.
- [ApiKeyMixin](Mixins/api_key_mixin.md) - Configuration for external API services.

#### Specialized Module Mixins
- [ContinuousBeamMixin](Mixins/beam_mixin.md) - Specialized interface for beam optimization.
- [MicrochipPageMixin](Mixins/microchip_mixin.md) - Hardware-in-the-loop or specialized controller interface.
- [ExtraOptMixin](Mixins/extra_opt_mixin.md) - Extra optimization features.
- [StochasticMixin](Mixins/stochastic_mixin.md) - Stochastic design orchestration.

### [Signal/Slot & Worker Registry](SignalSlotRegistry.md)
A comprehensive map of how UI signals trigger asynchronous Worker threads and how results are piped back to the GUI.

## Design Patterns
- **Mixin Composition**: The use of multiple inheritance to extend `MainWindow` without creating a deep inheritance hierarchy.
- **Worker Pattern**: Offloading heavy computations (optimization, physics) to `QThread` based workers to maintain UI responsiveness.
- **Dynamic Method Binding**: Used in `MainWindow.integrate_de_functionality` to dynamically attach methods from a mixin at runtime.
