# DeVana: Dynamic Vibration Absorber Optimization Framework

<div align="center">
  <img src="Logo.png" alt="DeVana Logo" width="300">
  <br />
  <p align="center">
    <b>A state-of-the-art engineering suite bridging the gap between theoretical vibration analysis and algorithmic, multi-objective DVA design.</b>
  </p>

  [![Version](https://img.shields.io/badge/version-v0.5.0-blue.svg)](https://github.com/mahan2079/DeVana)
  [![License](https://img.shields.io/badge/license-Apache%202.0-orange.svg)](LICENSE)
  [![Python](https://img.shields.io/badge/python-3.8%2B-green.svg)](https://www.python.org/)
  [![Platform](https://img.shields.io/badge/platform-win32%20%7C%20linux%20%7C%20macos-lightgrey.svg)]()
</div>

---

## 🛑 The Engineering Challenge & Research Gap

In the real world, mechanical systems—from rotary devices to robotic arms, aerospace components, and large civil structures—are **continuous systems**. Analyzing these systems for vibration control is fundamentally complex. For simple geometries (like uniform beams), elegant mathematical theories exist. However, for complex real-world geometries, engineers must either reduce them to combinations of simple geometries or rely heavily on numerical methods like Finite Element Analysis (FEM) or mesh-free methods. 

Historically, designing Dynamic Vibration Absorbers (DVAs)—which consist of masses, springs, dampers, and inerters—has relied intensely on **engineering intuition, heuristic knowledge bases, and manual trial-and-error**. There has been a distinct lack of open-source, algorithmic frameworks that can systematically handle:
1. The vast, high-dimensional design space of DVA parameters.
2. Multi-criteria objectives (e.g., minimizing cost, weight, complexity, and vibration simultaneously).
3. The transition from finding theoretical "single optimal points" to defining **manufacturable parameter ranges**.

---

## 💡 The DeVana Solution

**DeVana** was born out of comprehensive academic research to shift DVA design from an experience-based art to a rigorous, algorithmic science. 

The software operates by taking a discrete vibrational model (masses, springs, dampers, and inerters) derived from the original continuous system. By defining physical restrictions (e.g., specifying which masses or bases cannot be physically connected), DeVana creates a comprehensive, fully-coupled mathematical space. It then leverages advanced machine learning and metaheuristics to answer two fundamental engineering questions:

1. **The Optimal Topology (Q1):** What is the absolute best combination of DVA components based on user-defined criteria? DeVana systematically finds configurations that minimize the number of active parameters, minimize total manufacturing/maintenance cost, and maximize energy dissipation.
2. **The Safe Ranges (Q2):** Real-world manufacturing cannot produce infinite-precision continuous values (e.g., a spring stiffness of exactly 145.32 N/m). Given the infinite combinations of parameters, what are the **safe, reliable ranges** for each DVA parameter? DeVana performs deep statistical extraction so that *any* combination chosen within these ranges will satisfy the performance thresholds.

### The Algorithmic Paradigm

```mermaid
flowchart TD
    Start(["Real-World Continuous System"]) --> Discretize["Step 1: Discretization (Pre-DeVana)<br/>Reduce to discrete masses, springs, dampers, inerters"]
    Discretize --> Restrict["Step 2: Define Restrictions<br/>Set boundary conditions and non-connectable nodes"]
    Restrict --> DeVana{"Step 3: DeVana Optimization Engine<br/>Run GA, PSO, RL, CMA-ES, etc."}
    
    DeVana --> Q1["Answer Q1: Optimal Configuration<br/>(Min cost, max sparsity, peak performance)"]
    DeVana --> Q2["Answer Q2: Safe Parameter Ranges<br/>(Top 10% & Median 5% statistical extraction)"]
    
    Q1 --> Combine["Step 4: Final Discrete DVA Design"]
    Q2 --> Combine
    
    Combine --> Future["Step 5: Continuous Analysis (Future)<br/>Validate ranges against original continuous system"]
```

---

## 💎 Core Pillars

### 1. Advanced Optimization Suite
DeVana features a diverse library of optimization workers, each optimized for high-dimensional mechanical design spaces:
*   **Genetic Algorithms (GA)**: Adaptive crossover/mutation rates with ML-based parameter control.
*   **Particle Swarm (PSO)**: Multi-topology support with velocity clamping.
*   **Evolution Strategies (CMA-ES)**: State-of-the-art covariance matrix adaptation for non-convex landscapes.
*   **Multi-Objective (NSGA-II)**: Pareto-optimal front generation balancing vibration, sparsity, and cost.
*   **Reinforcement Learning (RL)**: Continuous policy-gradient agents for intelligent parameter tuning.
*   **Simulated Annealing (SA) & Differential Evolution (DE)**: Robust global search techniques.

### 2. Intelligent Seeding Engine
Accelerate convergence by bypassing random initialization with DeVana's proprietary seeding strategies:
*   **Neural Seeder**: Online learning via MLP ensembles to predict fitness landscapes using UCB and EI acquisition.
*   **Memory Seeder**: Cross-session persistence that learns from successful historical designs.
*   **Quasi-Random**: Sobol sequences and Latin Hypercube Sampling (LHS) for uniform space coverage.

### 3. High-Fidelity Analysis
*   **FRF Computation**: Robust Frequency Response Function solver with automatic DOF pruning and pseudo-inverse fallbacks for singular matrices.
*   **Sobol Sensitivity**: Global variance-based sensitivity analysis to identify critical design parameters.
*   **Peak & Slope Analysis**: Automated modal identification with high-precision topological prominence filtering.

---

## 🚀 Future Roadmap: Continuous System Integration

Currently, DeVana flawlessly handles highly complex discrete systems (e.g., the fully-coupled 2DOF-3DOF benchmark featured in our documentation). 

The **next major frontier** for DeVana is the seamless integration of **Continuous Analysis**. In the future, once DeVana extracts the "Safe Ranges" for the discrete DVA components, the software will automatically feed these ranges back into integrated FEM/continuous modules (such as the under-construction Continuous Beam module). This will provide an end-to-end pipeline: from real-world continuous structures, to discrete optimization, and straight back to continuous validation.

---

## 🛠 Project Structure

```text
DeVana/
├── codes/
│   ├── gui/                # Modular Mixin-based UI architecture
│   ├── workers/            # Multi-threaded optimization algorithms
│   ├── modules/            # Core physics and sensitivity engines
│   └── RL/                 # Reinforcement Learning agents
├── Documents/              # Comprehensive technical documentation & flowcharts
├── Icon/                   # Application assets
└── requirements.txt        # Environment dependencies
```

---

## 🏁 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/mahan2079/DeVana.git
   cd DeVana
   ```

2. **Set Up Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   python codes/run.py
   ```

---

## 📚 Documentation

For deep technical dives, algorithm flowcharts, mathematical models, and API references, please explore our comprehensive **[Documentation Index](Documents/INDEX.md)**.

---

## 🤝 Contributing

Contributions are what make the engineering community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## ✉️ Contact

**Mahan Dashti Gohari**  
Lead Developer & Researcher  
📧 [mahan.dashti.gohari@gmail.com](mailto:mahan.dashti.gohari@gmail.com)  
🔗 [GitHub Profile](https://github.com/mahan2079)

---

## 📜 License

Distributed under the **Apache License 2.0**. See `LICENSE` for more information.

---
<p align="center">
  Built with ❤️ for the Structural Dynamics Community
</p>