# DeVana: Dynamic Vibration Absorber Optimization Framework

![DeVana Logo](Logo.png)

**Version:** V0.2.2  
**Release Date:** January 2025  
**Author:** Mahan Dashti Gohari  

---

## Overview

**DeVana** is a state-of-the-art framework for modeling, analyzing, and optimizing Dynamic Vibration Absorbers (DVAs) in complex mechanical systems. Designed for engineers and researchers, DeVana facilitates the creation of efficient, robust, and stable systems across various industries, including structural engineering, aerospace, automotive, industrial machinery and more.

The framework bridges the gap between discrete system modeling and continuous system analysis, providing a comprehensive workflow for DVA optimization. DeVana handles the critical middle phase of the design process, taking discrete models as input and producing optimized DVA parameter ranges that can be further validated in continuous system analysis.

### Key Features

- **Comprehensive Modeling**: Multi-degree-of-freedom system modeling with integrated DVAs
- **Advanced Optimization Suite**: Multiple algorithms including Genetic Algorithms, Particle Swarm Optimization, Differential Evolution, Simulated Annealing, CMA-ES, and Bayesian Optimization
- **Sensitivity Analysis**: Sobol method for sensitivity analysis and prioritizing impactful parameters
- **Statistical Analysis**: Aggregation of multiple optimization runs to identify robust parameter ranges
- **Real-Time Visualization**: Interactive dashboards for Frequency Response Functions (FRF) and optimization progress
- **Continuous Beam Analysis**: Advanced composite beam modeling with temperature-dependent material properties
- **User-Friendly Configuration**: Intuitive JSON-based configuration files for flexibility and customization

---

## Performance Criteria

DeVana optimizes DVAs based on the following:

- **Minimizing Vibrational Energy**: Reducing stress and enhancing durability
- **Resonance Control**: Managing peak magnitudes and their frequencies
- **Bandwidth Optimization**: Ensuring robustness over a range of frequencies
- **Energy Dissipation Balance**: Enhancing stability and uniform energy distribution

For technical details, refer to the [DeVana Documentation](User_Manual.pdf).

---

## Complete DVA Design Workflow

### Phase 1: Discrete System Modeling (Pre-DeVana)
- Model the mechanical system as a discrete mass-spring-damper system
- Define degrees of freedom, connectivity, and external forces
- Establish initial system parameters (mass, stiffness, damping matrices)

### Phase 2: DVA Optimization (DeVana Core)

#### Step 1: Define System Parameters
Set primary system properties such as mass, stiffness, damping, and external forces in `config/system_config.json`.

#### Step 2: Configure DVAs
Customize DVA parameters (e.g., mass, stiffness, damping) in `config/dva_config.json`.

#### Step 3: Set Performance Targets
Specify optimization goals in `config/performance_targets.json`.

#### Step 4: Analyze System
Run Frequency Response Function (FRF) analysis to assess vibrational responses and establish baseline performance.

#### Step 5: Conduct Sensitivity Analysis
Identify influential parameters using the Sobol method to prioritize optimization efforts:
- Determine which DVA parameters have the greatest impact on system performance
- Visualize parameter sensitivity across frequency ranges
- Focus optimization on the most influential parameters

#### Step 6: Optimize DVAs
Execute optimization using one or multiple algorithms:
- **Genetic Algorithm (GA)**: Multi-objective evolutionary optimization with adaptive rates
- **Particle Swarm Optimization (PSO)**: Social behavior-inspired optimization with various topology options
- **Differential Evolution (DE)**: Population-based stochastic function minimizer
- **Simulated Annealing (SA)**: Temperature-based probabilistic technique for global optimization
- **CMA-ES**: Covariance Matrix Adaptation Evolution Strategy for non-linear, non-convex optimization
- **Bayesian Optimization**: Sample-efficient optimization using probabilistic surrogate models

#### Step 7: Statistical Analysis of Results
- Run multiple optimization instances with different initial conditions
- Aggregate results to identify robust parameter ranges
- Analyze convergence patterns and solution stability
- Generate statistical distributions of optimal parameters

### Phase 3: Continuous System Implementation (Post-DeVana)
- Apply optimized DVA parameter ranges to continuous system models
- Validate performance using the integrated Continuous Beam Analysis module
- Fine-tune parameters based on continuous system response
- Implement final design in physical system

---

## Installation

### Prerequisites
- Python 3.7+
- pip (Python package installer)

### Dependencies
- NumPy: Numerical computing
- Matplotlib & Seaborn: Data visualization
- Pandas: Data manipulation
- SciPy: Scientific computing
- PyQt5: GUI framework
- SALib: Sensitivity analysis
- joblib: Parallel computing
- adjustText: Text annotation in plots

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/mahan2079/DeVana.git
   cd DeVana
   ```

2. Create a virtual environment (optional):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch the application:
   ```bash
   python codes/run.py
   ```

---

## Advanced Features

### Continuous Beam Analysis
The integrated Continuous Beam module provides:
- Multi-layer composite beam analysis
- Temperature-dependent material properties
- Automatic calculation of effective flexural rigidity and mass per unit length
- Advanced finite element analysis with Euler-Bernoulli beam elements
- Natural frequency and mode shape calculation
- Time-domain vibration analysis
- Interactive visualization with real-time animations

### Optimization Algorithm Comparison
DeVana allows side-by-side comparison of different optimization algorithms:
- Performance benchmarking across algorithms
- Convergence rate visualization
- Solution quality assessment
- Computational efficiency metrics

### Comprehensive Metrics Visualization
- Real-time optimization progress tracking
- Parameter convergence plots
- Fitness landscape visualization
- Statistical distribution of solutions
- Sensitivity analysis heatmaps

---

## Contributing

Contributions are welcome! Fork the repository, create a branch for your feature or fix, and submit a pull request.

### Guidelines
- Ensure code quality and update documentation
- Include unit tests for new features or fixes
- Follow the existing code style and architecture

---

## Contact

**Mahan Dashti Gohari**  
Email: [mahan.dashti.gohari@gmail.com](mailto:mahan.dashti.gohari@gmail.com)  
GitHub: [https://github.com/mahan2079](https://github.com/mahan2079)

---

## Acknowledgments

Thanks to all contributors and the broader research community for their insights into vibration analysis and control systems.

## License
       Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright 2025 Mahan Dashti Gohari

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
