# Frequency Response Function (FRF) Analysis

## Overview
The FRF module (`FRF.py`) is the computational core of DeVana. It calculates the dynamic response of multi-degree-of-freedom systems across a specified frequency range ($\omega$). To handle ill-conditioned systems, zero-mass parameters, and complex modal interactions, the module utilizes a series of highly specialized sub-algorithms.

---

## 1. Main FRF Execution Flow
The master loop orchestrates the matrix assembly, solving, and post-processing.

```mermaid
flowchart TD
    Start["Calculate FRF"] --> Unpack["Unpack System & DVA Parameters"]
    Unpack --> Matrices["Construct Mass, Damping, Stiffness Matrices"]
    Matrices --> Clean["Eliminate Zero/Inactive DOFs"]
    Clean --> FreqLoop{"Iterate over Frequencies"}
    
    FreqLoop -- Next w --> BuildH["Build Dynamic Matrix: H = -w^2*M + iw*C + K"]
    BuildH --> Solve["Solve H * X = F via Robust Solver"]
    Solve --> FreqLoop
    
    FreqLoop -- Done --> Process["Process Responses for each Mass"]
    Process --> Interpolate["Apply Smoothing & Interpolation"]
    Interpolate --> Peaks["Detect Significant Peaks"]
    Peaks --> Metrics["Calculate Bandwidth, Slopes, Area"]
    Metrics --> Aggregate["Calculate Singular Composite Response"]
```

#### Pseudo-code
```text
BEGIN
  EXECUTE Calculate FRF
  EXECUTE Unpack System & DVA Parameters
  EXECUTE Construct Mass, Damping, Stiffness Matrices
  EXECUTE Eliminate Zero/Inactive DOFs
  EXECUTE Iterate over Frequencies
  EXECUTE Build Dynamic Matrix: H = -w^2*M + iw*C + K
  EXECUTE Solve H * X = F via Robust Solver
  EXECUTE Process Responses for each Mass
  EXECUTE Apply Smoothing & Interpolation
  EXECUTE Detect Significant Peaks
  EXECUTE Calculate Bandwidth, Slopes, Area
  EXECUTE Calculate Singular Composite Response
END
```

---

## 2. Zero DOF Elimination
Systems with non-active parameters (e.g., zero mass or zero stiffness) produce singular matrices. This algorithm safely reduces the system dimensions before solving.

```mermaid
flowchart TD
    Start["Analyze System Matrices"] --> CheckM["Identify rows/cols with zero Mass"]
    CheckM --> CheckK["Identify rows/cols with zero Stiffness"]
    CheckK --> Intersection["Find DOFs where Mass AND Stiffness are zero"]
    Intersection --> Reduce["Remove inactive rows/cols from M, C, K, F"]
    Reduce --> Output["Return Well-conditioned Reduced Matrices"]
```

#### Pseudo-code
```text
BEGIN
  EXECUTE Analyze System Matrices
  EXECUTE Identify rows/cols with zero Mass
  EXECUTE Identify rows/cols with zero Stiffness
  EXECUTE Find DOFs where Mass AND Stiffness are zero
  EXECUTE Remove inactive rows/cols from M, C, K, F
  EXECUTE Return Well-conditioned Reduced Matrices
END
```

---

## 3. Robust Linear Solver
When matrices are near-singular (ill-conditioned), standard solvers fail. This multi-stage solver guarantees a numeric response.

```mermaid
flowchart TD
    Start["Solve H * X = F"] --> TryLU["Attempt standard LU/Cholesky solve"]
    TryLU -- Success --> Output["Return Displacement Vector X"]
    TryLU -- Failure --> Reg["Add tiny regularization term to diagonal of H"]
    Reg --> TryLU2["Attempt solve with regularized H"]
    TryLU2 -- Success --> Output
    TryLU2 -- Failure --> Pinv["Compute Pseudo-Inverse of H"]
    Pinv --> LeastSq["X = Pseudo-Inverse * F (Least Squares)"]
    LeastSq --> Output
```

#### Pseudo-code
```text
BEGIN
  EXECUTE Solve H * X = F
  EXECUTE Attempt standard LU/Cholesky solve
  EXECUTE Return Displacement Vector X
  EXECUTE Add tiny regularization term to diagonal of H
  EXECUTE Attempt solve with regularized H
  EXECUTE Compute Pseudo-Inverse of H
  EXECUTE X = Pseudo-Inverse * F (Least Squares)
END
```

---

## 4. Peak Detection via Prominence
Identifying true resonant peaks among numerical noise is achieved using topological prominence filtering rather than static thresholds.

```mermaid
flowchart TD
    Start["Detect Peaks"] --> Maxima["Identify all local maxima in magnitude array"]
    Maxima --> Prom["Calculate topological prominence for each maximum"]
    Prom --> Filter["Discard peaks with prominence below adaptive threshold"]
    Filter --> Sort["Sort remaining peaks by magnitude"]
    Sort --> Limit["Keep top N most significant peaks"]
    Limit --> Output["Return Valid Resonant Peaks"]
```

#### Pseudo-code
```text
BEGIN
  EXECUTE Detect Peaks
  EXECUTE Identify all local maxima in magnitude array
  EXECUTE Calculate topological prominence for each maximum
  EXECUTE Discard peaks with prominence below adaptive threshold
  EXECUTE Sort remaining peaks by magnitude
  EXECUTE Keep top N most significant peaks
  EXECUTE Return Valid Resonant Peaks
END
```

---

## 5. Interpolation and Smoothing
Raw frequency step data is interpolated to find the exact frequency of resonant peaks without requiring computationally expensive micro-stepping.

```mermaid
flowchart TD
    Start["Interpolate Data"] --> CheckN{"Are there enough data points?"}
    CheckN -- No --> Linear["Apply Basic Linear Interpolation"]
    CheckN -- Yes --> Noise{"Is data highly noisy?"}
    Noise -- Yes --> SavGol["Apply Savitzky-Golay Smoothing Filter"]
    Noise -- No --> Spline["Apply Cubic/Akima Spline Interpolation"]
    SavGol --> Spline
    Spline --> Resample["Resample at high resolution to find exact peak tip"]
    Linear --> Resample
    Resample --> Output["Return Smooth Response Curve"]
```

#### Pseudo-code
```text
BEGIN
  EXECUTE Interpolate Data
  EXECUTE Are there enough data points?
  EXECUTE Apply Basic Linear Interpolation
  EXECUTE Is data highly noisy?
  EXECUTE Apply Savitzky-Golay Smoothing Filter
  EXECUTE Apply Cubic/Akima Spline Interpolation
  EXECUTE Resample at high resolution to find exact peak tip
  EXECUTE Return Smooth Response Curve
END
```

---

## 6. Mass Data Processing
Translates raw complex vectors into engineering metrics (bandwidth, slopes, energy transfer).

```mermaid
flowchart TD
    Start["Process Mass Response"] --> Mag["Calculate Absolute Magnitude"]
    Mag --> FindPeaks["Run Peak Detection Algorithm"]
    FindPeaks --> Bandwidth["Calculate distance between primary modes (Bandwidth)"]
    Bandwidth --> Slope["Calculate magnitude slope between peaks"]
    Slope --> Simpson["Calculate Area under curve via Simpson's Rule"]
    Simpson --> Output["Return Dictionary of Performance Metrics"]
```

#### Pseudo-code
```text
BEGIN
  EXECUTE Process Mass Response
  EXECUTE Calculate Absolute Magnitude
  EXECUTE Run Peak Detection Algorithm
  EXECUTE Calculate distance between primary modes (Bandwidth)
  EXECUTE Calculate magnitude slope between peaks
  EXECUTE Calculate Area under curve via Simpson's Rule
  EXECUTE Return Dictionary of Performance Metrics
END
```
