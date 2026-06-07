# Objective Function Formulation

## Overview
In DeVana, the optimization goal is to find the optimal DVA parameters $\mathbf{x} = [x_1, x_2, \ldots, x_n]^\top$ bounded by $[\mathbf{x}_L, \mathbf{x}_U]$. The objective function $f(\mathbf{x})$ evaluates the quality of a solution by balancing dynamic performance, system simplicity, and cost.

## Multi-Criteria Objective Structure
The global objective function is a weighted sum of three primary components:

$$ f(\mathbf{x}) = w_{\mathrm{FRF}} f_{\mathrm{FRF}}(\mathbf{x}) + w_{\mathrm{sparsity}} f_{\mathrm{sparsity}}(\mathbf{x}) + w_{\mathrm{cost}} f_{\mathrm{cost}}(\mathbf{x}) $$

### 1. Dynamic Performance ($f_{\mathrm{FRF}}$)
Measures how closely the system's Frequency Response Function (FRF) matches the target.
- **Primary Term**: $f_{\mathrm{primary}}(\mathbf{x}) = \left| \left(\sum_{i=1}^{N_m} \mathrm{CM}_i(\mathbf{x})\right) - 1.0 \right|$
- **Error Term**: Penalizes individual deviations across structural masses.

### 2. Sparsity and Simplicity ($f_{\mathrm{sparsity}}$)
Encourages simpler designs by penalizing non-zero and active parameters using L1 regularization and activation thresholds.
$$ f_{\mathrm{sparsity}}(\mathbf{x}) = \alpha \sum_{k=1}^{N_p} |x_k| + \beta \sum_{k=1}^{N_p} \mathbb{I}(|x_k| > \delta) $$
Where $\delta$ is the activation threshold.

### 3. Cost-Benefit ($f_{\mathrm{cost}}$)
Evaluates the economic viability considering material, manufacturing, maintenance, and operational costs.
$$ f_{\mathrm{cost}}(\mathbf{x}) = \sum_{k=1}^{N_p} C_k \cdot \mathbb{I}(|x_k| > \delta) $$

## Evaluation Workflow

```mermaid
flowchart TD
    Start["Start Evaluation"] --> Input["Input Design Parameters (x)"]
    Input --> Obj["Compute Objective f("x")"]
    Obj --> Penalty["Apply Penalties (Constraints)"]
    Penalty --> Fitness["Compute Final Fitness"]
    Fitness --> Output["Return Fitness to Optimizer"]
```

#### Pseudo-code
```text
BEGIN
  EXECUTE Start Evaluation
  EXECUTE Input Design Parameters (x)
  EXECUTE Compute Objective f(
  EXECUTE )
  EXECUTE Apply Penalties (Constraints)
  EXECUTE Compute Final Fitness
  EXECUTE Return Fitness to Optimizer
END
```

## Fitness Function Hierarchy

```mermaid
graph TD
    Root["Fitness Function f("x")"] --> FRF["FRF Performance"]
    Root --> Sparsity["Sparsity & Activation"]
    Root --> Cost["Cost/Benefit Ratio"]
    
    FRF --> Primary["Primary Objective Term"]
    FRF --> Error["Percentage Error Term"]
    
    Sparsity --> L1["L1 Regularization (Sum of |x|)"]
    Sparsity --> ActPen["Activation Penalty (x > threshold)"]
    
    Cost --> Econ["Economic Cost Term"]
    Cost --> BCR["Benefit-Cost Ratio"]
```

#### Pseudo-code
```text
BEGIN
  EXECUTE Fitness Function f(
  EXECUTE )
  EXECUTE FRF Performance
  EXECUTE Sparsity & Activation
  EXECUTE Cost/Benefit Ratio
  EXECUTE Primary Objective Term
  EXECUTE Percentage Error Term
  EXECUTE L1 Regularization (Sum of |x|)
  EXECUTE Activation Penalty (x > threshold)
  EXECUTE Economic Cost Term
  EXECUTE Benefit-Cost Ratio
END
```
