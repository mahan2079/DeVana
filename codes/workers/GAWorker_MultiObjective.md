# GAWorker: Multi-Objective Adaptation with NSGA-II

This document provides a detailed guide on how the `GAWorker` can be evolved from a single-objective genetic algorithm to a multi-objective one using the **Nondominated Sorting Genetic Algorithm II (NSGA-II)**. The explanation includes the underlying theory, data flow from the user interface, recommended Python libraries, and detailed pseudo-code.

## 1. From Single-Objective to Multi-Objective Optimization

The current `GAWorker` uses a **single-objective** approach. It combines multiple optimization criteria (e.g., performance, sparsity, cost) into a single fitness score using a weighted sum. This is effective but has a limitation: the user must decide the importance of each objective (the weights) *before* the optimization starts.

A **multi-objective** approach, by contrast, treats each criterion as a separate, competing objective. The goal is not to find a single "best" solution, but a set of optimal trade-off solutions, known as the **Pareto Front**. Each solution on the Pareto front is optimal in the sense that you cannot improve one objective without worsening at least one other.

This allows the user to see the landscape of possible solutions and make a more informed decision *after* the optimization is complete. **NSGA-II** is a highly effective and widely used algorithm for finding this Pareto front.

## 2. Recommended Python Libraries

The `DeVana` project already uses the **DEAP** (`deap`) library, which has excellent built-in support for multi-objective optimization, including a ready-to-use implementation of the NSGA-II selection algorithm. By leveraging DEAP's features, the transition to a multi-objective `GAWorker` can be made efficiently and with minimal new dependencies.

**Key DEAP components to be used:**
-   `creator` for defining a multi-objective fitness.
-   `tools.selNSGA2` for the main selection algorithm.
-   `tools.cxSimulatedBinaryBounded` and `tools.mutPolynomialBounded` for crossover and mutation, which work well with real-valued genes.

## 3. Data Flow: From GUI to `GAWorker`

Before detailing the algorithmic changes, it's important to understand how the `GAWorker` receives its configuration from the user interface. This data flow would remain largely the same in a multi-objective setup.

The process is initiated in the `GAOptimizationMixin` (`ga_mixin.py`):
1.  **User Interaction:** The user sets the optimization parameters in the "GA Hyperparameters" and "DVA Parameters" tabs of the GUI.
2.  **Data Collection on "Run GA":** When the "Run GA" button is clicked, the `run_ga` method is called. This method gathers all the necessary data:
    -   **DVA Parameter Ranges (`ga_parameter_data`):** A loop iterates through the `ga_param_table` (a `QTableWidget`). For each row, it reads the parameter name, its lower and upper bounds, and whether it is fixed. This is compiled into a list of tuples, which becomes the `ga_parameter_data` argument for the `GAWorker`.
        ```
        // Pseudo-code in ga_mixin.py
        FUNCTION get_ga_parameter_data():
            parameter_list = []
            FOR row in ga_param_table:
                name = get_text(row, "Parameter")
                low = get_value(row, "Lower Bound")
                high = get_value(row, "Upper Bound")
                is_fixed = get_checkbox_state(row, "Fixed")
                fixed_value = get_value(row, "Fixed Value")

                IF is_fixed:
                    parameter_list.add( (name, fixed_value, fixed_value, True) )
                ELSE:
                    parameter_list.add( (name, low, high, False) )
            RETURN parameter_list
        ```
    -   **Target Values and Weights (`target_values_dict`, `weights_dict`):** The application reads the target values and weights for each mass from the UI. In a multi-objective setup, the `weights_dict` would no longer be used to compute a single fitness score but could be used for post-run analysis or visualization.
    -   **System and GA Parameters:** All other settings (population size, number of generations, `omega` range, etc.) are read directly from their respective `QSpinBox`, `QDoubleSpinBox`, etc. widgets.
3.  **`GAWorker` Instantiation:** A new `GAWorker` instance is created, and all the collected data is passed to its `__init__` constructor.
4.  **Thread Execution:** The `ga_worker.start()` method is called, which in turn executes the `run()` method in a new thread.

## 4. Core Concepts and Implementation of NSGA-II

### 4.1. Detailed Non-Dominated Sorting

NSGA-II begins by sorting the population into "fronts".

**Pseudo-code for `non_dominated_sort`:**
```
FUNCTION non_dominated_sort(population):
    // Initialize data structures for each individual
    FOR each individual p in population:
        p.domination_count = 0  // Number of solutions that dominate p
        p.dominated_solutions = [] // List of solutions that p dominates

    // First pass: Compare every pair of individuals
    FOR each individual p in population:
        FOR each individual q in population:
            IF p is not q:
                IF p dominates q:
                    // If p is better than q
                    add q to p.dominated_solutions
                ELSE IF q dominates p:
                    // If q is better than p
                    p.domination_count = p.domination_count + 1

    // Identify the first front (individuals with domination_count == 0)
    fronts = [[]] // A list of fronts
    FOR each individual p in population:
        IF p.domination_count == 0:
            p.rank = 1 // Rank of the first front is 1
            add p to fronts[0]

    // Build subsequent fronts
    front_index = 0
    WHILE fronts[front_index] is not empty:
        create new empty front, next_front
        FOR each individual p in fronts[front_index]:
            FOR each individual q in p.dominated_solutions:
                q.domination_count = q.domination_count - 1
                IF q.domination_count == 0:
                    q.rank = front_index + 2
                    add q to next_front
        add next_front to fronts
        front_index = front_index + 1

    RETURN fronts (without the last empty front)
```

### 4.2. Detailed Crowding Distance Assignment

After sorting, a crowding distance is calculated for each individual in the same front. This helps maintain diversity.

**Pseudo-code for `crowding_distance_assignment`:**
```
FUNCTION crowding_distance_assignment(front):
    num_individuals = size(front)
    IF num_individuals == 0:
        RETURN

    // Initialize crowding distance for all individuals
    FOR each individual in front:
        individual.crowding_distance = 0

    num_objectives = number of objectives

    // For each objective...
    FOR each objective_index m from 0 to num_objectives - 1:
        // 1. Sort the front based on the current objective value
        sort front by objective m in ascending order

        // 2. Assign infinite distance to the boundary individuals
        front[0].crowding_distance = infinity
        front[num_individuals - 1].crowding_distance = infinity

        // 3. Get the min and max values for the current objective to normalize
        min_objective_value = front[0].objectives[m]
        max_objective_value = front[num_individuals - 1].objectives[m]
        objective_range = max_objective_value - min_objective_value

        IF objective_range == 0:
            CONTINUE // Skip if all have the same value for this objective

        // 4. Calculate distance for intermediate individuals
        FOR i FROM 1 TO num_individuals - 2:
            distance = front[i+1].objectives[m] - front[i-1].objectives[m]
            front[i].crowding_distance = front[i].crowding_distance + (distance / objective_range)
```

## 5. Proposed Changes to `GAWorker` for NSGA-II

### 5.1. Fitness and Individual Definition in DEAP

The `creator` in DEAP would be modified to handle multiple objectives.

**Python (DEAP) Code Snippet:**
```python
from deap import base, creator

# If we have 4 objectives to minimize (e.g., vibration, sparsity, cost, error)
# weights = (-1.0, -1.0, -1.0, -1.0)
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)
```

### 5.2. Multi-Objective Evaluation Function

The `evaluate_individual` function would no longer combine objectives with weights. Instead, it returns them as a tuple.

**Pseudo-code for the new `evaluate_individual`:**
```
FUNCTION evaluate_individual(individual_parameters):
    // Run the FRF analysis (this remains the same)
    results = frf(main_system_parameters, individual_parameters, ...)

    // --- Calculate each objective separately ---

    // Objective 1: Primary performance (e.g., singular response deviation)
    // We want to minimize the deviation from 1.0
    obj1_performance = abs(results['singular_response'] - 1.0)

    // Objective 2: Sparsity (sum of parameter magnitudes)
    // We want to minimize this to encourage simpler solutions
    obj2_sparsity = sum(abs(p) for p in individual_parameters)

    // Objective 3: Cost
    // We want to minimize the cost of the DVA configuration
    obj3_cost = calculate_cost_term(individual_parameters)

    // Objective 4: Percentage Error
    // We want to minimize the total percentage error across all masses and criteria
    obj4_error = calculate_percentage_error(results)

    // Return the objectives as a tuple
    RETURN (obj1_performance, obj2_sparsity, obj3_cost, obj4_error)
```

### 5.3. Toolbox Configuration for NSGA-II

The DEAP toolbox would be configured to use `tools.selNSGA2`.

**Python (DEAP) Code Snippet:**
```python
from deap import tools

toolbox = base.Toolbox()
# ... (register individual and population creation)

toolbox.register("evaluate", evaluate_individual)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUNDS_LOW, up=BOUNDS_UP, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUNDS_LOW, up=BOUNDS_UP, eta=20.0, indpb=0.1)
toolbox.register("select", tools.selNSGA2)
```

### 5.4. The Main NSGA-II Evolutionary Loop in `run()`

The main loop in the `run()` method would be structured as follows.

**Pseudo-code for the NSGA-II `run` method:**
```
FUNCTION run_nsga2():
    // 1. Initialization
    population_size = self.ga_pop_size
    num_generations = self.ga_num_generations
    
    // Create and evaluate the initial population (P_0)
    population = toolbox.population(n=population_size)
    evaluate_all(population) // Assign fitness tuples to each individual

    // --- Main Evolutionary Loop ---
    FOR gen FROM 1 TO num_generations:
        // Update GUI progress
        self.progress.emit( (gen / num_generations) * 100 )

        // 2. Create offspring population (Q_t) of size N using selection, crossover, and mutation
        // Note: selNSGA2 is a selection operator, but for creating offspring we use a tournament
        // based on rank and crowding distance for parent selection.
        offspring = toolbox.select(population, population_size) // This uses NSGA-II selection
        offspring = varAnd(offspring, toolbox, cxpb, mutpb) // Crossover and Mutation

        // 3. Evaluate the new offspring
        evaluate_all(offspring)

        // 4. Combine parent and offspring populations (R_t = P_t U Q_t)
        combined_population = population + offspring

        // 5. Perform non-dominated sort on the combined population
        // This is the core of NSGA-II selection, which is handled by selNSGA2
        // It sorts the combined population into fronts and calculates crowding distance
        
        // 6. Select the next generation's population (P_{t+1})
        // selNSGA2 selects the best N individuals from the combined population
        population = toolbox.select(combined_population, population_size)
        
        // Log statistics for the current Pareto front
        log_statistics(population, gen)

    // 7. Finalization
    // The final population contains the best set of non-dominated solutions
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

    // Emit the Pareto front solutions
    self.finished.emit(pareto_front)
```

## 6. Conclusion

Adapting `GAWorker` to use NSGA-II transforms it from a single-solution optimizer into a powerful tool for exploring design trade-offs. By treating performance, cost, and complexity as separate, competing objectives, the algorithm can generate a Pareto front of optimal solutions. This provides the end-user with a much richer and more useful set of results, enabling them to make better-informed decisions based on the trade-offs between different design goals. The use of the DEAP library makes this transition practical and efficient, as it provides robust, pre-built components for multi-objective optimization.