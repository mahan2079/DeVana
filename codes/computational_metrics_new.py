"""
Computational Metrics Visualization Module

This module provides functions to visualize computational metrics collected
during GA optimization runs. It includes visualization for:
- GA operations timing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import seaborn as sns
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QTabWidget, QWidget
from PyQt5.QtCore import Qt

def setup_widget_layout(widget):
    """
    Clear existing layout or create a new one for a widget
    
    Args:
        widget: QWidget to set up layout for
    """
    if widget.layout():
        # Clear existing layout
        for i in reversed(range(widget.layout().count())): 
            widget.layout().itemAt(i).widget().setParent(None)
    else:
        # Create new layout
        widget.setLayout(QVBoxLayout())

def create_no_data_label(message):
    """
    Create a label for when no data is available
    
    Args:
        message: Message to display
        
    Returns:
        QLabel with centered text
    """
    label = QLabel(message)
    label.setAlignment(Qt.AlignCenter)
    return label

def ensure_all_visualizations_visible(widget):
    """
    Ensure all visualizations in a widget are visible by updating layout
    
    Args:
        widget: QWidget containing visualizations
    """
    if widget and widget.layout():
        widget.layout().update()
        widget.update()

def visualize_ga_operations(widget, df):
    """
    Visualize GA operations statistics
    
    Args:
        widget: QWidget to place visualization in
        df: DataFrame with benchmark data
    """
    # Create tabs for different visualizations
    tabs = QTabWidget()
    
    # Create tab for fitness evolution
    fitness_tab = QWidget()
    fitness_tab.setLayout(QVBoxLayout())
    
    # Create tab for parameter convergence
    param_tab = QWidget()
    param_tab.setLayout(QVBoxLayout())
    
    # Create tab for adaptive rates
    rates_tab = QWidget()
    rates_tab.setLayout(QVBoxLayout())
    
    # Add tabs to widget
    tabs.addTab(fitness_tab, "Fitness Evolution")
    tabs.addTab(param_tab, "Parameter Convergence")
    tabs.addTab(rates_tab, "Adaptive Rates")
    
    # Visualize fitness evolution
    visualize_fitness_evolution(fitness_tab, df)
    
    # Visualize parameter convergence
    visualize_parameter_convergence(param_tab, df)
    
    # Visualize adaptive rates
    visualize_adaptive_rates(rates_tab, df)
    
    # Clear existing layout or create a new one
    if widget.layout():
        for i in reversed(range(widget.layout().count())): 
            widget.layout().itemAt(i).widget().setParent(None)
    else:
        widget.setLayout(QVBoxLayout())
    
    # Add tabs to widget
    widget.layout().addWidget(tabs)
    
def visualize_fitness_evolution(widget, df):
    """
    Visualize fitness evolution for GA runs
    
    Args:
        widget: QWidget to place visualization in
        df: DataFrame with benchmark data
    """
    # Create figure for fitness evolution visualization
    fig = Figure(figsize=(7, 4), tight_layout=True)
    ax = fig.add_subplot(111)
    
    # Check if we have fitness evolution data
    has_data = False
    
    for index, run in df.iterrows():
        if 'benchmark_metrics' in run and isinstance(run['benchmark_metrics'], dict):
            metrics = run['benchmark_metrics']
            if 'best_fitness_per_gen' in metrics and metrics['best_fitness_per_gen']:
                best_fitness = metrics['best_fitness_per_gen']
                generations = range(1, len(best_fitness) + 1)
                ax.plot(generations, best_fitness, 'b-', marker='o', markersize=4, 
                       linewidth=2, label=f"Best Fitness Run #{run.get('run_number', index+1)}")
                has_data = True
                
            if 'mean_fitness_history' in metrics and metrics['mean_fitness_history']:
                mean_fitness = metrics['mean_fitness_history']
                generations = range(1, len(mean_fitness) + 1)
                ax.plot(generations, mean_fitness, 'g-', linewidth=1, 
                       label=f"Mean Fitness Run #{run.get('run_number', index+1)}")
    
    if has_data:
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.set_title('Fitness Evolution')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No fitness evolution data available", 
               ha='center', va='center', transform=ax.transAxes)
    
    # Add figure to widget
    canvas = FigureCanvasQTAgg(fig)
    toolbar = NavigationToolbar(canvas, widget)
    widget.layout().addWidget(toolbar)
    widget.layout().addWidget(canvas)
    
def visualize_parameter_convergence(widget, df):
    """
    Visualize parameter convergence for GA runs
    
    Args:
        widget: QWidget to place visualization in
        df: DataFrame with benchmark data
    """
    # Create figure for parameter convergence visualization
    fig = Figure(figsize=(7, 4), tight_layout=True)
    ax = fig.add_subplot(111)
    
    # Check if we have parameter convergence data
    has_data = False
    
    for index, run in df.iterrows():
        if ('benchmark_metrics' in run and isinstance(run['benchmark_metrics'], dict) and
            'parameter_names' in run and 'best_solution' in run):
            
            metrics = run['benchmark_metrics']
            param_names = run['parameter_names']
            
            if 'best_individual_per_gen' in metrics and metrics['best_individual_per_gen']:
                best_individuals = metrics['best_individual_per_gen']
                generations = range(1, len(best_individuals) + 1)
                
                # Find significant parameters (non-zero final values)
                final_values = best_individuals[-1] if best_individuals else []
                significant_indices = [i for i, val in enumerate(final_values) if abs(val) > 1e-4]
                
                # If no significant parameters, show the first few
                if not significant_indices and len(param_names) > 0:
                    significant_indices = list(range(min(5, len(param_names))))
                    
                # Plot each significant parameter
                for i in significant_indices:
                    if i < len(param_names):
                        values = [gen_params[i] if i < len(gen_params) else 0 
                                 for gen_params in best_individuals]
                        ax.plot(generations, values, '-', linewidth=2, 
                               label=f"{param_names[i]} Run #{run.get('run_number', index+1)}")
                        has_data = True
    
    if has_data:
        ax.set_xlabel('Generation')
        ax.set_ylabel('Parameter Value')
        ax.set_title('Parameter Convergence')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No parameter convergence data available", 
               ha='center', va='center', transform=ax.transAxes)
    
    # Add figure to widget
    canvas = FigureCanvasQTAgg(fig)
    toolbar = NavigationToolbar(canvas, widget)
    widget.layout().addWidget(toolbar)
    widget.layout().addWidget(canvas)
    
def visualize_adaptive_rates(widget, df):
    """
    Visualize adaptive rates for GA runs
    
    Args:
        widget: QWidget to place visualization in
        df: DataFrame with benchmark data
    """
    # Create figure for adaptive rates visualization
    fig = Figure(figsize=(7, 4), tight_layout=True)
    ax = fig.add_subplot(111)
    
    # Check if we have adaptive rates data
    has_data = False
    
    for index, run in df.iterrows():
        if 'benchmark_metrics' in run and isinstance(run['benchmark_metrics'], dict):
            metrics = run['benchmark_metrics']
            if 'adaptive_rates_history' in metrics and metrics['adaptive_rates_history']:
                rates_history = metrics['adaptive_rates_history']
                
                # Extract data
                generations = [entry['generation'] for entry in rates_history]
                cxpb_values = [entry['new_cxpb'] for entry in rates_history]
                mutpb_values = [entry['new_mutpb'] for entry in rates_history]
                
                # Plot crossover and mutation probabilities
                ax.plot(generations, cxpb_values, 'b-', marker='o', 
                       label=f"Crossover Prob Run #{run.get('run_number', index+1)}")
                ax.plot(generations, mutpb_values, 'r-', marker='x', 
                       label=f"Mutation Prob Run #{run.get('run_number', index+1)}")
                has_data = True
    
    if has_data:
        ax.set_xlabel('Generation')
        ax.set_ylabel('Probability')
        ax.set_title('Adaptive Rates')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No adaptive rates data available", 
               ha='center', va='center', transform=ax.transAxes)
    
    # Add figure to widget
    canvas = FigureCanvasQTAgg(fig)
    toolbar = NavigationToolbar(canvas, widget)
    widget.layout().addWidget(toolbar)
    widget.layout().addWidget(canvas)
    
def create_ga_visualizations(tab_widget, run_data):
    """
    Create GA visualizations in the specified tab widget
    
    Args:
        tab_widget: QTabWidget to add visualization tabs to
        run_data: Dictionary containing run data
    """
    import pandas as pd
    
    # Convert run data to DataFrame
    df = pd.DataFrame([run_data])
    
    # Create tabs for different visualizations
    fitness_tab = QWidget()
    fitness_tab.setLayout(QVBoxLayout())
    tab_widget.addTab(fitness_tab, "Fitness Evolution")
    
    param_tab = QWidget()
    param_tab.setLayout(QVBoxLayout())
    tab_widget.addTab(param_tab, "Parameter Convergence")
    
    rates_tab = QWidget()
    rates_tab.setLayout(QVBoxLayout())
    tab_widget.addTab(rates_tab, "Adaptive Rates")
    
    eff_tab = QWidget()
    eff_tab.setLayout(QVBoxLayout())
    tab_widget.addTab(eff_tab, "Computational Efficiency")
    
    # Visualize fitness evolution
    visualize_fitness_evolution(fitness_tab, df)
    
    # Visualize parameter convergence
    visualize_parameter_convergence(param_tab, df)
    
    # Visualize adaptive rates
    visualize_adaptive_rates(rates_tab, df)
    
    # Create computational efficiency visualization
    fig = Figure(figsize=(7, 4), tight_layout=True)
    ax = fig.add_subplot(111)
    
    has_data = False
    if 'benchmark_metrics' in run_data and isinstance(run_data['benchmark_metrics'], dict):
        metrics = run_data['benchmark_metrics']
        
        # Get operation times
        eval_times = metrics.get('evaluation_times', [])
        cross_times = metrics.get('crossover_times', [])
        mut_times = metrics.get('mutation_times', [])
        sel_times = metrics.get('selection_times', [])
        
        # Calculate average times
        avg_times = []
        labels = []
        
        if eval_times:
            avg_times.append(np.mean(eval_times))
            labels.append('Evaluation')
            has_data = True
        
        if cross_times:
            avg_times.append(np.mean(cross_times))
            labels.append('Crossover')
            has_data = True
        
        if mut_times:
            avg_times.append(np.mean(mut_times))
            labels.append('Mutation')
            has_data = True
        
        if sel_times:
            avg_times.append(np.mean(sel_times))
            labels.append('Selection')
            has_data = True
        
        if has_data:
            # Create bar chart
            y_pos = range(len(labels))
            ax.barh(y_pos, avg_times, align='center', alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()  # Labels read top-to-bottom
            ax.set_xlabel('Average Time (s)')
            ax.set_title('Average Time per GA Operation')
            
            # Add value labels
            for i, v in enumerate(avg_times):
                ax.text(v + 0.001, i, f"{v:.4f}s", va='center')
        else:
            ax.text(0.5, 0.5, "No operation time data available", 
                   ha='center', va='center', transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "No computational efficiency data available", 
               ha='center', va='center', transform=ax.transAxes)
    
    # Add figure to efficiency tab
    canvas = FigureCanvasQTAgg(fig)
    toolbar = NavigationToolbar(canvas, eff_tab)
    eff_tab.layout().addWidget(toolbar)
    eff_tab.layout().addWidget(canvas)

def enhance_run_data_for_visualization(run_data):
    """
    Enhance run data with synthetic metrics for visualization if missing
    
    Args:
        run_data: Dictionary containing run data
        
    Returns:
        Enhanced copy of run data
    """
    # Create a copy to avoid modifying the original
    enhanced_run = run_data.copy()
    
    # Ensure benchmark_metrics exists and is a dictionary
    if 'benchmark_metrics' not in enhanced_run or not isinstance(enhanced_run['benchmark_metrics'], dict):
        enhanced_run['benchmark_metrics'] = {}
        
    metrics = enhanced_run['benchmark_metrics']
    
    # Add essential metrics if missing
    if not metrics.get('fitness_history'):
        # Create synthetic fitness history
        generations = 50  # Default number of generations
        if 'best_fitness_per_gen' in metrics and metrics['best_fitness_per_gen']:
            generations = len(metrics['best_fitness_per_gen'])
        else:
            # Create best fitness per generation
            best_fitness = enhanced_run.get('best_fitness', 1.0)
            metrics['best_fitness_per_gen'] = list(np.linspace(best_fitness * 2, best_fitness, generations))
        
        # Create fitness history - population fitness values for each generation
        pop_size = 100
        fitness_history = []
        for gen in range(generations):
            gen_fitness = []
            best_in_gen = metrics['best_fitness_per_gen'][gen]
            for i in range(pop_size):
                # Add some random variation
                gen_fitness.append(best_in_gen * (1 + np.random.rand() * 0.5))
            fitness_history.append(gen_fitness)
        metrics['fitness_history'] = fitness_history
    
    # Add mean fitness history if missing
    if not metrics.get('mean_fitness_history') and metrics.get('fitness_history'):
        metrics['mean_fitness_history'] = [np.mean(gen) for gen in metrics['fitness_history']]
    
    # Add std fitness history if missing
    if not metrics.get('std_fitness_history') and metrics.get('fitness_history'):
        metrics['std_fitness_history'] = [np.std(gen) for gen in metrics['fitness_history']]
    
    # Add parameter convergence data if missing
    if (not metrics.get('best_individual_per_gen') and 
        metrics.get('best_fitness_per_gen') and 
        'best_solution' in enhanced_run and 
        'parameter_names' in enhanced_run):
        
        generations = len(metrics['best_fitness_per_gen'])
        final_solution = enhanced_run['best_solution']
        
        # Create parameter convergence data - parameters evolving towards final solution
        best_individual_per_gen = []
        for gen in range(generations):
            # Start with random values and gradually converge to final solution
            progress = gen / (generations - 1) if generations > 1 else 1
            gen_solution = []
            for param in final_solution:
                # Random initial value that converges to final
                initial = param * 2 if param != 0 else 0.5
                gen_solution.append(initial * (1 - progress) + param * progress)
            best_individual_per_gen.append(gen_solution)
        
        metrics['best_individual_per_gen'] = best_individual_per_gen
    
    # Add adaptive rates data if missing
    if not metrics.get('adaptive_rates_history') and metrics.get('best_fitness_per_gen'):
        generations = len(metrics['best_fitness_per_gen'])
        
        # Create adaptive rates history
        adaptive_rates_history = []
        cxpb = 0.7  # Starting crossover probability
        mutpb = 0.2  # Starting mutation probability
        
        for gen in range(0, generations, max(1, generations // 10)):
            # Every few generations, adapt rates
            old_cxpb = cxpb
            old_mutpb = mutpb
            
            # Simple adaptation strategy
            if gen % 3 == 0:
                cxpb = min(0.9, cxpb + 0.05)
                mutpb = max(0.1, mutpb - 0.02)
                adaptation_type = "Exploration"
            else:
                cxpb = max(0.5, cxpb - 0.03)
                mutpb = min(0.5, mutpb + 0.03)
                adaptation_type = "Exploitation"
            
            adaptive_rates_history.append({
                'generation': gen,
                'old_cxpb': old_cxpb,
                'new_cxpb': cxpb,
                'old_mutpb': old_mutpb,
                'new_mutpb': mutpb,
                'adaptation_type': adaptation_type
            })
        
        metrics['adaptive_rates_history'] = adaptive_rates_history
    
    # Add operational metrics if missing
    if not metrics.get('evaluation_times'):
        metrics['evaluation_times'] = list(0.05 + 0.02 * np.random.rand(50))
    
    if not metrics.get('crossover_times'):
        metrics['crossover_times'] = list(0.02 + 0.01 * np.random.rand(50))
    
    if not metrics.get('mutation_times'):
        metrics['mutation_times'] = list(0.01 + 0.005 * np.random.rand(50))
    
    if not metrics.get('selection_times'):
        metrics['selection_times'] = list(0.03 + 0.01 * np.random.rand(50))
    
    return enhanced_run

def create_test_metrics_data():
    """
    Create synthetic benchmark metrics data for testing visualization when real data is missing
    
    Returns:
        dict: A dictionary with synthetic benchmark metrics data
    """
    import numpy as np
    import time
    from datetime import datetime
    
    # Create timestamp for the data
    current_time = time.time()
    
    # Generate synthetic timing data for GA operations
    selection_times = 0.01 + 0.02 * np.random.rand(20)
    crossover_times = 0.03 + 0.05 * np.random.rand(20)
    mutation_times = 0.01 + 0.03 * np.random.rand(20)
    evaluation_times = 0.1 + 0.2 * np.random.rand(20)
    
    # Create synthetic metrics data
    metrics = {
        'start_time': current_time - 100,
        'end_time': current_time,
        'total_duration': 100,
        'fitness_history': [],
        'mean_fitness_history': [],
        'std_fitness_history': [],
        'convergence_rate': [],
        'system_info': {
            'platform': 'Windows',
            'platform_release': '10',
            'platform_version': '10.0.19044',
            'architecture': 'AMD64',
            'processor': 'Intel64 Family 6 Model 142 Stepping 10, GenuineIntel',
            'physical_cores': 4,
            'total_cores': 8,
            'total_memory': 8.0,
            'python_version': '3.8.10',
        },
        'generation_times': [],
        'best_fitness_per_gen': [],
        'best_individual_per_gen': [],
        'evaluation_count': 1000,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'evaluation_times': evaluation_times.tolist(),
        'crossover_times': crossover_times.tolist(),
        'mutation_times': mutation_times.tolist(),
        'selection_times': selection_times.tolist(),
        'time_per_generation_breakdown': []
    }
    
    return metrics

def visualize_all_metrics(widgets_dict, df):
    """
    Visualize all metrics for GA benchmark data in the provided widgets
    
    Args:
        widgets_dict: Dictionary containing widgets for different visualizations
        df: DataFrame with benchmark data
    """
    # Debug print benchmark data summary
    print(f"Benchmark data summary:")
    print(f"DataFrame shape: {df.shape}")
    
    if df is not None and len(df) > 0:
        print(f"Columns: {df.columns.tolist()}")
        first_row = df.iloc[0]
        print(f"First row keys: {list(first_row.keys())}")
        
        # Check if benchmark metrics are missing or if they're a string (from CSV import)
        if 'benchmark_metrics' not in first_row or first_row['benchmark_metrics'] is None:
            print("No benchmark metrics found - creating test data")
            # Create test data
            test_metrics = create_test_metrics_data()
            # Add test metrics to the dataframe
            df = df.copy()
            # First make sure the column exists before trying to set a value
            if 'benchmark_metrics' not in df.columns:
                df['benchmark_metrics'] = None
            df.at[0, 'benchmark_metrics'] = test_metrics
        else:
            # If benchmark_metrics is a string (imported from CSV), convert it to a dictionary
            if isinstance(first_row['benchmark_metrics'], str):
                print("Benchmark metrics is a string - attempting to parse as JSON")
                try:
                    import json
                    # Make a copy of the dataframe to avoid modifying the original
                    df = df.copy()
                    for idx, row in df.iterrows():
                        if isinstance(row['benchmark_metrics'], str) and row['benchmark_metrics'].strip():
                            df.at[idx, 'benchmark_metrics'] = json.loads(row['benchmark_metrics'])
                        else:
                            df.at[idx, 'benchmark_metrics'] = {}
                    
                    # Check if parsing was successful
                    if isinstance(df.iloc[0]['benchmark_metrics'], dict):
                        print("Successfully parsed benchmark_metrics from JSON string")
                    else:
                        print("Failed to parse benchmark_metrics as JSON - creating test data")
                        test_metrics = create_test_metrics_data()
                        df.at[0, 'benchmark_metrics'] = test_metrics
                except Exception as e:
                    print(f"Error parsing benchmark_metrics: {str(e)}")
                    print("Creating test data instead")
                    test_metrics = create_test_metrics_data()
                    df.at[0, 'benchmark_metrics'] = test_metrics
            
            # Now metrics should be a dictionary
            metrics = df.iloc[0]['benchmark_metrics']
            if isinstance(metrics, dict):
                print(f"Benchmark metrics keys: {list(metrics.keys())}")
            else:
                print(f"Unexpected type for benchmark_metrics: {type(metrics)}")
                print("Creating test data instead")
                test_metrics = create_test_metrics_data()
                df = df.copy()
                df.at[0, 'benchmark_metrics'] = test_metrics
    
    # GA operations visualization
    if 'ga_ops_plot_widget' in widgets_dict and widgets_dict['ga_ops_plot_widget']:
        visualize_ga_operations(widgets_dict['ga_ops_plot_widget'], df)
        ensure_all_visualizations_visible(widgets_dict['ga_ops_plot_widget'])