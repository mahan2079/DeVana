# Main Application Architecture

## Overview
DeVana's graphical user interface (`mainwindow.py`) is built using PyQt5 and follows a modular "Mixin" architecture. This design pattern allows for a clean separation of concerns, where each functional block (e.g., GA, PSO, FRF plotting) is encapsulated in its own mixin class.

## Mixin Architecture
The `MainWindow` class inherits from multiple mixins, each providing specialized functionality:
- **`SidebarMixin`**: Manages navigation and switching between different analysis modules.
- **`InputMixin`**: Handles user input for main system parameters and DVA bounds.
- **`GA/PSO/DE/SA/CMAES Mixins`**: Logic for setting up and launching optimization workers.
- **`FRFMixin`**: Coordinates real-time FRF plotting and peak detection visualization.
- **`SobolMixin`**: Interface for global sensitivity analysis.
- **`ThemeMixin`**: Manages the application's visual style and dark/light modes.

## Component Flowchart

```mermaid
graph TD
    App(["DeVana Application"]) --> MW["Main Window"]
    MW --> Mixins["Mixin Classes"]
    
    subgraph CoreMixins ["Core Functionality"]
        Mixins --> Input["InputMixin: Parameter Management"]
        Mixins --> FRF["FRFMixin: Visualization & Peak Analysis"]
        Mixins --> Sidebar["SidebarMixin: Navigation"]
    end
    
    subgraph WorkerMixins ["Optimization Workers"]
        Mixins --> Opt["Optimization Mixins: GA, PSO, SA, etc."]
        Opt --> Workers["Worker Threads: QThread-based"]
        Workers --> Results["Results Processing & Logging"]
    end
    
    subgraph UIComponents ["UI Components"]
        MW --> ModernTabs["ModernQTabWidget"]
        MW --> Dashboard["ResultsDashboard"]
        MW --> Plots["Matplotlib Figure Canvas"]
    end
```

## Key UI Features
- **Modern Styling**: Custom CSS and modern widgets (SidebarButtons, ModernTabs) for a professional look.
- **Real-time Feedback**: Live progress bars and status updates from worker threads.
- **Interactive Plots**: Draggable annotations and zoomable Matplotlib figures for detailed analysis.
- **Benchmarking Integration**: Real-time display of CPU and memory usage during heavy computations.
