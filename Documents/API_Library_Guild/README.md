# DeVana API and Library Guild (Master Index)

Welcome to the **API and Library Guild** documentation. This domain covers the headless execution frameworks, the REST API interface, and the core structural logic of the `devana` library.

## Domain Overview

This guild is responsible for:
1. **The DeVana REST API:** A high-performance FastAPI backend allowing remote execution of physics simulations and optimization tasks.
2. **The DeVana Python Library:** The decoupled, standalone Python package (`devana`) that encapsulates all physics, ML, systems, and utility logic, permitting integration into external scripts and environments.

## Documentation Index

- [**API Architecture**](API_Architecture.md): Detailed documentation of the FastAPI endpoints, request/response schemas, security mechanisms, and logic flows.
- [**Devana Library Reference**](Devana_Library.md): Comprehensive documentation of the `devana` package, including physics solvers, ML brains (PINN, Neural Surrogate, Seeders), sensitivity analysis, and implementation blueprints.

> **Note:** Optimization solvers are primarily documented in the Optimization Guild, but their headless integrations (via `devana.optimize`) follow the standard patterns outlined in the library reference.
