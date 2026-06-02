---
title: 'Gala: A Python package for galactic dynamics'
tags:
  - Python
  - astronomy
  - dynamics
authors:
  - name: Adrian M. Price-Whelan
    orcid: 0000-0003-0872-7098
    affiliation: 1
affiliations:
 - name: Princeton University
   index: 1
date: 2017
bibliography: paper.bib
---

# Summary

Galactic dynamics is the study of the kinematics and dynamics of stars and gas in galaxies. 
The Python package `Gala` provides a suite of tools for galactic dynamics, 
including gravitational potential models, numerical integration of orbits, and tools for 
working with stellar streams.

# Statement of need

Research in galactic dynamics often requires custom potential models and fast orbit integration. 
While many researchers use private C or Fortran codes, there was a need for a high-level, 
extensible Python package that integrates with the Astropy ecosystem. `Gala` fills this gap 
by providing a flexible interface for defining potentials and a fast, compiled backend for 
integrating orbits.

# State of the Field

`Gala` is built on top of `Astropy` and `NumPy`. It complements other packages like 
`galpy` by focusing on different integration methods and providing a more modular 
approach to potential definitions.

# Acknowledgements

A.M.P.W. is supported by the Princeton University Society of Fellows.

# References
