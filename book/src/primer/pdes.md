# Partial Differential Equations (PDEs)

Diffsol is an ODE solver, but it can also solve PDEs. The idea is to discretize the PDE in space and time, and then solve the resulting system of ODEs. This is called the method of lines.

Discretizing a PDE is a large topic, and there are many ways to do it. Common methods include finite difference, finite volume, finite element, and spectral methods. Finite difference methods are the simplest to understand and implement, so some of the examples in this book will demonstrate this method to give you a flavour of how to solve PDEs with Diffsol. However, in general we recommend that you use another package to discretise your PDE, and then import the resulting ODE system into Diffsol for solving.

## Some useful packages

There are many packages in the Python and Julia ecosystems that can help you discretise your PDE. Here are a few, but there are many more out there:

Python
- [FEniCS](https://fenicsproject.org/): A finite element package. Uses the Unified Form Language (UFL) to specify PDEs.
- [FireDrake](https://firedrakeproject.org/): A finite element package, uses the same UFL as FEniCS.
- [FiPy](https://www.ctcms.nist.gov/fipy/): A finite volume package.
- [scikit-fdiff](https://scikit-fdiff.readthedocs.io/en/latest/): A finite difference package.

Julia:
- [MethodOfLines.jl](https://github.com/SciML/MethodOfLines.jl): A finite difference package.
- [Gridap.jl](https://github.com/gridap/Gridap.jl): A finite element package.


