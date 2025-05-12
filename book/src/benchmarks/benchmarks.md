# Benchmarks

The goal of this chapter is to compare the performance of the Diffsol implementation with other similar ode solver libraries.

The libraries we will compare against are:
- [Sundials](https://computing.llnl.gov/projects/sundials): A suite of advanced numerical solvers for ODEs and DAEs, implemented in C.
- [Diffrax](https://docs.kidger.site/diffrax/): A Python library for solving ODEs and SDEs implemented using JAX.
- [Casadi](https://web.casadi.org/): A C++ library with Python anbd MATLAB bindings, for solving ODEs and DAEs, nonlinear optimisation and algorithmic differentiation.

The comparison with Sundials will be done in Rust by wrapping C functions and comparing them to Rust implementations. The comparsion with the Python libraries (Diffrax and Casadi) will be done by wrapping Diffsol in Python using the PyO3 library, and comparing the performance of the wrapped functions. As well as benchmarking the Diffsol solvers, this also serves [as an example](https://github.com/martinjrobins/diffsol_python_benchmark) of how to wrap and use Diffsol in other languages.
