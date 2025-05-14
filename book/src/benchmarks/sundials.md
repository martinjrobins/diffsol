# Sundials Benchmarks

To begin with we have focused on comparing against the popular [Sundials](https://github.com/LLNL/sundials) solver suite, developed by the Lawrence Livermore National Laboratory. 

## Test Problems
To choose the test problems we have used several of the examples provided in the Sundials library. The problems are:
- `robertson` : A stiff DAE system with 3 equations (2 differential and 1 algebraic). In Sundials this is part of the IDA examples 
  and is contained in the file `ida/serial/idaRoberts_dns.c`. In Sundials the problem is solved using the Sundials dense linear solver and `Sunmatrix_Dense`, in Diffsol we use the
  dense LU linear solver, dense matrices and vectors from the [nalgebra](https://nalgebra.org) library.
- `robertson_ode`: The same problem as `robertson` but in the form of an ODE. This problem has a variable size implemented 
  by duplicating the 3 original equations \\(n^2\\) times, where \\(n\\) is the size input parameter. In Sundials problem is solved using the KLU sparse linear solver and the `Sunmatrix_Sparse` matrix, and in Diffsol we use the
  same KLU solver from the [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse) library along with the [faer](https://github.com/sarah-ek/faer-rs) sparse matrix. 
  This example is part of the Sundials CVODE examples and is contained in the file `cvode/serial/cvRoberts_block_klu.c`.
- `heat2d`: A 2D heat DAE problem with boundary conditions imposed via algebraic equations. The size \\(n\\) input parameter sets the number of grid points along each dimension, so the
  total number of equations is \\(n^2\\). This is part of the IDA examples and is contained in the file `ida/serial/idaHeat2D_klu.c`. 
  In Sundials this problem is solved using the KLU sparse linear solver and the Sunmatrix_Sparse matrix, and in Diffsol we use the same KLU solver along with the faer sparse matrix.
- `foodweb`: A predator-prey problem with diffusion on a 2D grid. The size \\(n\\) input parameter sets the number of grid points along each dimension and we have 2 species, so the
  total number of equations is \\(2n^2\\) This is part of the IDA examples and is contained in the file `ida/serial/idaFoodWeb_bnd.c`.
  In Sundials the problem is solved using a banded linear solver and the `Sunmatrix_Band` matrix. Diffsol does not have a banded linear solver, so we use the KLU solver for this problem along with the faer sparse matrix.

## Method

In each case we have taken the example files from the Sundials library at version 6.7.0, compiling and linking them against the same version of the code. 
We have made minimal modifications to the files to remove all `printf` output and to change the `main` functions to named functions to allow them to be called from rust.
We have then implemented the same problem in Rust using the Diffsol library, porting the residual functions defined in the Sundials examples to Diffsol-compatible functions representing the RHS, mass matrix and jacobian multiplication functions for the problem.
We have used the outputs published in the Sundials examples as the reference outputs for the tests to ensure that the implementations are equivalent. 
The relative and absolute tolerances for the solvers were set to the same values in both implementations.

There are a number of differences between the Sundials and Diffsol implementations that may affect the performance of the solvers. The main differences are:
- The Sundials IDA solver has the problem defined as a general DAE system, while the Diffsol solver has access to the RHS and mass matrix functions separately.
  The Sundials CVODE solver has the problem defined as an ODE system and the mass is implicitly an identity matrix, and this is the same for the Diffsol implementation for the `robertson_ode` problem. 
- In the Sundials examples that use a user-defined jacobian (`robertson`, `robertson_ode`, `heat2d`), the jacobian is provided as a sparse or dense matrix. In Diffsol the jacobian is provided as a function that multiplies the jacobian by a vector,
  so Diffsol needs to do additional work to generate a jacobian matrix from this function.
- Generally the types of matrices and linear solver are matched between the two implementations (see details in the "Test Problems" section above). However, the `foodweb` problem is slightly different in that 
  it is solved in Sundials using a banded linear solver and banded matrices and the jacobian is calculated using finite differences.
  In Diffsol we use the KLU sparse linear solver and sparse matrices, and the jacobian is calculated using the jacobian function provided by the user.
- The Sundials implementations make heavy use of indexing into arrays, as does the Diffsol implementations. In Rust these indexing is bounds checked, which affects performance slightly but was not found to be a significant factor. 

Finally, we have also implemented the `robertson`, `heat2d` and `foodweb`
problems in the DiffSl language. For the `heat2d` and `foodweb` problems we
wrote out the diffusion matrix and mass matrix from the Rust implementations and
wrote the rest of the model by hand. For the `robertson` problem we wrote the
entire model by hand. 

## Results

These results were generated using Diffsol v0.5.1, and were run on a Dell PowerEdge R7525 2U rack server, with dual AMD EPYC 7343 3.2Ghz 16C CPU and 128GB Memory.

The performance of each implementation was timed and includes all setup and solve time. The exception to this is for the DiffSl implementations, where the JIT compilation for the model was not included in the timings 
(since the compilation time for the C and Rust code was also not included). 
We have presented the results in the following graphs, where the x-axis is the size of the problem \\(n\\) and the y-axis is the time taken to solve the problem relative to the time taken by the Sundials implementation 
(so \\(10^0\\) is the same time as Sundials, \\(10^1\\) is 10 times slower etc.)

### Bdf solver


![Bdf](./images/bench_bdf.svg)

The BDF solver is the same method as that used by the Sundials IDA and CVODE solvers so we expect the performance to be largely similar, and this is generally the case.
There are differences due to the implementation details for each library, and the differences in the implementations for the linear solvers and matrices as discussed above.

For the small, dense, stiff `robertson` problem the Diffsol implementation is very close and only slightly faster than Sundials (about 0.9).

For the sparse `heat2d` problem the Diffsol implementation is slower than Sundials for smaller problems (about 2) but the performance improves significantly for larger problems until it is at about 0.3.
Since this improvement for larger systems is not seen in `foodweb` or `robertson_ode` problems, it is likely due to the fact that the `heat2d` problem has a constant jacobian matrix and the Diffsol implementation has an advantage in this case.

For the `foodweb` problem the Diffsol implementation is between 1.8 - 2.1 times slower than Sundials. It is interesting that when the problem is implemented in the DiffSl language (see benchmark below) the slowdown reduces to 1.5, indicating
that much of the performance difference is likely due to the implementation details of the Rust code for the rhs, jacobian and mass matrix functions.

The Diffsol implementation of the `robertson_ode` problem ranges between 1.2 - 1.8 times slower than Sundials over the problem range.

### tr_bdf2 & esdirk34 solvers (SDIRK)

![Sdirk](./images/bench_tr_bdf2_esdirk.svg)

The low-order `tr_bdf2` solver is slower than the `bdf` solver for all the problems, perhaps due to the generally tight tolerances used (`robertson` and `robertson_ode` have tolerances of 1e-6-1e-8, `heat2d` was 1e-7 and `foodweb` was 1e-5). The `esdirk34` solver is faster than `bdf` for the `foodweb` problem, but slightly slower for the other problems.


### Bdf + DiffSl

![Bdf + DiffSl](./images/bench_bdf_diffsl.svg)

The main difference between this plot and the previous for the Bdf solver is the use of the DiffSl language for the equations, rather than Rust closures. 
The trends in each case are mostly the same, and the DiffSl implementation only has a small slowdown comparared with rust closures for most problems. For some problems, such as `foodweb`, the DiffSl implementation is actually faster than the Rust closure implementation.
This may be due to the fact that the rust closures bound-check the array indexing, while the DiffSl implementation does not.

This plot demonstrates that a DiffSL implementation can be comparable in speed to a hand-written Rust or C implementation, but much more easily wrapped and used from a high-level language like Python or R, where the equations can be passed down
to the rust solver as a simple string and then JIT compiled at runtime.
