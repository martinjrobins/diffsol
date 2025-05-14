# Mass matrix

In some cases, it is necessary to include a mass matrix in the problem, such that the problem is of the form

\\[M(t) \frac{dy}{dt} = f(t, y, p).\\]

A mass matrix is useful for PDE discretisation that lead to a non-identity mass matrix, or for DAE problems that can be transformed into ODEs with a singular mass matrix.
Diffsol can handle singular and non-singular mass matrices, and the mass matrix can be time-dependent.

## Example

To illustrate the addition of a mass matrix to a problem, we will once again take the logistic equation, but this time we will add an additional variable that is set via an algebraic equation.
This system is written as

\\[\frac{dy}{dt} = r y (1 - y/K),\\]
\\[0 = y - z,\\]

where \\(z\\) is the additional variable with a solution \\(z = y\\). When this system is put in the form \\(M(t) \frac{dy}{dt} = f(t, y, p)\\), the mass matrix is

\\[M(t) = \begin{bmatrix} 1 & 0 \\\\ 0 & 0 \end{bmatrix}.\\]

Like the Jacobian, the Diffsol builder does not require the full mass matrix, instead users can provide a function that gives a GEMV (General Matrix-Vector) product of the mass matrix with a vector.

\\[m(\mathbf{v}, \mathbf{p}, t, \beta, \mathbf{y}) = M(p, t) \mathbf{v} + \beta \mathbf{y}. \\]

Thus, to specify this problem using Diffsol, we can use the [`OdeBuilder`](https://docs.rs/diffsol/latest/diffsol/ode_solver/builder/struct.OdeBuilder.html) struct and provide the functions:

 \\[f(\mathbf{y}, \mathbf{p}, t) = \begin{bmatrix} r y_0 (1 - y_0/K) \\\\ y_0 - y_1 \end{bmatrix},\\]
 \\[f'(\mathbf{y}, \mathbf{p}, t, \mathbf{v}) = \begin{bmatrix} r v_0 (1 - 2 y_0/K) \\\\ v_0 - v_1 \end{bmatrix},\\]
 \\[m(\mathbf{v}, \mathbf{p}, t, \beta, \mathbf{y}) = \begin{bmatrix} v_0 + \beta y_0 \\\\ \beta y_1 \end{bmatrix}.\\]

 where \\(f\\) is the right-hand side of the ODE, \\(f'\\) is the Jacobian of \\(f\\) multiplied by a vector, and \\(m\\) is the mass matrix multiplied by a vector.

```rust,ignore
{{#include ../../../../examples/intro-logistic-closures/src/problem_mass.rs}}
```
