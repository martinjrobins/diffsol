# Example: Heat equation

Lets consider a simple example, the heat equation. The heat equation is a PDE that describes how the temperature of a material changes over time. In one dimension, the heat equation is

\\[
\frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2}
\\]

where \\(u(x, t)\\) is the temperature of the material at position \\(x\\) and time \\(t\\), and \\(D\\) is the thermal diffusivity of the material. To solve this equation, we need to discretize it in space and time. We can use a finite difference method to do this.

## Finite difference method

The finite difference method is a numerical method for discretising a spatial derivative like \\(\frac{\partial^2 u}{\partial x^2}\\). It approximates this *continuous* term by a *discrete* term, in this case the multiplication of a matrix by a vector. We can use this discretisation method to convert the heat equation into a system of ODEs suitable for DiffSol.

We will not go into the details of the finite difference method here but mearly derive a single finite difference approximation for the term \\(\frac{\partial^2 u}{\partial x^2}\\), or \\(u_{xx}\\) using more compact notation. 

The central FD approximation of \\(u_{xx}\\) is:

\\[
u_{xx} \approx \frac{u(x + h) - 2u(x) + u(x-h)}{h^2}
\\]

where \\(h\\) is the spacing between points along the x-axis.

We will discretise \\(u_{xx} = 0\\) at \\(N\\) regular points along \\(x\\) from 0 to 1, given by \\(x_1\\), \\(x_2\\), ...

              +----+----+----------+----+> x
              0   x_1  x_2    ... x_N   1

Using this set of point and the discretised equation, this gives a set of \\(N\\) equations at each interior point on the domain:

\\[
\frac{v_{i+1} - 2v_i + v_{i-1}}{h^2} \text{ for } i = 1...N
\\]

where \\(v_i \approx u(x_i)\\)

We will need additional equations at \\(x=0\\) and \\(x=1\\), known as the *boundary conditions*. For this example we will use \\(u(x) = g(x)\\) at \\(x=0\\) and \\(x=1\\) (also known as a non-homogenous Dirichlet bc), so that \\(v_0 = g(0)\\), and \\(v\_{N+1} = g(1)\\), and the equation at \\(x_1\\) becomes:

\\[
\frac{v_{i+1} - 2v_i + g(0)}{h^2}
\\]

and the equation at \\(x_N\\) becomes:

\\[
\frac{g(1) - 2v_i + v_{i-1}}{h^2}
\\]

We can therefore represent the final \\(N\\) equations in matrix form like so:

\\[
\frac{1}{h^2}
\begin{bmatrix} -2      & 1      &         &   &     \\\\
 1      & -2     & 1       &       & \\\\
&\ddots & \ddots  &  \ddots &\\\\
&        & 1      &  -2     &  1     \\\\
&        &        &   1     & -2     \end{bmatrix}
\begin{bmatrix} v_1    \\\\
v_2    \\\\
\vdots \\\\
v_{N-1}\\\\
v_{N}
\end{bmatrix} + \frac{1}{h^2} \begin{bmatrix} g(0)    \\\\
0    \\\\
\vdots \\\\
0    \\\\
g(1)
\end{bmatrix}
\\]

The relevant sparse matrix here is \\(A\\), given by

\\[
A = \begin{bmatrix} -2      & 1      &         &   &     \\\\
 1      & -2     & 1       &       & \\\\
&\ddots & \ddots  &  \ddots &\\\\
&        & 1      &  -2     &  1     \\\\
&        &        &   1     & -2     \end{bmatrix}
\\]

As you can see, the number of non-zero elements grows linearly with the size \\(N\\), so a sparse matrix format is much preferred over a dense matrix holding all \\(N^2\\) elements!
The additional vector that encodes the boundary conditions is \\(b\\), given by

\\[
b = \begin{bmatrix} g(0)    \\\\
0    \\\\
\vdots \\\\
0    \\\\
g(1)
\end{bmatrix}
\\]


## Method of Lines Approximation

We can use our FD approximation of the spatial derivative to convert the heat equation into a system of ODEs. We can write the heat equation as:

\\[
\frac{du}{dt} = D \frac{d^2 u}{dx^2} \approx \frac{D}{h^2} (A u + b)
\\]

where \\(u\\) is a vector of temperatures at each point in space, \\(A\\) and \\(b\\) is the sparse matrix and vector we derived above. This is a system of ODEs that we can solve using DiffSol.

## DiffSol Implementation

```rust
use diffsol::{
    DiffSl, OdeBuilder, CraneliftModule, SparseColMat, FaerSparseLU, 
    OdeSolverMethod
};
use plotly::{
    layout::{Axis, Layout}, Plot, Surface
};
# use std::fs;
# fn main() {
type M = SparseColMat<f64>;
type LS = FaerSparseLU<f64>;
type CG = CraneliftModule;

let eqn = DiffSl::<M, CG>::compile("
    D { 1.0 }
    h { 1.0 }
    g { 0.0 }
    m { 1.0 }
    A_ij {
        (0, 0): -2.0,
        (0, 1): 1.0,
        (1, 0): 1.0,
        (1, 1): -2.0,
        (1, 2): 1.0,
        (2, 1): 1.0,
        (2, 2): -2.0,
        (2, 3): 1.0,
        (3, 2): 1.0,
        (3, 3): -2.0,
        (3, 4): 1.0,
        (4, 3): 1.0,
        (4, 4): -2.0,
        (4, 5): 1.0,
        (5, 4): 1.0,
        (5, 5): -2.0,
        (5, 6): 1.0,
        (6, 5): 1.0,
        (6, 6): -2.0,
        (6, 7): 1.0,
        (7, 6): 1.0,
        (7, 7): -2.0,
        (7, 8): 1.0,
        (8, 7): 1.0,
        (8, 8): -2.0,
        (8, 9): 1.0,
        (9, 8): 1.0,
        (9, 9): -2.0,
        (9, 10): 1.0,
        (10, 9): 1.0,
        (10, 10): -2.0,
        (10, 11): 1.0,
        (11, 10): 1.0,
        (11, 11): -2.0,
        (11, 12): 1.0,
        (12, 11): 1.0,
        (12, 12): -2.0,
        (12, 13): 1.0,
        (13, 12): 1.0,
        (13, 13): -2.0,
        (13, 14): 1.0,
        (14, 13): 1.0,
        (14, 14): -2.0,
        (14, 15): 1.0,
        (15, 14): 1.0,
        (15, 15): -2.0,
        (15, 16): 1.0,
        (16, 15): 1.0,
        (16, 16): -2.0,
        (16, 17): 1.0,
        (17, 16): 1.0,
        (17, 17): -2.0,
        (17, 18): 1.0,
        (18, 17): 1.0,
        (18, 18): -2.0,
        (18, 19): 1.0,
        (19, 18): 1.0,
        (19, 19): -2.0,
        (19, 20): 1.0,
        (20, 19): 1.0,
        (20, 20): -2.0,
    }
    b_i { 
        (0): g,
        (1:20): 0.0,
        (20): g,
    }
    u_i {
        (0:5): g,
        (5:15): g + m,
        (15:21): g,
    }
    heat_i { A_ij * u_j }
    F_i {
        D * (heat_i + b_i) / (h * h)
    }
").unwrap();


let problem = OdeBuilder::<M>::new()
    .build_from_eqn(eqn)
    .unwrap();
let times = (0..50).map(|i| i as f64).collect::<Vec<f64>>();
let mut solver = problem.bdf::<LS>().unwrap();
let sol = solver.solve_dense(&times).unwrap();

let x = (0..21).map(|i| i as f64).collect::<Vec<f64>>();
let y = times;
let z = sol.col_iter().map(|v| v.iter().copied().collect::<Vec<f64>>()).collect::<Vec<Vec<f64>>>();
let trace = Surface::new(z).x(x).y(y);
let mut plot = Plot::new();
plot.add_trace(trace);
let layout = Layout::new()
    .x_axis(Axis::new().title("x"))
    .y_axis(Axis::new().title("t"))
    .z_axis(Axis::new().title("u"));
plot.set_layout(layout);
let plot_html = plot.to_inline_html(Some("heat-equation"));
# fs::write("../src/primer/images/heat-equation.html", plot_html).expect("Unable to write file");
# }
```
{{#include images/heat-equation.html}}