# Example: Heat equation

Lets consider a simple example, the heat equation. The heat equation is a PDE that describes how the temperature of a material changes over time. In one dimension, the heat equation is

\\[
\frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2}
\\]

where \\(u(x, t)\\) is the temperature of the material at position \\(x\\) and time \\(t\\), and \\(D\\) is the thermal diffusivity of the material. To solve this equation, we discretise it in space only. We can use a finite difference method to discretise the spatial derivative, and then solve the resulting system of ODEs using diffsol.

## Finite difference method

The finite difference method is a numerical method for discretising a spatial derivative like \\(\frac{\partial^2 u}{\partial x^2}\\). It approximates this *continuous* term by a *discrete* term, in this case the multiplication of a matrix by a vector. We can use this discretisation method to convert the heat equation into a system of ODEs suitable for diffsol.

We will not go into the details of the finite difference method here but merely derive a single finite difference approximation for the term \\(\frac{\partial^2 u}{\partial x^2}\\), or \\(u_{xx}\\) using more compact notation. 

The central FD approximation of \\(u_{xx}\\) is:

\\[
u_{xx} \approx \frac{u(x + h) - 2u(x) + u(x-h)}{h^2}
\\]

where \\(h\\) is the spacing between points along the x-axis.

We will discretise \\(u_{xx}\\) at \\(N\\) regular points along \\(x\\) from 0 to 1, given by \\(x_1\\), \\(x_2\\), ...

              +----+----+----------+----+> x
              0   x_1  x_2    ... x_N   1

Using this set of points and the discrete approximation, this gives a set of \\(N\\) equations at each interior point on the domain:

\\[
\frac{v_{i+1} - 2v_i + v_{i-1}}{h^2} \text{ for } i = 1...N
\\]

where \\(v_i \approx u(x_i)\\)

We will need additional equations at \\(x=0\\) and \\(x=1\\), known as the *boundary conditions*. For this example we will use \\(u(x) = g(x)\\) at \\(x=0\\) and \\(x=1\\) (also known as a Dirichlet bc), so that \\(v_0 = g(0)\\), and \\(v\_{N+1} = g(1)\\), and the equation at \\(x_1\\) becomes:

\\[
\frac{v_2 - 2v_1 + g(0)}{h^2}
\\]

and the equation at \\(x_N\\) becomes:

\\[
\frac{g(1) - 2v_N + v_{N-1}}{h^2}
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

We can use our FD approximation of the spatial derivative to convert the heat equation into a system of ODEs. Starting from our original definition of the heat equation:

\\[
\frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2}
\\]

and using our finite difference approximation and definition of the sparse matrix \\(A\\) and vector \\(b\\), this becomes:


\\[
\frac{du}{dt} = \frac{D}{h^2} (A u + b)
\\]

where \\(u\\) is a vector of temperatures at each point in space. This is a system of ODEs that we can solve using diffsol.

## Problem parameters

The example below uses \\(D = 0.1\\) and \\(N = 21\\) interior points. The boundaries are at \\(x = 0\\) and \\(x = 1\\), so the spacing is \\(h = 1/(N+1) = 1/22\\). The boundary value is \\(g = 0\\) at both ends, which is a homogeneous Dirichlet bc. Thus the vector \\(b\\) is zero in this example, but the code keeps it to show the general form.

The initial condition is a "top-hat" function. The temperature is \\(g + m = 1\\) at the middle points \\(x_6, \\ldots, x_{15}\\) and \\(g = 0\\) at all other points. Diffusion then smooths the top-hat and the heat flows out through the boundaries.

## diffsol Implementation

```rust,ignore
{{#include ../../../examples/pde-heat/src/main.rs}}
```
{{#include images/heat-equation.html}}