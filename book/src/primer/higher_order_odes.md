# Higher Order ODEs

The *order* of an ODE is the highest derivative that appears in the equation. So far, we have only looked at first order ODEs, which involve only the first derivative of the state variable with respect to time. However, many physical systems are described by higher order ODEs, which involve second or higher derivatives of the state variable. A simple example of a second order ODE is the motion of a mass under the influence of gravity. The equation of motion for the mass can be written as:

\\[
\frac{d^2x}{dt^2} = -g
\\]

where \\(x\\) is the position of the mass, \\(t\\) is time, and \\(g\\) is the acceleration due to gravity. This is a second order ODE because it involves the second derivative of the position with respect to time. 

Higher order ODEs can always be rewritten as a system of first order ODEs by introducing new variables. For example, we can rewrite the second order ODE above as a system of two first order ODEs by introducing a new variable for the velocity of the mass:

\\[
\begin{align*}
\frac{dx}{dt} &= v \\\\
\frac{dv}{dt} &= -g
\end{align*}
\\]

where \\(v = \frac{dx}{dt}\\) is the velocity of the mass. This is a system of two first order ODEs, which can be written in vector form as:

\\[
\frac{d\mathbf{y}}{dt} = \mathbf{f}(\mathbf{y}, t)
\\]

where

\\[
\mathbf{y} = \begin{bmatrix} x \\\\ v \end{bmatrix}
\\]

and

\\[
\mathbf{f}(\mathbf{y}, t) = \begin{bmatrix} v \\\\ -g \end{bmatrix}
\\]

In the next section, we'll look at another example of a higher order ODE system: the spring-mass system, and solve this using Diffsol.