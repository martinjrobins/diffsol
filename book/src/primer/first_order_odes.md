# Explicit First Order ODEs

Ordinary Differential Equations (ODEs) are often called rate equations because they describe how the *rate of change* of a system depends on its current *state*. For example, lets assume we wish to model the growth of a population of cells within a petri dish. We could define the *state* of the system as the concentration of cells in the dish, and assign this state to a variable \\(c\\). The *rate of change* of the system would then be the rate at which the concentration of cells changes with time, which we could denote as \\(\frac{dc}{dt}\\). We know that our cells will grow at a rate proportional to the current concentration of cells, so this can be written as:

\\[
\frac{dc}{dt} = k c
\\]

where \\(k\\) is a constant that describes the growth rate of the cells. This is a first order ODE, because it involves only the first derivative of the state variable \\(c\\) with respect to time.

We can extend this further to solve multiple equations simultaineously, in order to model the rate of change of more than one quantity. For example, say we had *two* populations of cells in the same dish that grow with different rates. We could define the state of the system as the concentrations of the two cell populations, and assign these states to variables \\(c_1\\) and \\(c_2\\). could then write down both equations as:

\\[
\begin{align*}
\frac{dc_1}{dt} &= k_1 c_1  \\\\
\frac{dc_2}{dt} &= k_2 c_2
\end{align*}
\\]

and then combine them in a vector form as:

\\[
\begin{bmatrix}
\frac{dc_1}{dt} \\\\
\frac{dc_2}{dt}
\end{bmatrix} = \begin{bmatrix}
k_1 c_1 \\\\
k_2 c_2
\end{bmatrix}
\\]

By defining a new *vector* of state variables \\(\mathbf{y} = [c_1, c_2]\\) and a vector valued function \\(\mathbf{f}(\mathbf{y}, t) = \begin{bmatrix} k_1 c_1 \\\\ k_2 c_2 \end{bmatrix}\\), we are left with the standard form of a *explicit* first order ODE system:

\\[
\frac{d\mathbf{y}}{dt} = \mathbf{f}(\mathbf{y}, t)
\\]

This is an explicit equation for the derivative of the state, \\(\frac{d\mathbf{y}}{dt}\\), as a function of the state variables \\(\mathbf{y}\\) and of time \\(t\\).

We need one more piece of information to solve this system of ODEs: the initial conditions for the populations at time \\(t = 0\\). For example, if we started with a concentration of 10 for the first population and 5 for the second population, we would write:

\\[
\mathbf{y}(0) = \begin{bmatrix} 10 \\\\ 5 \end{bmatrix}
\\]

Many ODE solver libraries, like Diffsol, require users to provide their ODEs in the form of a set of explicit first order ODEs. Given both the system of ODEs and the initial conditions, the solver can then integrate the equations forward in time to find the solution \\(\mathbf{y}(t)\\). This is the general process for solving ODEs, so it is important to know how to translate your problem into a set of first order ODEs, and thus to the general form of a explicit first order ODE system shown above. In the next two sections, we will look at an example of a first order ODE system in the area of population dynamics, and then solve it using Diffsol.